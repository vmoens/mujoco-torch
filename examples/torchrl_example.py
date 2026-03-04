#!/usr/bin/env python3
"""TorchRL integration example for mujoco-torch.

Demonstrates wrapping mujoco-torch in a **batched** TorchRL EnvBase with
``batch_size=[num_envs]``.  Each call to ``_step`` advances all environments
in parallel via ``torch.vmap``.

On GPU, wrap the vmap step with ``torch.compile`` for further speedups --
see ``examples/batched_comparison.py``.

Run:
    pip install torchrl
    python examples/torchrl_example.py
"""

import mujoco
import torch
from etils import epath
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.utils import check_env_specs
from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger

import mujoco_torch

MODEL_XML = (epath.resource_path("mujoco_torch") / "test_data" / "ant.xml").read_text()

# A simpler model with a FIXED camera for pixel rendering demos.
# The ant.xml cameras use TRACKCOM / TARGETBODY modes which require
# camera-mode-aware kinematics not yet implemented in mujoco-torch.
PIXEL_MODEL_XML = """
<mujoco model="simple_arm">
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 -1 3" diffuse="1 1 1" ambient="0.2 0.2 0.2"
           specular="0.3 0.3 0.3"/>
    <geom type="plane" size="2 2 0.01" rgba="0.4 0.4 0.4 1"/>
    <body name="upper" pos="0 0 0.5">
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.06"
            rgba="0.2 0.6 0.9 1"/>
      <body name="lower" pos="0 0 0.5">
        <joint name="elbow" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.04"
              rgba="0.9 0.3 0.2 1"/>
      </body>
    </body>
    <camera name="side" pos="1.5 -1.5 1" xyaxes="1 1 0 0 0 1"
            fovy="60"/>
  </worldbody>
  <actuator>
    <motor joint="shoulder" ctrlrange="-1 1"/>
    <motor joint="elbow"    ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


class MujocoTorchEnv(EnvBase):
    """A batched TorchRL environment backed by mujoco-torch.

    Args:
        xml_string: MuJoCo XML model string.
        num_envs: number of parallel environments (sets ``batch_size``).
        max_episode_steps: truncation horizon per environment.
        device: torch device.
        dtype: floating-point dtype for observations, actions, rewards.
        from_pixels: if *True*, render pixel observations each step using
            the pure-PyTorch ray-cast renderer.
        pixels_only: if *True* (requires ``from_pixels``), drop ``qpos`` /
            ``qvel`` from the observation spec — only pixels are returned.
        camera_id: camera index used for rendering.
        render_width: pixel observation width.
        render_height: pixel observation height.
    """

    def __init__(
        self,
        xml_string: str,
        num_envs: int = 16,
        max_episode_steps: int = 1000,
        device=None,
        dtype=torch.float64,
        from_pixels: bool = False,
        pixels_only: bool = False,
        camera_id: int = 0,
        render_width: int = 64,
        render_height: int = 64,
    ):
        super().__init__(device=device, batch_size=torch.Size([num_envs]))
        self.dtype = dtype
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only
        self.camera_id = camera_id
        self.render_width = render_width
        self.render_height = render_height

        m_mj = mujoco.MjModel.from_xml_string(xml_string)
        self._m_mj = m_mj
        self.mx = mujoco_torch.device_put(m_mj)
        if device is not None:
            self.mx = self.mx.to(device)

        nq, nv, nu = m_mj.nq, m_mj.nv, m_mj.nu

        obs_keys = {}
        if not pixels_only:
            obs_keys["qpos"] = Unbounded(shape=(num_envs, nq), dtype=dtype, device=self.device)
            obs_keys["qvel"] = Unbounded(shape=(num_envs, nv), dtype=dtype, device=self.device)
        if from_pixels:
            obs_keys["pixels"] = Bounded(
                low=0,
                high=255,
                shape=(num_envs, 3, render_height, render_width),
                dtype=torch.uint8,
                device=self.device,
            )
            self._render_precomp = mujoco_torch.precompute_render_data(self.mx)
        self.observation_spec = Composite(**obs_keys, batch_size=[num_envs])

        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(num_envs, nu),
            dtype=dtype,
            device=self.device,
        )
        self.reward_spec = Unbounded(shape=(num_envs, 1), dtype=dtype, device=self.device)

        d_mj = mujoco.MjData(m_mj)
        mujoco.mj_forward(m_mj, d_mj)
        self._dx0 = mujoco_torch.device_put(d_mj)
        if device is not None:
            self._dx0 = self._dx0.to(device)

    def _make_batch(self, n):
        return torch.stack([self._dx0.clone() for _ in range(n)])

    def _render_pixels(self):
        """Render pixel observations for every env in the batch."""
        frames = []
        for i in range(self.num_envs):
            rgb, _, _ = mujoco_torch.render(
                self.mx,
                self._dx[i],
                camera_id=self.camera_id,
                width=self.render_width,
                height=self.render_height,
                precomp=self._render_precomp,
            )
            frames.append((rgb * 255).clamp(0, 255).to(torch.uint8).permute(2, 0, 1))
        return torch.stack(frames)

    def _obs_dict(self):
        """Build the observation dict from the current state."""
        obs = {}
        if not self.pixels_only:
            obs["qpos"] = self._dx.qpos.to(self.dtype)
            obs["qvel"] = self._dx.qvel.to(self.dtype)
        if self.from_pixels:
            obs["pixels"] = self._render_pixels()
        return obs

    def _reset(self, tensordict=None, **kwargs):
        reset_mask = None
        if tensordict is not None and "_reset" in tensordict.keys():
            reset_mask = tensordict["_reset"].squeeze(-1)

        if reset_mask is None or not hasattr(self, "_dx"):
            self._dx = self._make_batch(self.num_envs)
            self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        else:
            n_reset = int(reset_mask.sum())
            if n_reset > 0:
                self._dx[reset_mask] = self._make_batch(n_reset)
                self._step_count[reset_mask] = 0

        return TensorDict(
            {
                **self._obs_dict(),
                "done": torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict):
        action = tensordict["action"].to(self.dtype)

        self._dx = self._dx.replace(ctrl=action)
        step_fn = lambda d: mujoco_torch.step(self.mx, d)  # noqa: E731
        self._dx = step_fn(self._dx[0]).unsqueeze(0) if self.num_envs == 1 else torch.vmap(step_fn)(self._dx)
        self._step_count += 1

        ctrl_cost = 0.5 * (action**2).sum(dim=-1, keepdim=True)
        reward = (-ctrl_cost).to(self.dtype)

        terminated = torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device)
        truncated = (self._step_count >= self.max_episode_steps).unsqueeze(-1)
        done = terminated | truncated

        return TensorDict(
            {
                **self._obs_dict(),
                "reward": reward,
                "done": done,
                "terminated": terminated,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed):
        torch.manual_seed(seed)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    num_envs = 16

    # --- State-only environment ---
    env = MujocoTorchEnv(MODEL_XML, num_envs=num_envs, max_episode_steps=200)
    print(f"Created batched env: {num_envs} parallel envs  (batch_size={env.batch_size})")

    print("Checking env specs ...")
    check_env_specs(env, seed=42)
    print("Env specs OK!\n")

    rollout = env.rollout(max_steps=50)
    total_reward = rollout["next", "reward"].sum(dim=0).squeeze(-1)
    print(f"Rollout: {rollout.shape[-1]} steps x {num_envs} envs")
    print(f"  Mean total reward:  {total_reward.mean():.2f}")
    print(f"  Reward std:         {total_reward.std():.2f}")
    print(f"  Final qpos[0, :3]:  {rollout[-1]['next', 'qpos'][0, :3].tolist()}")

    # --- Pixel observation environment ---
    print("\n--- Pixel mode (simple arm with FIXED camera) ---")
    num_envs_px = 4
    env_px = MujocoTorchEnv(
        PIXEL_MODEL_XML,
        num_envs=num_envs_px,
        max_episode_steps=200,
        from_pixels=True,
        pixels_only=False,
        camera_id=0,
        render_width=64,
        render_height=64,
    )
    print(f"Created pixel env: {num_envs_px} envs, 64x64 pixels")
    print(f"  Observation keys: {list(env_px.observation_spec.keys())}")

    print("Checking env specs ...")
    check_env_specs(env_px, seed=42)
    print("Env specs OK!\n")

    rollout_px = env_px.rollout(max_steps=10)
    pixels = rollout_px["next", "pixels"]
    print(f"Rollout: {rollout_px.shape[-1]} steps x {num_envs_px} envs")
    print(f"  Pixel tensor shape: {pixels.shape}")
    print(f"  Pixel range: [{pixels.min():.3f}, {pixels.max():.3f}]")

    # Save a sample frame
    try:
        import torchvision

        frame = pixels[-1, 0]  # last step, first env — (3, H, W)
        torchvision.utils.save_image(frame.float() / 255.0, "arm_render.png")
        print("  Saved arm_render.png")
    except ImportError:
        print("  (install torchvision to save a sample frame)")

    # --- Video recording with VideoRecorder ---
    print("\n--- Video recording ---")
    logger = CSVLogger(exp_name="arm_render", log_dir="videos", video_format="mp4")
    recorder = VideoRecorder(logger=logger, tag="arm_video", in_keys=["pixels"])
    env_video = TransformedEnv(
        MujocoTorchEnv(
            PIXEL_MODEL_XML,
            num_envs=1,
            max_episode_steps=200,
            from_pixels=True,
            pixels_only=False,
            camera_id=0,
            render_width=128,
            render_height=128,
        ),
        recorder,
    )
    rollout_vid = env_video.rollout(max_steps=100)
    recorder.dump()
    print(f"  Recorded {rollout_vid.shape[-1]} frames at 128x128")
    print("  Video saved to videos/")
