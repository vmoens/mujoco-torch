"""Base batched TorchRL environment backed by mujoco-torch."""

import re
from abc import abstractmethod

import mujoco
import torch
from etils import epath
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

import mujoco_torch

_TEST_DATA = epath.resource_path("mujoco_torch") / "test_data"

ENVS: dict[str, type["MujocoTorchEnv"]] = {}


def register_env(name: str):
    """Class decorator that registers an env class in the ``ENVS`` dict."""

    def decorator(cls):
        ENVS[name] = cls
        return cls

    return decorator


class MujocoTorchEnv(EnvBase):
    """A batched TorchRL environment backed by mujoco-torch.

    Subclasses must implement:
        - ``_xml_path``: classmethod returning the XML filename.
        - ``_obs_spec_dict``: returns a dict of TorchRL spec objects for the
          observation space (without the leading batch dimension).
        - ``_make_obs``: builds the observation dict from simulation data.
        - ``_compute_reward``: computes scalar reward per env.
        - ``_compute_terminated``: computes boolean termination per env.

    Args:
        num_envs: number of parallel environments (sets ``batch_size``).
        max_episode_steps: truncation horizon per environment.
        device: torch device.
        dtype: floating-point dtype for observations, actions, rewards.
        frame_skip: number of physics steps per agent action.  Defaults to
            the class-level ``FRAME_SKIP`` (1 for the base, higher for
            subclasses).
        compile_kwargs: extra keyword arguments forwarded to ``torch.compile``
            (e.g. ``mode``, ``backend``, ``fullgraph``).  Ignored when
            ``compile_step`` is False.
        from_pixels: if ``True``, include rendered pixel observations.
        pixel_only: if ``True``, drop state observations and return only pixels.
            Requires ``from_pixels=True``.
        render_width: pixel observation width.
        render_height: pixel observation height.
    """

    RESET_NOISE_SCALE = 0.01
    FRAME_SKIP = 1
    RENDER_BACKGROUND = (0.4, 0.6, 0.8)

    def __init__(
        self,
        num_envs: int = 64,
        max_episode_steps: int = 1000,
        device=None,
        dtype=torch.float64,
        compile_step: bool = False,
        compile_kwargs: dict | None = None,
        frame_skip: int | None = None,
        from_pixels: bool = False,
        pixel_only: bool = False,
        render_width: int = 64,
        render_height: int = 64,
    ):
        if frame_skip is not None:
            self.FRAME_SKIP = frame_skip
        super().__init__(device=device, batch_size=torch.Size([num_envs]))
        self.dtype = dtype
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.from_pixels = from_pixels
        self.pixel_only = pixel_only
        self.render_width = render_width
        self.render_height = render_height

        xml_string = (_TEST_DATA / self._xml_path()).read_text()
        xml_string = self._patch_xml(xml_string)
        m_mj = mujoco.MjModel.from_xml_string(xml_string)
        self._m_mj = m_mj
        self._dt = m_mj.opt.timestep * self.FRAME_SKIP
        self.mx = mujoco_torch.device_put(m_mj)
        if device is not None:
            self.mx = self.mx.to(device)

        nu = m_mj.nu

        obs_keys = self._obs_spec_dict(num_envs, dtype, self.device)
        if from_pixels:
            pixel_spec = Bounded(
                low=0,
                high=255,
                shape=(num_envs, render_height, render_width, 3),
                dtype=torch.uint8,
                device=self.device,
            )
            if pixel_only:
                obs_keys = {"pixels": pixel_spec}
            else:
                obs_keys["pixels"] = pixel_spec
        self.observation_spec = Composite(**obs_keys, batch_size=[num_envs])

        low, high = self._action_range()
        self.action_spec = Bounded(
            low=low,
            high=high,
            shape=(num_envs, nu),
            dtype=dtype,
            device=self.device,
        )
        self.reward_spec = Unbounded(shape=(num_envs, 1), dtype=dtype, device=self.device)

        d_mj = mujoco.MjData(m_mj)
        mujoco.mj_forward(m_mj, d_mj)
        dx0 = mujoco_torch.device_put(d_mj)
        if device is not None:
            dx0 = dx0.to(device)
        # Run one step so all dtypes match what vmap(step) produces.
        dx0 = mujoco_torch.step(self.mx, dx0)
        self._dx0 = dx0
        self._render_precomp = mujoco_torch.precompute_render_data(self.mx)

        _step_fn = lambda d: mujoco_torch.step(self.mx, d)  # noqa: E731
        frame_skip = self.FRAME_SKIP
        _compile_kwargs = compile_kwargs or {}
        _vmap_step = torch.vmap(_step_fn)
        if num_envs == 1:
            def _multi_step(d):
                for _ in range(frame_skip):
                    d = _step_fn(d)
                return d
            self._physics_step = torch.compile(_multi_step, **_compile_kwargs) if compile_step else _multi_step
            self._single_env = True
        else:
            def _vmap_multi_step(d):
                for _ in range(frame_skip):
                    d = _vmap_step(d)
                return d
            self._physics_step = torch.compile(_vmap_multi_step, **_compile_kwargs) if compile_step else _vmap_multi_step
            self._single_env = False

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def _xml_path(cls) -> str:
        """Return the XML filename inside ``mujoco_torch/test_data/``."""
        ...

    @staticmethod
    @abstractmethod
    def _obs_spec_dict(num_envs: int, dtype: torch.dtype, device: torch.device) -> dict:
        """Return a dict of TorchRL specs for the observation space."""
        ...

    @abstractmethod
    def _make_obs(self) -> dict:
        """Build observation dict from ``self._dx``."""
        ...

    @abstractmethod
    def _compute_reward(self, qpos_before: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute per-env reward.  Shape: ``(num_envs, 1)``."""
        ...

    @abstractmethod
    def _compute_terminated(self) -> torch.Tensor:
        """Compute per-env termination flag.  Shape: ``(num_envs, 1)``."""
        ...

    @classmethod
    def _action_range(cls) -> tuple[float, float]:
        """Return ``(low, high)`` for the action spec.  Default ``(-1, 1)``."""
        return (-1.0, 1.0)

    def _prepare_ctrl(self, action: torch.Tensor) -> torch.Tensor:
        """Transform agent action into simulator ctrl vector.

        Override for partial actuation (e.g. satellite CMGs where rotors
        are held at constant speed and only gimbals are agent-controlled).
        """
        return action

    def _build_obs(self) -> dict:
        """Build the full observation dict, optionally including pixels."""
        obs = {} if (self.from_pixels and self.pixel_only) else self._make_obs()
        if self.from_pixels:
            obs["pixels"] = self._render_pixels()
        return obs

    def _render_pixels(self) -> torch.Tensor:
        """Render all environments.  Returns ``(num_envs, H, W, 3)`` uint8."""
        frames = []
        for i in range(self.num_envs):
            rgb, _, _ = mujoco_torch.render(
                self.mx,
                self._dx[i],
                camera_id=0,
                width=self.render_width,
                height=self.render_height,
                precomp=self._render_precomp,
                background=self.RENDER_BACKGROUND,
            )
            frames.append((rgb * 255).clamp(0, 255).to(torch.uint8))
        return torch.stack(frames)

    @classmethod
    def _camera_xml(cls) -> str:
        """Return the XML for the default camera.

        Subclasses can override for environment-specific framing.
        """
        return '<camera name="side" pos="0 -4 3" xyaxes="1 0 0 0 0.45 1" fovy="60"/>'

    @classmethod
    def _patch_xml(cls, xml: str) -> str:
        """Modify the XML string before model creation.

        Replaces all cameras and lights with a well-lit, fixed viewpoint.
        Also injects a ground plane with collision if the model has none.
        """
        xml = re.sub(r"<camera\b[^/]*/>\s*", "", xml)
        xml = re.sub(r"<light\b[^/]*/>\s*", "", xml)
        camera = cls._camera_xml()
        light = (
            '<light name="top" pos="0 0 4" dir="0 0 -1" '
            'diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" '
            'directional="true"/>'
        )
        floor = ""
        if not re.search(r'<geom\b[^>]*type="plane"', xml):
            floor = (
                '\n  <geom name="floor" type="plane" size="10 10 0.1" '
                'rgba="0.8 0.85 0.8 1" conaffinity="1" condim="3"/>'
            )
        xml = xml.replace(
            "<worldbody>",
            f"<worldbody>\n  {camera}\n  {light}{floor}",
        )
        return xml

    # ------------------------------------------------------------------
    # TorchRL interface
    # ------------------------------------------------------------------

    def _make_batch(self, n: int):
        batch = self._dx0.expand(n).clone()
        noise = self.RESET_NOISE_SCALE
        if noise > 0:
            batch.qpos.add_(torch.empty_like(batch.qpos).uniform_(-noise, noise))
            batch.qvel.add_(torch.empty_like(batch.qvel).uniform_(-noise, noise))
        return batch

    def _reset(self, tensordict=None, **kwargs):
        reset_mask = None
        if tensordict is not None and "_reset" in tensordict.keys():
            reset_mask = tensordict["_reset"].squeeze(-1)

        if reset_mask is None or not hasattr(self, "_dx"):
            # Full reset (initial or forced)
            self._dx = self._make_batch(self.num_envs)
            self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        else:
            n_reset = int(reset_mask.sum())
            if n_reset > 0:
                self._dx[reset_mask] = self._make_batch(n_reset)
                self._step_count[reset_mask] = 0

        return TensorDict(
            {
                **self._build_obs(),
                "done": torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict):
        action = tensordict["action"].to(self.dtype)
        ctrl = self._prepare_ctrl(action)

        qpos_before = self._dx.qpos.clone()

        self._dx.update_(ctrl=ctrl)
        if self._single_env:
            self._dx = self._physics_step(self._dx[0]).unsqueeze(0)
        else:
            self._dx = self._physics_step(self._dx)

        # Physics health check: catch blow-ups at the source
        _PHYS_THRESHOLD = 1e10
        qpos = self._dx.qpos
        qvel = self._dx.qvel
        qpos_bad = qpos.isnan().any() or qpos.isinf().any() or (qpos.abs() > _PHYS_THRESHOLD).any()
        qvel_bad = qvel.isnan().any() or qvel.isinf().any() or (qvel.abs() > _PHYS_THRESHOLD).any()
        if qpos_bad or qvel_bad:
            qpos_issue = qpos.abs().max(dim=-1).values
            qvel_issue = qvel.abs().max(dim=-1).values
            bad_envs = (
                (qpos_issue > _PHYS_THRESHOLD)
                | qpos.isnan().any(-1)
                | (qvel_issue > _PHYS_THRESHOLD)
                | qvel.isnan().any(-1)
            )
            bad_idx = bad_envs.nonzero(as_tuple=True)[0]
            msg = (
                f"\n{'=' * 70}\n"
                f"FATAL: Physics blow-up at step_count={self._step_count[bad_idx[0]].item()}\n"
                f"  {bad_idx.numel()}/{self.num_envs} envs affected, first bad env idx={bad_idx[0].item()}\n"
                f"  qpos range: [{qpos.min():.2e}, {qpos.max():.2e}]  nan={qpos.isnan().sum().item()}\n"
                f"  qvel range: [{qvel.min():.2e}, {qvel.max():.2e}]  nan={qvel.isnan().sum().item()}\n"
                f"  qpos_before range: [{qpos_before.min():.2e}, {qpos_before.max():.2e}]\n"
                f"  action range: [{action.min():.2e}, {action.max():.2e}]\n"
                f"  ctrl range: [{ctrl.min():.2e}, {ctrl.max():.2e}]\n"
                f"  bad env qpos: {qpos[bad_idx[0]].cpu().tolist()}\n"
                f"  bad env qvel: {qvel[bad_idx[0]].cpu().tolist()}\n"
                f"  bad env qpos_before: {qpos_before[bad_idx[0]].cpu().tolist()}\n"
                f"  bad env action: {action[bad_idx[0]].cpu().tolist()}\n"
                f"{'=' * 70}"
            )
            raise RuntimeError(msg)
        self._step_count += 1

        reward = self._compute_reward(qpos_before, action)

        # Reward health check
        if reward.isnan().any() or reward.isinf().any() or (reward.abs() > _PHYS_THRESHOLD).any():
            bad_mask = (
                reward.squeeze(-1).isnan() | reward.squeeze(-1).isinf() | (reward.squeeze(-1).abs() > _PHYS_THRESHOLD)
            )
            bad_idx = bad_mask.nonzero(as_tuple=True)[0]
            msg = (
                f"\n{'=' * 70}\n"
                f"FATAL: Extreme reward at step_count={self._step_count[bad_idx[0]].item()}\n"
                f"  {bad_idx.numel()}/{self.num_envs} envs affected\n"
                f"  reward range: [{reward.min():.2e}, {reward.max():.2e}]  nan={reward.isnan().sum().item()}\n"
                f"  worst reward: {reward[bad_idx[0]].item():.2e}\n"
                f"  bad env qpos: {self._dx.qpos[bad_idx[0]].cpu().tolist()}\n"
                f"  bad env qvel: {self._dx.qvel[bad_idx[0]].cpu().tolist()}\n"
                f"  bad env qpos_before: {qpos_before[bad_idx[0]].cpu().tolist()}\n"
                f"  bad env action: {action[bad_idx[0]].cpu().tolist()}\n"
                f"{'=' * 70}"
            )
            raise RuntimeError(msg)

        terminated = self._compute_terminated()
        truncated = (self._step_count >= self.max_episode_steps).unsqueeze(-1)
        done = terminated | truncated

        # Build obs from the terminal state BEFORE resetting
        obs = self._build_obs()

        # Branchless reset: always compute reset candidate, select with torch.where.
        # No control flow → fully compilable, no graph breaks.
        noise = self.RESET_NOISE_SCALE
        reset_dx = self._dx0.expand_as(self._dx).clone()
        reset_dx.qpos = reset_dx.qpos + torch.empty_like(reset_dx.qpos).uniform_(-noise, noise)
        reset_dx.qvel = reset_dx.qvel + torch.empty_like(reset_dx.qvel).uniform_(-noise, noise)
        done_mask = done.squeeze(-1)
        self._dx = self._dx.apply(
            lambda v, rv: torch.where(done_mask.view(-1, *((1,) * (v.ndim - 1))), rv, v),
            reset_dx,
        )
        self._step_count = torch.where(done.squeeze(-1), torch.zeros_like(self._step_count), self._step_count)

        return TensorDict(
            {
                **obs,
                "reward": reward,
                "done": done,
                "terminated": terminated,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed):
        torch.manual_seed(seed)

    def render(self, width=256, height=256, camera_id=0):
        """Render the first environment using the mujoco-torch ray-cast renderer.

        Returns an RGB uint8 array of shape ``(H, W, 3)``.
        """
        rgb, _, _ = mujoco_torch.render(
            self.mx,
            self._dx[0],
            camera_id=camera_id,
            width=width,
            height=height,
            precomp=self._render_precomp,
            background=self.RENDER_BACKGROUND,
        )
        return (rgb * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
