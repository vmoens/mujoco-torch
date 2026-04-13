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
            when ``compile_step`` is enabled.
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
        auto_reset: bool = False,
        frame_skip: int | None = None,
        from_pixels: bool = False,
        pixel_only: bool = False,
        render_width: int = 64,
        render_height: int = 64,
    ):
        if frame_skip is not None:
            self.FRAME_SKIP = frame_skip
        self.auto_reset = auto_reset
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
        self._sim_dtype = self._dx0.qpos.dtype
        self._ctrl_dtype = self._dx0.ctrl.dtype
        self._render_precomp = mujoco_torch.precompute_render_data(self.mx)

        _step_fn = lambda d: mujoco_torch.step(self.mx, d)  # noqa: E731
        self._raw_step_fn = _step_fn
        frame_skip = self.FRAME_SKIP
        compile_kwargs = compile_kwargs or {}
        _vmap_step = torch.vmap(_step_fn)
        self._raw_vmap_step = _vmap_step
        if num_envs == 1:

            def _multi_step(d):
                for _ in range(frame_skip):
                    d = _step_fn(d)
                return d

            self._physics_step = torch.compile(_multi_step, **compile_kwargs) if compile_step else _multi_step
            self._single_env = True
        else:

            def _vmap_multi_step(d):
                for _ in range(frame_skip):
                    d = _vmap_step(d)
                return d

            self._physics_step = torch.compile(_vmap_multi_step, **compile_kwargs) if compile_step else _vmap_multi_step
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
        return action.to(self._ctrl_dtype)

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
        elif not self.auto_reset:
            # Only do per-env reset if auto_reset is off (otherwise _step already did it)
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

        # -- NaN diagnostic: per-sub-step loop with guard --
        # Bypass self._physics_step (compiled wrapper) and run sub-steps
        # manually so we can check for NaN between each sub-step.
        for sub_step in range(self.FRAME_SKIP):
            _dx_before = self._dx.clone()
            if self._single_env:
                self._dx = self._raw_step_fn(self._dx[0]).unsqueeze(0)
            else:
                self._dx = self._raw_vmap_step(self._dx)

            # Check for primary NaN event: finite input → NaN output
            if not getattr(self, "_nan_dumped", False):
                qpos_bad = ~torch.isfinite(self._dx.qpos)
                qvel_bad = ~torch.isfinite(self._dx.qvel)
                any_bad = qpos_bad.flatten(1).any(dim=-1) | qvel_bad.flatten(1).any(dim=-1)
                if any_bad.any():
                    # Check if input was finite (primary event)
                    before_qpos_finite = torch.isfinite(_dx_before.qpos).all()
                    before_qvel_finite = torch.isfinite(_dx_before.qvel).all()
                    before_ctrl_finite = torch.isfinite(_dx_before.ctrl).all()
                    self._nan_dumped = True
                    dump = {
                        "before": {
                            "qpos": _dx_before.qpos[any_bad].detach().cpu(),
                            "qvel": _dx_before.qvel[any_bad].detach().cpu(),
                            "qacc": _dx_before.qacc[any_bad].detach().cpu(),
                            "ctrl": _dx_before.ctrl[any_bad].detach().cpu(),
                        },
                        "before_all": {
                            "qpos": _dx_before.qpos.detach().cpu(),
                            "qvel": _dx_before.qvel.detach().cpu(),
                            "qacc": _dx_before.qacc.detach().cpu(),
                            "ctrl": _dx_before.ctrl.detach().cpu(),
                        },
                        "after": {
                            "qpos": self._dx.qpos[any_bad].detach().cpu(),
                            "qvel": self._dx.qvel[any_bad].detach().cpu(),
                            "qacc": self._dx.qacc[any_bad].detach().cpu(),
                            "ctrl": self._dx.ctrl[any_bad].detach().cpu(),
                        },
                        "after_all": {
                            "qpos": self._dx.qpos.detach().cpu(),
                            "qvel": self._dx.qvel.detach().cpu(),
                            "qacc": self._dx.qacc.detach().cpu(),
                            "ctrl": self._dx.ctrl.detach().cpu(),
                        },
                        "action": action[any_bad].detach().cpu(),
                        "step_count": self._step_count.detach().cpu(),
                        "sub_step": sub_step,
                        "frame_skip": self.FRAME_SKIP,
                        "_bad_env_mask": any_bad.detach().cpu(),
                        "_qpos_bad_mask": qpos_bad[any_bad].detach().cpu(),
                        "_qvel_bad_mask": qvel_bad[any_bad].detach().cpu(),
                        "_before_all_finite": {
                            "qpos": before_qpos_finite.item(),
                            "qvel": before_qvel_finite.item(),
                            "ctrl": before_ctrl_finite.item(),
                        },
                    }
                    torch.save(dump, "/tmp/mujoco_nan_dump.pt")
                    import logging
                    log = logging.getLogger(__name__)
                    n_bad = any_bad.sum().item()
                    log.error(
                        "NaN in physics sub-step %d/%d: %d/%d envs. "
                        "Input finite: qpos=%s qvel=%s ctrl=%s. "
                        "Dump saved to /tmp/mujoco_nan_dump.pt",
                        sub_step, self.FRAME_SKIP, n_bad, self.num_envs,
                        before_qpos_finite.item(), before_qvel_finite.item(),
                        before_ctrl_finite.item(),
                    )

        self._step_count += 1

        reward = self._compute_reward(qpos_before, action)
        terminated = self._compute_terminated()
        truncated = (self._step_count >= self.max_episode_steps).unsqueeze(-1)
        done = terminated | truncated

        # Build obs from the terminal state BEFORE resetting, so the returned
        # observation matches the state that produced the reward.
        obs = self._build_obs()

        # Fused auto-reset: reset done envs AFTER building obs.
        # maybe_reset will call _reset (which is a no-op with auto_reset)
        # and _reset returns obs from the already-reset _dx state.
        if self.auto_reset:
            done_mask = done.squeeze(-1)
            if done_mask.any():
                self._dx[done_mask] = self._make_batch(int(done_mask.sum()))
                self._step_count[done_mask] = 0

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
