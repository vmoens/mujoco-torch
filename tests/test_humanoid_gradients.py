"""Gradient accuracy tests for the Humanoid differentiable physics pipeline.

Verifies that autograd gradients through differentiable mujoco-torch match
finite-difference approximations (``torch.autograd.gradcheck``) for:

1. Single physics step: ctrl → qpos
2. Multi-step rollout: ctrl → qpos through N steps
3. Smooth reward: action → reward through env step
4. Full TorchRL env: action → reward through SmoothHumanoidEnv.step()
"""

import mujoco
import torch

import mujoco_torch
from mujoco_torch import mujoco_logger as logger

torch.set_default_dtype(torch.float64)

_XML_PATH = str(mujoco_torch._src.__file__.replace("__init__.py", "").replace("_src/", "") + "test_data/humanoid.xml")


def _make_humanoid():
    """Load Humanoid model and create a reference data state."""
    from etils import epath

    xml = (epath.resource_path("mujoco_torch") / "test_data" / "humanoid.xml").read_text()
    m_mj = mujoco.MjModel.from_xml_string(xml)
    mx = mujoco_torch.device_put(m_mj)

    d_mj = mujoco.MjData(m_mj)
    mujoco.mj_forward(m_mj, d_mj)
    dx = mujoco_torch.device_put(d_mj)
    # One warmup step so dtypes are consistent
    dx = mujoco_torch.step(mx, dx)
    return mx, dx, m_mj


# ------------------------------------------------------------------
# Test 1: single step  ctrl → qpos
# ------------------------------------------------------------------


def test_single_step_gradcheck():
    """Autograd gradient of a single physics step matches finite differences."""
    mx, dx, _ = _make_humanoid()

    def fn(ctrl):
        d = dx.clone().replace(ctrl=ctrl)
        with mujoco_torch.differentiable_mode(smooth_collisions=True, cfd=True):
            d = mujoco_torch.step(mx, d, fixed_iterations=True)
        return d.qpos

    ctrl = dx.ctrl.clone().detach().requires_grad_(True)
    ok = torch.autograd.gradcheck(fn, (ctrl,), eps=1e-6, atol=1e-4, rtol=1e-3)
    assert ok, "gradcheck failed for single step ctrl → qpos"
    logger.info("  single_step gradcheck: PASS")


# ------------------------------------------------------------------
# Test 2: multi-step  ctrl → qpos through N steps
# ------------------------------------------------------------------


def test_multi_step_gradcheck():
    """Gradient through 3 consecutive physics steps matches finite differences."""
    mx, dx, _ = _make_humanoid()
    n_steps = 3

    def fn(ctrl):
        d = dx.clone().replace(ctrl=ctrl)
        with mujoco_torch.differentiable_mode(smooth_collisions=True, cfd=True):
            for _ in range(n_steps):
                d = mujoco_torch.step(mx, d, fixed_iterations=True)
        # Return a scalar to keep gradcheck tractable
        return d.qpos.sum().unsqueeze(0)

    ctrl = dx.ctrl.clone().detach().requires_grad_(True)
    # Looser tolerance for multi-step: small numerical errors accumulate
    ok = torch.autograd.gradcheck(fn, (ctrl,), eps=1e-5, atol=1e-3, rtol=1e-2)
    assert ok, f"gradcheck failed for {n_steps}-step ctrl → qpos"
    logger.info(f"  multi_step ({n_steps}) gradcheck: PASS")


# ------------------------------------------------------------------
# Test 3: smooth reward  action → reward
# ------------------------------------------------------------------


def test_smooth_reward_gradcheck():
    """Gradient of smooth Humanoid reward w.r.t. action matches finite differences."""
    mx, dx, m_mj = _make_humanoid()
    dt = m_mj.opt.timestep

    def fn(action):
        d = dx.clone().replace(ctrl=action)
        qpos_before = d.qpos.clone()
        with mujoco_torch.differentiable_mode(smooth_collisions=True, cfd=True):
            d = mujoco_torch.step(mx, d, fixed_iterations=True)

        forward_vel = (d.qpos[0] - qpos_before[0]) / dt
        z = d.qpos[2]
        soft_healthy = torch.sigmoid(10.0 * (z - 1.0)) * torch.sigmoid(10.0 * (2.0 - z))
        ctrl_cost = 0.1 * (action**2).sum()
        reward = forward_vel + 5.0 * soft_healthy - ctrl_cost
        return reward.unsqueeze(0)

    action = dx.ctrl.clone().detach().requires_grad_(True)
    ok = torch.autograd.gradcheck(fn, (action,), eps=1e-6, atol=1e-4, rtol=1e-3)
    assert ok, "gradcheck failed for smooth reward"
    logger.info("  smooth_reward gradcheck: PASS")


# ------------------------------------------------------------------
# Test 4: TorchRL env step  action → reward
# ------------------------------------------------------------------


def test_env_step_gradcheck():
    """Gradient through SmoothHumanoidEnv.step() matches finite differences."""
    # Import here to avoid circular import issues
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    from train_direct_humanoid import SmoothHumanoidEnv

    env = SmoothHumanoidEnv(num_envs=1, frame_skip=1)
    env.reset()

    # Save env state for reproducible calls
    dx_saved = env._dx.clone()
    obs_saved = env._make_obs()["observation"].clone().detach()

    def fn(action):
        # Restore env state before each call (gradcheck calls fn many times)
        env._dx = dx_saved.clone()
        env._step_count = torch.zeros(1, dtype=torch.long, device=env.device)
        from tensordict import TensorDict

        td = TensorDict(
            {"observation": obs_saved, "action": action},
            batch_size=[1],
        )
        with mujoco_torch.differentiable_mode(smooth_collisions=True, cfd=True):
            next_td = env.step(td)
        return next_td["next", "reward"]

    action = torch.zeros(1, env.action_spec.shape[-1], requires_grad=True)
    ok = torch.autograd.gradcheck(fn, (action,), eps=1e-6, atol=1e-4, rtol=1e-3)
    assert ok, "gradcheck failed for TorchRL env step"
    logger.info("  env_step gradcheck: PASS")


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Humanoid gradient accuracy tests (float64 vs finite diff)")
    logger.info("=" * 60)

    test_single_step_gradcheck()
    test_multi_step_gradcheck()
    test_smooth_reward_gradcheck()
    test_env_step_gradcheck()

    logger.info("All Humanoid gradient tests PASSED.")
