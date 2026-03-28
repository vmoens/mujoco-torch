#!/usr/bin/env python3
"""Numerical accuracy comparison: mujoco-torch vs MuJoCo C for Humanoid.

Tests that dynamics, reward, and action effects match between backends.
Run on CPU (no GPU needed).

Usage::

    python examples/humanoid_accuracy_test.py
"""

import mujoco
import numpy as np
import torch

import mujoco_torch
from mujoco_torch._src import test_util

torch.set_default_dtype(torch.float64)

# ---- Config ----
MODEL_FILE = "humanoid.xml"
FRAME_SKIP = 5
NSTEPS = 50  # multi-step trajectory length
N_RANDOM_STATES = 5
SEED = 42
CTRL_SCALE = 0.5  # scale for random actions

# Humanoid reward constants (matching zoo/humanoid.py)
HEALTHY_Z_LOW = 1.0
HEALTHY_Z_HIGH = 2.0
HEALTHY_REWARD = 5.0
CTRL_COST_WEIGHT = 0.1


def compute_reward_np(qpos_before, qpos_after, action, dt):
    """Compute humanoid reward using numpy (reference implementation)."""
    forward_vel = (qpos_after[0] - qpos_before[0]) / dt
    ctrl_cost = CTRL_COST_WEIGHT * np.sum(action**2)
    z = qpos_after[2]
    healthy = HEALTHY_REWARD if (HEALTHY_Z_LOW <= z <= HEALTHY_Z_HIGH) else 0.0
    reward = forward_vel + healthy - ctrl_cost
    return reward, forward_vel, healthy, ctrl_cost


def compute_reward_torch(qpos_before, qpos_after, action, dt):
    """Compute humanoid reward using torch (matching zoo implementation)."""
    forward_vel = (qpos_after[..., 0] - qpos_before[..., 0]) / dt
    ctrl_cost = CTRL_COST_WEIGHT * (action**2).sum(dim=-1)
    z = qpos_after[..., 2]
    healthy_reward = torch.where(
        (z >= HEALTHY_Z_LOW) & (z <= HEALTHY_Z_HIGH),
        torch.tensor(HEALTHY_REWARD, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
    )
    reward = forward_vel + healthy_reward - ctrl_cost
    return reward, forward_vel, healthy_reward, ctrl_cost


def run_mujoco_c(m_mj, qpos_init, qvel_init, actions, frame_skip):
    """Run MuJoCo C simulation with given actions.

    actions: (nsteps, nu) array
    Returns list of (qpos, qvel) after each action frame.
    """
    d_mj = mujoco.MjData(m_mj)
    d_mj.qpos[:] = qpos_init
    d_mj.qvel[:] = qvel_init
    mujoco.mj_forward(m_mj, d_mj)

    results = []
    for t in range(len(actions)):
        qpos_before = d_mj.qpos.copy()
        d_mj.ctrl[:] = actions[t]
        for _ in range(frame_skip):
            mujoco.mj_step(m_mj, d_mj)
        results.append(
            {
                "qpos_before": qpos_before,
                "qpos": d_mj.qpos.copy(),
                "qvel": d_mj.qvel.copy(),
                "time": d_mj.time,
            }
        )
    return results


def run_mujoco_torch_single(m_mj, qpos_init, qvel_init, actions, frame_skip):
    """Run mujoco-torch single-env simulation with given actions."""
    mx = mujoco_torch.device_put(m_mj)

    d_mj = mujoco.MjData(m_mj)
    d_mj.qpos[:] = qpos_init
    d_mj.qvel[:] = qvel_init
    mujoco.mj_forward(m_mj, d_mj)
    dx = mujoco_torch.device_put(d_mj)

    step_fn = lambda d: mujoco_torch.step(mx, d)

    results = []
    for t in range(len(actions)):
        qpos_before = dx.qpos.clone()
        ctrl = torch.from_numpy(actions[t])
        dx = dx.replace(ctrl=ctrl)
        for _ in range(frame_skip):
            dx = step_fn(dx)
        results.append(
            {
                "qpos_before": qpos_before.numpy(),
                "qpos": dx.qpos.numpy().copy(),
                "qvel": dx.qvel.numpy().copy(),
                "time": float(dx.time),
            }
        )
    return results


def run_mujoco_torch_vmap(m_mj, qpos_init, qvel_init, actions, frame_skip, batch_size=4):
    """Run mujoco-torch vmap simulation. All envs get same init and actions."""
    mx = mujoco_torch.device_put(m_mj)

    envs = []
    for _ in range(batch_size):
        d_mj = mujoco.MjData(m_mj)
        d_mj.qpos[:] = qpos_init
        d_mj.qvel[:] = qvel_init
        mujoco.mj_forward(m_mj, d_mj)
        envs.append(mujoco_torch.device_put(d_mj))
    dx = torch.stack(envs, dim=0)

    vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))

    results = []
    for t in range(len(actions)):
        qpos_before = dx.qpos.clone()
        ctrl = torch.from_numpy(actions[t]).unsqueeze(0).expand(batch_size, -1)
        dx = dx.replace(ctrl=ctrl)
        for _ in range(frame_skip):
            dx = vmap_step(dx)
        results.append(
            {
                "qpos_before": qpos_before[0].numpy(),
                "qpos": dx.qpos[0].numpy().copy(),
                "qvel": dx.qvel[0].numpy().copy(),
                "time": float(dx.time[0]),
            }
        )
    return results


def compare_results(ref_results, test_results, label, atol=1e-6):
    """Compare two trajectory results. Returns max errors."""
    max_qpos_err = 0.0
    max_qvel_err = 0.0
    max_time_err = 0.0
    all_ok = True

    for i, (ref, test) in enumerate(zip(ref_results, test_results)):
        qpos_err = np.abs(ref["qpos"] - test["qpos"]).max()
        qvel_err = np.abs(ref["qvel"] - test["qvel"]).max()
        time_err = abs(ref["time"] - test["time"])

        max_qpos_err = max(max_qpos_err, qpos_err)
        max_qvel_err = max(max_qvel_err, qvel_err)
        max_time_err = max(max_time_err, time_err)

        if qpos_err > atol or qvel_err > atol:
            if all_ok:
                print(f"  DIVERGENCE at step {i}: qpos_err={qpos_err:.2e}, qvel_err={qvel_err:.2e}")
            all_ok = False

    status = "PASS" if all_ok else "FAIL"
    print(
        f"  [{status}] {label}: max_qpos_err={max_qpos_err:.2e},"
        f" max_qvel_err={max_qvel_err:.2e}, max_time_err={max_time_err:.2e}"
    )
    return all_ok, max_qpos_err, max_qvel_err


def test_zero_ctrl(m_mj):
    """Test: zero control should match between backends."""
    print("\n=== TEST 1: Zero control, default init ===")
    d_mj = mujoco.MjData(m_mj)
    mujoco.mj_forward(m_mj, d_mj)
    qpos_init = d_mj.qpos.copy()
    qvel_init = d_mj.qvel.copy()

    actions = np.zeros((NSTEPS, m_mj.nu))

    ref = run_mujoco_c(m_mj, qpos_init, qvel_init, actions, FRAME_SKIP)
    torch_single = run_mujoco_torch_single(m_mj, qpos_init, qvel_init, actions, FRAME_SKIP)
    torch_vmap = run_mujoco_torch_vmap(m_mj, qpos_init, qvel_init, actions, FRAME_SKIP)

    ok1, _, _ = compare_results(ref, torch_single, "MuJoCo C vs mujoco-torch (single)", atol=1e-6)
    ok2, _, _ = compare_results(ref, torch_vmap, "MuJoCo C vs mujoco-torch (vmap)", atol=1e-4)
    return ok1 and ok2


def test_random_ctrl(m_mj):
    """Test: random control should produce matching trajectories."""
    print("\n=== TEST 2: Random control, default init ===")
    rng = np.random.RandomState(SEED)

    d_mj = mujoco.MjData(m_mj)
    mujoco.mj_forward(m_mj, d_mj)
    qpos_init = d_mj.qpos.copy()
    qvel_init = d_mj.qvel.copy()

    actions = CTRL_SCALE * rng.randn(NSTEPS, m_mj.nu)
    actions = np.clip(actions, -1.0, 1.0)

    ref = run_mujoco_c(m_mj, qpos_init, qvel_init, actions, FRAME_SKIP)
    torch_single = run_mujoco_torch_single(m_mj, qpos_init, qvel_init, actions, FRAME_SKIP)

    ok, _, _ = compare_results(ref, torch_single, "MuJoCo C vs mujoco-torch (single, random ctrl)", atol=1e-5)
    return ok


def test_action_effect(m_mj):
    """Test: actions should actually change the trajectory."""
    print("\n=== TEST 3: Action effect (non-zero ctrl changes trajectory) ===")
    rng = np.random.RandomState(SEED)

    d_mj = mujoco.MjData(m_mj)
    mujoco.mj_forward(m_mj, d_mj)
    qpos_init = d_mj.qpos.copy()
    qvel_init = d_mj.qvel.copy()

    # Run with zero ctrl
    zero_actions = np.zeros((10, m_mj.nu))
    ref_zero = run_mujoco_c(m_mj, qpos_init, qvel_init, zero_actions, FRAME_SKIP)
    torch_zero = run_mujoco_torch_single(m_mj, qpos_init, qvel_init, zero_actions, FRAME_SKIP)

    # Run with random ctrl
    rand_actions = 0.8 * rng.randn(10, m_mj.nu).clip(-1, 1)
    ref_rand = run_mujoco_c(m_mj, qpos_init, qvel_init, rand_actions, FRAME_SKIP)
    torch_rand = run_mujoco_torch_single(m_mj, qpos_init, qvel_init, rand_actions, FRAME_SKIP)

    # Check that ctrl makes a difference
    c_diff = np.abs(ref_zero[-1]["qpos"] - ref_rand[-1]["qpos"]).max()
    t_diff = np.abs(torch_zero[-1]["qpos"] - torch_rand[-1]["qpos"]).max()

    print(f"  MuJoCo C: max qpos diff (zero vs rand ctrl) = {c_diff:.4e}")
    print(f"  mujoco-torch: max qpos diff (zero vs rand ctrl) = {t_diff:.4e}")

    ok = True
    if c_diff < 1e-6:
        print("  FAIL: Actions have NO effect in MuJoCo C!")
        ok = False
    else:
        print(f"  PASS: Actions change MuJoCo C trajectory (diff={c_diff:.4e})")

    if t_diff < 1e-6:
        print("  FAIL: Actions have NO effect in mujoco-torch!")
        ok = False
    else:
        print(f"  PASS: Actions change mujoco-torch trajectory (diff={t_diff:.4e})")

    # Check the diffs are similar
    ratio = t_diff / max(c_diff, 1e-15)
    print(f"  Effect ratio (torch/C): {ratio:.4f}")
    if abs(ratio - 1.0) > 0.1:
        print(f"  WARNING: Action effect differs by {abs(ratio - 1) * 100:.1f}%")

    return ok


def test_random_states(m_mj):
    """Test: trajectories from random initial states should match."""
    print(f"\n=== TEST 4: Random initial states ({N_RANDOM_STATES} states) ===")
    rng = np.random.RandomState(SEED + 100)
    all_ok = True

    d_mj = mujoco.MjData(m_mj)
    mujoco.mj_forward(m_mj, d_mj)
    qpos_ref = d_mj.qpos.copy()
    qvel_ref = d_mj.qvel.copy()

    for s in range(N_RANDOM_STATES):
        # Perturb from default
        noise = 0.01
        qpos_init = qpos_ref + noise * rng.randn(*qpos_ref.shape)
        qvel_init = qvel_ref + noise * rng.randn(*qvel_ref.shape)

        actions = CTRL_SCALE * rng.randn(20, m_mj.nu).clip(-1, 1)

        ref = run_mujoco_c(m_mj, qpos_init, qvel_init, actions, FRAME_SKIP)
        torch_single = run_mujoco_torch_single(m_mj, qpos_init, qvel_init, actions, FRAME_SKIP)

        ok, qe, ve = compare_results(
            ref,
            torch_single,
            f"State {s}: MuJoCo C vs mujoco-torch",
            atol=1e-5,
        )
        all_ok = all_ok and ok

    return all_ok


def test_reward_match(m_mj):
    """Test: reward computed from MuJoCo C trajectory matches torch."""
    print("\n=== TEST 5: Reward computation ===")
    rng = np.random.RandomState(SEED + 200)
    dt = m_mj.opt.timestep * FRAME_SKIP

    d_mj = mujoco.MjData(m_mj)
    mujoco.mj_forward(m_mj, d_mj)
    qpos_init = d_mj.qpos.copy()
    qvel_init = d_mj.qvel.copy()

    actions = CTRL_SCALE * rng.randn(20, m_mj.nu).clip(-1, 1)

    # Run MuJoCo C and compute rewards
    ref = run_mujoco_c(m_mj, qpos_init, qvel_init, actions, FRAME_SKIP)

    all_ok = True
    for i, (step_result, action) in enumerate(zip(ref, actions)):
        # Numpy reward
        r_np, fv_np, h_np, cc_np = compute_reward_np(
            step_result["qpos_before"],
            step_result["qpos"],
            action,
            dt,
        )
        # Torch reward
        qb = torch.from_numpy(step_result["qpos_before"])
        qa = torch.from_numpy(step_result["qpos"])
        a = torch.from_numpy(action)
        r_t, fv_t, h_t, cc_t = compute_reward_torch(qb, qa, a, dt)

        r_err = abs(r_np - r_t.item())
        if r_err > 1e-10:
            print(f"  Step {i}: reward mismatch: np={r_np:.6f}, torch={r_t.item():.6f}, err={r_err:.2e}")
            all_ok = False

        if i < 3 or i == len(actions) - 1:
            print(
                f"  Step {i:2d}: reward={r_np:8.4f} "
                f"(fwd_vel={fv_np:7.4f}, healthy={h_np:.1f}, ctrl_cost={cc_np:.4f}) "
                f"z={step_result['qpos'][2]:.4f}"
            )

    if all_ok:
        print("  [PASS] All rewards match between numpy and torch")
    return all_ok


def test_forward_velocity(m_mj):
    """Test: humanoid actually moves when given directional actions."""
    print("\n=== TEST 6: Forward velocity under sustained action ===")
    d_mj = mujoco.MjData(m_mj)
    mujoco.mj_forward(m_mj, d_mj)
    qpos_init = d_mj.qpos.copy()
    qvel_init = d_mj.qvel.copy()
    dt = m_mj.opt.timestep * FRAME_SKIP

    # Apply constant action (hip flexion to try walking)
    actions = np.zeros((100, m_mj.nu))
    # hip_y_right (index 5, gear=120), hip_y_left (index 11, gear=120)
    actions[:, 5] = 0.5  # hip_y_right
    actions[:, 11] = -0.5  # hip_y_left (alternating)

    ref_c = run_mujoco_c(m_mj, qpos_init, qvel_init, actions, FRAME_SKIP)
    ref_t = run_mujoco_torch_single(m_mj, qpos_init, qvel_init, actions, FRAME_SKIP)

    x_c = [r["qpos"][0] for r in ref_c]
    x_t = [r["qpos"][0] for r in ref_t]
    z_c = [r["qpos"][2] for r in ref_c]
    z_t = [r["qpos"][2] for r in ref_t]

    print(f"  MuJoCo C: x displacement = {x_c[-1] - x_c[0]:.4f} (from {x_c[0]:.4f} to {x_c[-1]:.4f})")
    print(f"  mujoco-torch: x displacement = {x_t[-1] - x_t[0]:.4f} (from {x_t[0]:.4f} to {x_t[-1]:.4f})")
    print(f"  MuJoCo C: z final = {z_c[-1]:.4f} (started {z_c[0]:.4f})")
    print(f"  mujoco-torch: z final = {z_t[-1]:.4f} (started {z_t[0]:.4f})")

    # Check velocities at various points
    for t_idx in [0, 9, 19, 49, 99]:
        fv_c = (ref_c[t_idx]["qpos"][0] - ref_c[t_idx]["qpos_before"][0]) / dt
        fv_t = (ref_t[t_idx]["qpos"][0] - ref_t[t_idx]["qpos_before"][0]) / dt
        print(f"  Step {t_idx:3d}: fwd_vel C={fv_c:8.4f}, torch={fv_t:8.4f}, diff={abs(fv_c - fv_t):.2e}")

    x_diff = abs(x_c[-1] - x_t[-1])
    print(f"  Final x diff between backends: {x_diff:.2e}")

    ok = x_diff < 0.1  # should be very close
    if ok:
        print("  [PASS] Both backends agree on forward motion")
    else:
        print("  [FAIL] Backends disagree on forward motion!")
    return ok


def test_xml_patching_effect():
    """Test: zoo XML patching (camera/light changes) doesn't affect dynamics."""
    print("\n=== TEST 7: XML patching (zoo) doesn't change dynamics ===")
    rng = np.random.RandomState(SEED + 300)

    from etils import epath
    from mujoco_torch.zoo.humanoid import HumanoidEnv

    _TEST_DATA = epath.resource_path("mujoco_torch") / "test_data"
    raw_xml = (_TEST_DATA / "humanoid.xml").read_text()
    patched_xml = HumanoidEnv._patch_xml(raw_xml)
    m_raw = mujoco.MjModel.from_xml_string(raw_xml)
    m_patched = mujoco.MjModel.from_xml_string(patched_xml)

    print(f"  Raw model: nq={m_raw.nq}, nv={m_raw.nv}, nu={m_raw.nu}, nbody={m_raw.nbody}")
    print(f"  Patched model: nq={m_patched.nq}, nv={m_patched.nv}, nu={m_patched.nu}, nbody={m_patched.nbody}")
    print(f"  Raw timestep: {m_raw.opt.timestep}")
    print(f"  Patched timestep: {m_patched.opt.timestep}")

    if m_raw.nq != m_patched.nq or m_raw.nv != m_patched.nv:
        print("  [FAIL] Model dimensions differ!")
        return False

    d_raw = mujoco.MjData(m_raw)
    mujoco.mj_forward(m_raw, d_raw)
    qpos_init = d_raw.qpos.copy()
    qvel_init = d_raw.qvel.copy()

    actions = CTRL_SCALE * rng.randn(20, m_raw.nu).clip(-1, 1)

    ref_raw = run_mujoco_c(m_raw, qpos_init, qvel_init, actions, FRAME_SKIP)
    ref_patched = run_mujoco_c(m_patched, qpos_init, qvel_init, actions, FRAME_SKIP)

    ok, _, _ = compare_results(ref_raw, ref_patched, "Raw XML vs Patched XML (MuJoCo C)", atol=1e-10)
    return ok


def test_zoo_env_vs_manual(m_mj):
    """Test: zoo HumanoidEnv produces same dynamics as manual stepping."""
    print("\n=== TEST 8: Zoo env vs manual stepping ===")
    rng = np.random.RandomState(SEED + 400)
    dt = m_mj.opt.timestep * FRAME_SKIP

    # Create zoo env with 1 env
    from mujoco_torch.zoo.humanoid import HumanoidEnv

    env = HumanoidEnv(num_envs=1, device="cpu", frame_skip=FRAME_SKIP)

    # Reset and get initial state
    env.reset()
    qpos_init = env._dx.qpos[0].numpy().copy()
    qvel_init = env._dx.qvel[0].numpy().copy()

    print(f"  Zoo env initial z: {qpos_init[2]:.4f}")

    # Step the zoo env with a sequence of actions
    actions = CTRL_SCALE * rng.randn(10, m_mj.nu).clip(-1, 1)

    # Also run MuJoCo C with the patched model (since zoo patches the XML)
    from etils import epath

    _TEST_DATA = epath.resource_path("mujoco_torch") / "test_data"
    raw_xml = (_TEST_DATA / "humanoid.xml").read_text()
    patched_xml = HumanoidEnv._patch_xml(raw_xml)
    m_patched = mujoco.MjModel.from_xml_string(patched_xml)

    ref_c = run_mujoco_c(m_patched, qpos_init, qvel_init, actions, FRAME_SKIP)

    # Step zoo env
    all_ok = True
    from tensordict import TensorDict

    for t in range(len(actions)):
        action_t = torch.from_numpy(actions[t]).unsqueeze(0)
        td_in = TensorDict({"action": action_t}, batch_size=[1])
        td_out = env._step(td_in)

        zoo_qpos = env._dx.qpos[0].numpy()
        zoo_qvel = env._dx.qvel[0].numpy()

        qpos_err = np.abs(ref_c[t]["qpos"] - zoo_qpos).max()
        qvel_err = np.abs(ref_c[t]["qvel"] - zoo_qvel).max()

        # Also check reward
        r_np, fv_np, h_np, cc_np = compute_reward_np(
            ref_c[t]["qpos_before"],
            ref_c[t]["qpos"],
            actions[t],
            dt,
        )
        r_zoo = td_out["reward"][0, 0].item()
        r_err = abs(r_np - r_zoo)

        if t < 3 or qpos_err > 1e-5:
            print(
                f"  Step {t}: qpos_err={qpos_err:.2e}, qvel_err={qvel_err:.2e}, "
                f"reward_err={r_err:.2e} (C={r_np:.4f}, zoo={r_zoo:.4f})"
            )

        if qpos_err > 1e-4 or qvel_err > 1e-4:
            all_ok = False
        if r_err > 1e-4:
            print(f"    REWARD MISMATCH! C={r_np:.6f} zoo={r_zoo:.6f}")
            print(f"    fwd_vel={fv_np:.6f}, healthy={h_np:.1f}, ctrl_cost={cc_np:.6f}")
            all_ok = False

    status = "PASS" if all_ok else "FAIL"
    print(f"  [{status}] Zoo env vs MuJoCo C manual stepping")
    return all_ok


def main():
    print("=" * 70)
    print("Humanoid Numerical Accuracy: mujoco-torch vs MuJoCo C")
    print("=" * 70)

    # Load the raw model (as used in test suite)
    m_mj = test_util.load_test_file(MODEL_FILE)
    print(f"\nModel: {MODEL_FILE}")
    print(f"  nq={m_mj.nq}, nv={m_mj.nv}, nu={m_mj.nu}, nbody={m_mj.nbody}")
    print(f"  timestep={m_mj.opt.timestep}, frame_skip={FRAME_SKIP}")
    print(f"  dt={m_mj.opt.timestep * FRAME_SKIP}")

    results = []
    results.append(("Zero ctrl", test_zero_ctrl(m_mj)))
    results.append(("Random ctrl", test_random_ctrl(m_mj)))
    results.append(("Action effect", test_action_effect(m_mj)))
    results.append(("Random states", test_random_states(m_mj)))
    results.append(("Reward match", test_reward_match(m_mj)))
    results.append(("Forward velocity", test_forward_velocity(m_mj)))
    results.append(("XML patching", test_xml_patching_effect()))
    results.append(("Zoo env vs manual", test_zoo_env_vs_manual(m_mj)))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\nAll tests PASSED - dynamics and rewards match between backends.")
    else:
        print("\nSome tests FAILED - investigate mismatches above.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
