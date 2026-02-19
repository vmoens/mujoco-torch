# Visualization Story

**Priority:** Low
**Category:** Feature/Docs
**Difficulty:** Low-Medium

## Problem

MJX has `mjx-viewer` for visualization. mujoco-torch has nothing. Users need to see what their simulation is doing, especially during debugging and reward shaping.

## What to Do

1. **Document the `device_get_into` workflow** — mujoco-torch already has `device_get_into` which copies torch Data back into a MuJoCo `MjData`. This can be used with MuJoCo's built-in viewer:
   ```python
   import mujoco, mujoco.viewer
   m_mj = mujoco.MjModel.from_xml_path("model.xml")
   d_mj = mujoco.MjData(m_mj)
   mx = mujoco_torch.device_put(m_mj)
   dx = mujoco_torch.make_data(mx)
   with mujoco.viewer.launch_passive(m_mj, d_mj) as viewer:
       for _ in range(1000):
           dx = mujoco_torch.step(mx, dx)
           mujoco_torch.device_get_into(d_mj, dx)
           viewer.sync()
   ```
2. **Create `examples/viewer_example.py`** demonstrating this pattern.
3. **Add to README** — a "Visualization" section explaining how to render.
4. **Optional**: Consider a thin wrapper `mujoco_torch.viewer.launch(mx, dx)` that handles the boilerplate. But don't over-engineer — the MuJoCo viewer is already good.

## Files to Touch

- `examples/viewer_example.py` — new
- `README.md` — add visualization section

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
