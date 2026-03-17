# Contributing to mujoco-torch

Thanks for your interest in contributing! This guide covers the basics.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

For `torch.compile(fullgraph=True)` support, build PyTorch from the
[`mujoco-torch-features`](https://github.com/vmoens/pytorch/tree/mujoco-torch-features)
branch (see the README for details).

## Running tests

```bash
pytest test/ -x -v
```

Some tests compare against MuJoCo MJX (JAX). Install the optional deps first:

```bash
pip install "jax[cpu]" "mujoco[mjx]"
```

## Code style

- The project uses [ruff](https://docs.astral.sh/ruff/) for linting and
  formatting.  Configuration lives in `pyproject.toml` (line length 120).
- Run `ruff check .` and `ruff format --check .` before submitting.
- If pre-commit hooks are installed (`pip install pre-commit && pre-commit install`),
  checks run automatically on each commit.

## General guidelines

- **Match MuJoCo C output** — every physics feature must be validated against
  `mujoco.mj_step` (or the relevant `mj_*` function) at float64 precision.
- **Tests are required** — bug-fixes and features must include a test in the
  same commit.
- **Avoid numpy on hot paths** — anything called during `step()` must use
  PyTorch tensors so that `torch.compile` can trace through it.  Numpy is fine
  for one-time model-loading code in `device.py`.
- **Top-level imports** — avoid function-level imports unless required to break
  a circular dependency.

## Porting a feature from MJX

The general pattern for adding an MJX feature:

1. Find the MJX implementation in
   [`mujoco/mjx/_src/`](https://github.com/google-deepmind/mujoco/tree/main/mjx/mujoco/mjx/_src).
2. Translate JAX ops to PyTorch equivalents (`jnp.` → `torch.`,
   `jax.vmap` → `torch.vmap`, etc.).
3. Add any new types or enum members to `mujoco_torch/_src/types.py`.
4. Update validation in `mujoco_torch/_src/device.py` if the feature
   requires accepting new model options.
5. Write a test that compares output against MuJoCo C.

## Adding a new collision type

1. Implement the collision function in `mujoco_torch/_src/collision_primitive.py`
   (or a new file for complex types like `collision_hfield.py`).
2. Register the geom-pair entry in `_COLLISION_FUNC` in
   `mujoco_torch/_src/collision_driver.py`.
3. Add a test model and compare contact positions/normals/depths against
   MuJoCo C.

## Pull requests

- One logical change per PR.
- Include a clear description and test plan.
- If your change touches multiple steps, consider using
  [ghstack](https://github.com/ezyang/ghstack) to submit stacked PRs.
