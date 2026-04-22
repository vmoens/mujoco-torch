# Release Runbook

Manual checklist to run **before publishing** a new version of
`mujoco-torch`.  None of these steps are wired into CI — they are
expected to be run on a machine with a CUDA GPU (any cluster is fine).
The goal is to catch expensive regressions (especially Dynamo
recompiles) that the default CI suite does not cover.

## 1. Run the default test suite

```bash
pytest
```

All tests must pass.  The default invocation skips tests marked
`integration` (see `pyproject.toml`).

## 2. Run the integration suite

```bash
pytest -m integration
```

This currently includes:

- `test/compile_recompile_integration_test.py` — verifies that
  `torch.compile(torch.vmap(step), fullgraph=True)` does **not**
  recompile across successive calls on each of the bundled environments
  (humanoid, halfcheetah, ant, hopper, walker2d, swimmer, cartpole).
  A recompile after the first call is the single most common
  performance regression in this codebase, and each recompile takes
  several minutes, so one bad PR can make training 10x slower without
  any functional failure.

Expected runtime is ~15–25 minutes per env on an H100/H200 GPU.  Run
overnight if needed.

### Torch version

`mujoco-torch` is developed and tested against **PyTorch nightly**.
Dynamo guard plumbing and vmap stride handling have been known to shift
between stable releases, so on a failure first confirm you're on a
recent nightly — stable torch is not a supported target for the
integration suite.

If the integration suite fails with a recompile, set
`TORCH_LOGS=recompiles` and re-run the specific parametrization to see
the guard that flipped.

## 3. Tag and publish

Only after both `pytest` and `pytest -m integration` are green:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

…and follow the usual PyPI / conda release steps.
