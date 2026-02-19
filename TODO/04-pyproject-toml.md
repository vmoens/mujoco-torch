# Migrate to pyproject.toml

**Status**: solved
**Priority:** High
**Category:** Infrastructure
**Difficulty:** Low

## Problem

The project uses a legacy `setup.py` for packaging. Modern Python packaging (PEP 517/518/621) uses `pyproject.toml`. This matters for:
- Proper build isolation
- Correct metadata for PyPI
- Tool configuration (pytest, ruff, mypy, etc.) in one place

## What to Do

1. Create `pyproject.toml` with:
   - `[build-system]` using setuptools
   - `[project]` metadata migrated from `setup.py` (name, version, description, dependencies, python_requires, license, etc.)
   - `[project.optional-dependencies]` for test extras
   - `[tool.pytest.ini_options]` if any pytest config is needed
2. Keep `setup.py` as a minimal shim or remove it entirely (prefer removal).
3. Verify `pip install -e .` and `pip install -e ".[test]"` still work.
4. Verify `pytest test/ -x -v` still passes.
5. Update `.github/workflows/tests.yml` if the install command changes.
6. Use proper ruff linting

## Reference

Current `setup.py` content:
- name: `mujoco-torch`
- version: `0.1.0`
- python_requires: `>=3.10`
- install_requires: torch>=2.1, mujoco>=3.0, numpy, tensordict>=0.11, absl-py, etils, scipy, trimesh
- extras: test → pytest
- package_data: `test_data/**/*`

## Submission Instructions

- branch out from origin/main. Commit only changed / relevant new files.
- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- If the changes are contained in one single commit, use `gh pr` instead.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
