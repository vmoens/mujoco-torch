# Add Pre-commit Hooks

**Priority:** Medium
**Category:** Infrastructure
**Difficulty:** Low

## Problem

No automated code quality checks. No `.pre-commit-config.yaml`, no linter config. For open-source, consistent formatting and basic lint catches are important.

## What to Do

1. Create `.pre-commit-config.yaml` with:
   - **ruff** — fast Python linter + formatter (replaces flake8, isort, black)
   - **trailing-whitespace**, **end-of-file-fixer**, **check-yaml**, **check-added-large-files** from `pre-commit-hooks`
2. Add a `[tool.ruff]` section to `pyproject.toml` (or create `ruff.toml`) with sensible defaults:
   - Line length: 88 or 100
   - Select rules: E, F, W, I (isort)
   - Ignore rules that conflict with existing style (check existing code first)
3. Run `pre-commit run --all-files` and fix any issues that surface.
4. Do NOT reformat the entire codebase in one giant commit — instead, configure ruff to match the existing style as closely as possible, only fixing clear errors.

## Files to Touch

- `.pre-commit-config.yaml` — new
- `pyproject.toml` or `ruff.toml` — ruff config
- Possibly minor whitespace fixes across the codebase

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
