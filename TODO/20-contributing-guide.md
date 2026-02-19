# Add Contributing Guide

**Priority:** Low
**Category:** Docs
**Difficulty:** Low

## Problem

No `CONTRIBUTING.md`, no issue templates, no PR template. Blocks community contributions.

## What to Do

1. **Create `CONTRIBUTING.md`** with:
   - How to set up a dev environment (`source .venv/bin/activate`, `pip install -e ".[test]"`)
   - How to run tests (`pytest test/ -x -v`)
   - Code style expectations (match existing style, or ruff config if TODO/09 is done)
   - How to add a new collision type
   - How to add a new sensor type
   - How to port a feature from MJX (the general pattern)
   - PR expectations: tests required, match MuJoCo C output
2. **Create `.github/ISSUE_TEMPLATE/`** with:
   - `bug_report.md` — model XML, expected vs actual, MuJoCo C comparison
   - `feature_request.md` — what MJX feature, priority
3. **Create `.github/PULL_REQUEST_TEMPLATE.md`** with:
   - Description
   - Test plan
   - MuJoCo C comparison results

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
