# Add NOTICE File for Attribution

**Priority:** Critical
**Category:** Legal/Docs
**Difficulty:** Low

## Problem

The source code has DeepMind copyright headers (e.g. "Copyright 2023 DeepMind Technologies Limited") because it's ported from MJX. The Apache 2.0 license requires a NOTICE file when redistributing derivative works. Without it, the licensing story is unclear and could block adoption.

## What to Do

1. Create a `NOTICE` file at the repo root with:
   - Statement that this project is a derivative work of MuJoCo MJX
   - Original copyright: "Copyright 2023 DeepMind Technologies Limited"
   - Link to the original: https://github.com/google-deepmind/mujoco/tree/main/mjx
   - The original license (Apache 2.0)
   - Your own copyright for new additions
2. Ensure the existing `LICENSE` file is present and is the full Apache 2.0 text.
3. Update file headers for new files you've written (not ported from MJX) to use your own copyright.

## Reference

Apache 2.0 Section 4(d):
> If the Work includes a "NOTICE" text file [...] You must include a readable copy of the attribution notices contained within such NOTICE file.

## Submission Instructions

- If your changes have more than one step, use ghstack to submit. ghstack sends each commit as a separate PR, so make sure each commit message is a proper PR name.
- BugFix and features must have a test in the same commit.
- You can submit PRs; if you do, monitor the runs using gh.
