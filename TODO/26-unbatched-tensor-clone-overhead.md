# UnbatchedTensor clone overhead under torch.compile

## Observation

PR #29 (`smooth-perf-cleanup`) moved tendon precomputation from inline
numpy/torch calls in `smooth.tendon()` to precomputed `UnbatchedTensor` fields
on `Model`.  This is the right architectural move (avoids recomputing constants
every step, removes numpy from the hot path), but it introduced a measurable
runtime regression under `torch.compile(fullgraph=True)`:

| walker2d batch | main | PR #29 | delta |
|--:|--:|--:|--:|
| 1 | 78 | 66 | -15% |
| 128 | 7,481 | 6,479 | -13% |
| 1,024 | 54,458 | 48,209 | -11% |
| 4,096 | 197,898 | 184,693 | -7% |
| 32,768 | 527,044 | 504,676 | -4% |

An A/B test reverting only the `scatter()` → `clone()+indexing` change (keeping
`UnbatchedTensor`) recovered only ~4-5%, confirming the regression comes from
`UnbatchedTensor`, not from `scatter`.

## Hypothesis

Every internal `torch.vmap` call clones its non-batched inputs (`in_dims=None`).
In `mujoco_torch`, the `Model` is passed as a non-batched arg to many vmaps:

- `xfrc_accumulate` → `torch.vmap(apply_ft, (None, None, 0, 0, 0, 0))`
- Every `_nvmap` call in `scan.flat()` and `scan.body_tree()`

Each of these clones visits **all** Model fields.  Before PR #29, the tendon
indices didn't exist on the Model.  After, there are 7 new `UnbatchedTensor`
fields (`tendon_has_jnt`, `tendon_qposadr_jnt`, `tendon_moment_jnt`,
`tendon_segment_ids`, `tendon_tendon_id_jnt`, `tendon_adr_moment_jnt`,
`tendon_dofadr_moment_jnt`).

`UnbatchedTensor.clone()` is more expensive than a plain tensor clone because it
goes through the tensorclass machinery (`__init__` → `_clone` → reconstruct
wrapper`).  Under Dynamo tracing, this emits extra ops into the compiled graph.
Multiplied across hundreds of vmap calls per step, the overhead adds up.

## Possible mitigations

1. **Move tendon precomputed data to `_device_precomp` dict** instead of Model
   fields.  `_device_precomp` is excluded from the traced graph and not cloned
   by vmap.  The existing `factor_m_updates`, `solve_m_updates_j/i` data already
   lives there.

2. **Reduce the number of vmap clone calls**.  If `in_dims=None` args are known
   constant across the vmap, the vmap implementation could skip cloning them.
   This would require upstream changes in `torch._functorch`.

3. **Make UnbatchedTensor.clone() cheaper under Dynamo**.  The tensorclass
   constructor overhead could be reduced with a `__torch_function__` fast path
   or a Dynamo-specific `clone` implementation.

4. **Batch the 7 tendon index tensors into fewer fields** (e.g. a single stacked
   tensor with named slices) to reduce the per-field clone cost.

## Status

Filed as a tensordict bug (`UNBATCHED_TF_BUG.md` in the tensordict repo).  The
fix belongs in `UnbatchedTensor.__torch_function__` — the tendon data should
stay as `UnbatchedTensor` on Model since that's architecturally correct.

Blocked on the tensordict fix.  The regression is moderate (4-15% depending on
batch size) and the PR's correctness benefits (removing numpy from hot path,
precomputing constants) are worth keeping.
