# BigST/PatchSTG implementation and feasibility plan

Reviewed November 11, 2026. Architecture-review milestone complete. The work
originally proposed below was reviewed. The user subsequently approved PatchSTG
spatial/model implementation and CPU checks, now complete. See
[implementation record](patchstg_implementation.md). GPU execution and training
remain separate unapproved milestones; BigST scope remains pending.

## What the comparison can answer

Compare prediction accuracy and computational cost on SD under explicit input
budgets. Our recommendation is PatchSTG versus the named BigST short-history
variant described in [the BigST guide](papers/bigst.md). User confirmation of
that variant remains pending. Omitting LSFE and spatial regularization must
appear in result tables, README and configuration, not only a footnote.
Full BigST remains an alternative requiring a revised work/time budget.

No changes to source arrays, splits or baseline scores are needed for the
recommended variant. If extra history changes forecast eligibility, all compared
models and baselines must use common origins before claiming a controlled table.
Both model adapters should expose (B,12,716,1) normalized outputs; original-unit
loss/evaluation uses inverse-transformed predictions and untransformed targets.
The baseline evaluators remain unchanged.

## Completed implementation milestone: PatchSTG preprocessing and model tests

Keep preprocessing and forward behavior reviewable before adding a trainer.

| Proposed component | Responsibility | Acceptance check |
| --- | --- | --- |
| Spatial index utility | Coordinate splitting, reference similarity, padding and index maps | Known tiny graphs; every original sensor appears once in restoration; padding maps are in range |
| Provenance record | Source hashes, training boundary, scaler, segment length and metadata order | Cache cannot be reused under a different split or ordering |
| Embedding/attention modules | Paper/config mapping in patchstg.md | Shape checks, attention-axis tests, gradients and fixed-weight dense-reference checks |
| Forecast adapter | Shared sample input, normalized output, calendar expansion | Output shape and original-unit conversion checked with known values |
| Model smoke command | CPU forward/backward, save/reload, optional CUDA | Finite predictions/gradients and identical eval output after reload |

Start with [the code-aligned SD configuration](papers/patchstg.md), without silent
width/layer reduction. Preserve reference 288-step similarity segmentation for
the first implementation; a 96-step change would be separately labeled. Add
explicit guards for degenerate similarity and index/tie edge cases. Do not invent
an arbitrary graph when a preprocessing assumption fails; explain the case and
record any adaptation. Use NumPy/PyTorch already available where sufficient.

Write ordinary PyTorch attention independently and test its qkv/projection,
normalization, activation and dropout behavior. Do not substitute a generic
Transformer layer with different defaults. Include attribution and original MIT
notices if any source is adapted. Source terms and commit verification precede
copying. BigST source reuse is unresolved; prefer independent paper implementation.

## What we can estimate now

These are arithmetic estimates, not profiled memory or speed results. Derived
from the source dimensions in the paper guides:

- 512 capacity-two leaves hold 716 originals and 308 padded slots: 1,024 total.
- With 16 merged patches of size 64, depth and breadth score matrices contain
  16*64^2 + 64*16^2 = 81,920 entries per sample, temporal token, head and block.
- FP32 scores therefore occupy 0.3125 MiB per block/sample, or 1.5625 MiB across
  five blocks if one score tensor per operation is retained. Backpropagation
  saves other tensors too; this is not a peak VRAM estimate.
- One padded hidden representation occupies 1024*192*4 = 0.75 MiB per sample.
  Parameters, gradients, Adam states, Q/K/V, MLPs, buffers and allocator overhead
  add memory. A smaller score count alone does not establish a speedup.

For proposed BigST widths, one 716*64 random-feature array occupies about
0.175 MiB/sample in FP32. Other activations and states still need profiling.
Its random projection generation policy must be explicit: query-derived seeding
can depend on batch contents. Test batch-partition behavior and isolate global
RNG side effects; do not claim fixed projections reproduce that reference path.

## First GPU feasibility check — later, separately approved

After CPU tests pass, run full-width forward/backward at batch 1 on the host GPU,
then try batches 2 and 4 only if memory permits. Use the user's pml terminal if
agent GPU access remains restricted. Record exact GPU, software, config, seed,
precision and allocated/reserved peaks. Test an optimizer step as well as
forward-only memory. No AMP/compilation for initial checks. Synchronize CUDA for
measured timings. Do not promise batch 16 because it appears in author configs.

No GPU model smoke test has been run in this review. Environment CUDA success
is not evidence that either architecture trains within 8 GiB.

## Separate training milestone

Add differentiable original-unit masked MAE, skip all-zero-target training
batches without updating weights, fail on invalid predictions, and count skips.
Use pooled validation MAE and validation-only best-checkpoint selection. Resolve
optimizer/scheduler and epoch budget before runs: shared evaluation does not
require forcing identical optimizers when authors specify different ones.
Report model-specific training choices and a bounded validation tuning budget.

Lock dependencies and source/config provenance first. Time a short measured run
before allocating final epochs/seeds. Protect November 22–24 for analysis and
submission. With review completing November 11, earlier roadmap dates are now
historical targets; do not claim the original Nov 8–10 review target was met.

## Exit criteria for this review

Paper/component explanations, shape traces, discrepancy records, source/license
observations and a testable implementation plan are documented. Remaining user
choice is full BigST versus the explicitly named short-history variant. Source
pinning is partial and honestly recorded in [model_sources.json](model_sources.json).
Model implementation, numerical reproduction and hardware feasibility are not
claimed as completed review deliverables.
