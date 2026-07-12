# One-Step Off-Policy Agent Instructions

These instructions apply to work under `verl/experimental/one_step_off_policy/`.
They supplement, but do not replace, the repository root `AGENTS.md`.

## Dynamic Resize Migration

### Goal

Migrate the dynamic resize feature from the old experimental branch onto the
current verl architecture without directly merging the old implementation. Reuse
pure control and metadata logic where practical, but adapt trainer, worker,
rollout, and checkpoint interactions to the current codebase.

### Migration Order

1. Build the minimum runnable schedule-resize path first.
   - Migrate pure Python control modules such as `resize_controller.py`,
     `resize_budget.py`, `resize_metrics.py`, and `trace_utils.py`.
   - Add `trainer.dynamic_resize` config parsing, schedule gating, and resize
     metrics to the current one-step-off trainer.
   - Support only the hard-switch baseline at this stage: save actor state via
     checkpoint or handoff, destroy and recreate the actor group and rollout
     server, then synchronize rollout weights through the current
     `CheckpointEngineManager.update_weights()` path.

2. Connect resource switching to the current separation architecture.
   - Extend the separation resource-pool construction to support
     `trainer.dynamic_resize.shared_pool` and actor/rollout splits from the
     resize schedule.
   - Have the trainer track the active actor/rollout topology and rebuild the
     corresponding `RayWorkerGroup`, `LLMServerManager`, and
     `CheckpointEngineManager` during resize.
   - Commit topology switches only at a full step boundary: after actor update
     and rollout weight update, before the next rollout batch starts.

3. Add performance optimizations only after the hard-switch path is stable.
   - `staging_backend.py` may be migrated early because it is primarily
     metadata.
   - Rewrite pinned CPU staging and optimizer deferred restore against the
     current `TrainingWorker` and engine APIs. Do not copy the old
     `fsdp_workers.py` implementation wholesale.
   - Add communicator cache last. First verify the current rollout and
     checkpoint manager lifecycle is stable, then reduce switch overhead.

### Current-Stage Restrictions

- Do not implement pinned CPU staging, communicator cache, optimizer async
  preload, or optimizer deferred restore until the minimum hard-switch path is
  implemented and verified.
- Do not replace the current unified engine worker architecture with old
  one-step-off worker copies.
- Do not change default one-step-off behavior when `trainer.dynamic_resize` is
  disabled.

### Git And Scope

- Keep changes narrowly scoped to dynamic resize migration and its required
  one-step-off/separation integration points.
- Preserve existing AGENTS instructions. Use `AGENTS.override.md` only when a
  dynamic resize rule must intentionally override an upper-level rule.
- Do not include unrelated refactors, broad formatting churn, or old branch
  history dumps in migration commits.

### Verification

For each phase, run the narrowest useful checks available in the environment:

- Import or config-load checks for newly added control modules and config keys.
- Unit tests for schedule gating, budget decisions, and metrics helpers when
  those modules are migrated.
- A small one-step-off dynamic resize smoke run when the hard-switch path is
  implemented, covering at least one actor/rollout topology change.
- Regression checks showing existing one-step-off behavior is unchanged when
  dynamic resize is disabled.

Record exact commands and results in the final response or PR description.

### Completion Standard

A phase is complete only when it has a runnable path, relevant metrics are
logged, disabled-by-default behavior is preserved, and the verification commands
for that phase have passed or their blocker is documented.
