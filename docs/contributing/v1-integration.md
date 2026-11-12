# Research v1 integration

Last updated: 11/10/2026.

`HanlinDu/verl:v1` is the research integration branch. All research pull requests
target `v1`. Upstream updates enter through a separately reviewed pull request.

The integration starts at `a91542dc59686d243843040dac4390c29a68c873`.
The shared source branch is archived at
`6bd480639824f1b4e03ed5597abd8720db7e3f58` with tag
`archive/dr-v1-20261105`. Keep that source branch intact.

The author batches merge in this order:

1. `work/v1-dingzigeng`: the hybrid-runtime and FSDP2 transfer changes from
   `c16a5723` and `4d4189a0`, plus migration corrections.
2. `work/v1-leonardw`: the nine follow-up commits in `4d4189a0..6bd48063`,
   covering cross-node enablement, teardown, optional imports, and acceptance.

The second batch depends on the first. Use merge commits so the first batch's
ancestry is preserved and the second PR's diff narrows after the first merges.
Both PRs target `v1`; neither batch merges into the source branch.

A further batch replays the eight commits in `4d4189a0..f3ef6ba3`
(shard-to-shard resize of the training runtime, rollout replica reuse) onto the
first batch. It does not contain the second batch and overlaps it; their
relative order is the maintainer's decision.

## Supported handoff contract

Ordinary one-step-off configurations retain their separate worker and NCCL
defaults. Hybrid workers require dynamic resize to be explicitly enabled.

The direct path (`handoff.backend=gpu_direct`) supports CUDA FSDP2, a
one-dimensional full-shard mesh, contiguous actor ranks starting at zero, and
both actor expansion and shrink. It moves parameters, per-parameter optimizer
state, module buffers, optimizer hyper-parameters and the LR-scheduler state
shard to shard from the old ranks to the new ones; a token-classification
critic follows the actor. No checkpoint is written or read for the switch, and
a failed switch aborts the run instead of falling back to disk. RNG state is
neither saved nor restored by a switch: every worker process keeps its own
stream, so bitwise reproducibility is not guaranteed. `VERL_RESIZE_VERIFY=1`
compares checksums of every moved tensor. Unsupported configurations are
listed under "Preconditions" in
`verl/experimental/one_step_off_policy/DYNAMIC_RESIZE.md`.

The checkpoint round trip (`handoff.backend=disk_fallback`) is unchanged: it
writes a full-model disk checkpoint, keeps a freshly built optimizer after a
world-size change, and restores scheduler/RNG extras with the rank-zero
fallback. It is weights-only continuation, not exact training-state
continuation. Pinned staging, optimizer async/deferred restore, and
communicator-cache work remain gated on a verified hard-switch baseline under
the existing area rules.

## Review and verification

Use `uv` with Python 3.12. Run the `v1 integration` CPU workflow and the applicable
repository pre-commit checks. CPU checks validate control and teardown contracts;
they do not certify CUDA or cross-node transport.

Before merging, record a real hard-switch smoke run, a disabled-resize regression,
and direct-transfer state comparisons. The cross-node batch also needs actual
cross-host payload evidence, post-resize weight synchronization and training,
and a clean terminal event after both scheduled expansions.

The human submitter must review every changed line and run relevant tests under
the repository contribution policy. Record AI assistance and source SHAs in both
PR descriptions. A draft or a CPU-only pass does not satisfy GPU acceptance.

After reviewers and working checks are available, protect `v1` against direct
pushes, force pushes, and deletion, require PR review and status checks, and
apply the rule to administrators. Configure an independent eligible code owner
before requiring CODEOWNER approval. Change the fork default branch to `v1`
after integration acceptance.
