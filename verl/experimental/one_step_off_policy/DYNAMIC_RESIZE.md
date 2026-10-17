# Dynamic resize in the hybrid one-step-off trainer

This document describes how `verl.experimental.one_step_off_policy` moves GPUs between rollout and
training while a run is in progress ("dynamic resize"): what a resize is, how to run the single-node
example, what happens during a hard switch, and how the result is verified. It is the reader's guide to
the code; the commit messages carry the per-change rationale.

Related files:

* `README.md` -- the one-step-off recipe itself (pipeline, parameter sync, configuration).
* `DYNAMIC_RESIZE_DIRECT_RESHARD_PLAN.md` -- the original design plan; its section 0 tracks which plan
  items are implemented on this branch. Sections 1-18 of that file predate the implementation.
* `config/one_step_off_ppo_trainer_dynamic_resize.yaml` -- the configuration the example uses.
* `shell/run_dynamic_resize.sh` -- the single-node example launcher.

## What a resize is

A one-step-off run has two kinds of GPU work: **rollout** (vLLM generating responses) and **training**
(the actor, plus critic/reference if configured). Their ideal GPU shares change over a run, so the trainer
can re-partition a **shared pool** of GPUs between them at step boundaries.

* `hybrid_engine=true`: one `ActorRolloutRefWorker` process per GPU covers the whole pool. Each process
  holds a training engine, a rollout adapter and a weight-sync (checkpoint-engine) client. Processes are
  never killed by a resize; only their *roles* change.
* Layout: training ranks are always the prefix `range(actor_size)` of the pool, rollout ranks the suffix.
  `trainer.dynamic_resize.shared_pool.n_gpus_per_node` is the pool size; every schedule stage must satisfy
  `actor_size + rollout_size == pool size`.
* Rollout is organised in **replicas**: one vLLM engine per `tensor_model_parallel_size` consecutive GPUs
  (`rollout.yaml` defaults to TP=2, so a replica is a GPU pair). `replica_rank = physical_rank // TP`;
  the server actor is named `server_{replica_rank}_{node_rank}`. The split point must be a multiple of the
  replica size (`_validate_hybrid_split`), so a replica is either wholly rollout or wholly training.
* `trainer.dynamic_resize.mode=schedule`: `schedule.stageN` entries (`step`, `actor_pool.size`,
  `rollout_pool.size`, ...) say which topology applies from which step; `stage0` with `step: -1` is the
  initial split. `hard_switch.enable=true` makes the trainer actually move resources; otherwise the switch
  is only recorded as deferred.
* The **hard switch** runs at the end of a training step, after `update_actor`, with no rollout in flight
  (the step that resizes does not pre-launch the next generation). Training state that has to survive it:
  fp32 parameters, optimizer state, LR-scheduler state and module buffers, all FSDP2-sharded along dim 0,
  so a change of world size changes every shard boundary.

## Running the single-node example

```bash
bash verl/experimental/one_step_off_policy/shell/run_dynamic_resize.sh
```

The script converts a GSM8K parquet pair into the verl format (under `$CKPT_DIR/data`), turns the
environment variables below into hydra overrides and starts
`python -m verl.experimental.one_step_off_policy.main_ppo --config-name=one_step_off_ppo_trainer_dynamic_resize.yaml`.
Any extra command-line arguments are forwarded to hydra unchanged (`"$@"`), so arbitrary overrides can be
appended, e.g. `bash .../run_dynamic_resize.sh trainer.total_training_steps=3`.

| Variable | Default | Meaning |
|---|---|---|
| `NUM_GPUS` / `SHARED_POOL_GPUS` | 8 / `NUM_GPUS` | pool size |
| `N_GPUS_ROLLOUT` | 4 | initial rollout GPUs; training gets `NUM_GPUS - N_GPUS_ROLLOUT` |
| `RESIZE_STEP` | 2 | step of the first switch; the second is `RESIZE_STEP + 2` |
| `FIRST_SPLIT_PLAN` / `SECOND_SPLIT_PLAN` | `[6,2]` / `[2,6]` | `[actor, rollout]` sizes of the two switches |
| `TOTAL_TRAINING_STEPS` | second switch + 2 | run length |
| `MODEL_PATH`, `TRAIN_FILE`, `VAL_FILE`, `CKPT_DIR` | site paths | model, data and output locations |
| `TRAIN_BATCH_SIZE` / `PPO_MINI_BATCH_SIZE` | 960 / 240 | must divide evenly by every actor size in the schedule |
| `USE_KL_LOSS` | false | `true` adds a reference model, which `gpu_direct` rejects at start-up (see *Preconditions*) |
| `HARD_SWITCH_ENABLE` | true | actually move resources |
| `HOST_STAGING_ENABLE` / `HOST_STAGING_BACKEND` | true / `gpu_direct` | `handoff.enable` / `handoff.backend` |
| `HYSTERESIS_*`, `BUDGET_PROTECTION_ENABLE`, `COMM_CACHE_ENABLE` | off | optional controller, memory gate, communicator cache |
| `PROFILER_ENABLE` | true | `global_profiler.enabled`; set `false` for quick iterations |
| `VERL_RESIZE_VERIFY` | unset | `1` selects the verification version of the launcher, see below |

The yaml's `stage0`, the initial split, is `[4,4]`. A pool or an initial split other than 4 + 4 therefore
also needs `trainer.dynamic_resize.schedule.stage0.actor_pool.size=[actor,rollout]` and
`...stage0.rollout_pool.size=[actor,rollout]` on the command line, next to `NUM_GPUS` and `N_GPUS_ROLLOUT`.

The launcher has two versions, and this one variable selects between them:

* **plain** (`VERL_RESIZE_VERIFY` unset) -- the launcher starts training and nothing else. A switch still
  passes the checks that are always on (see *Verification*), and verl logs warnings and errors only.
* **verification** (`VERL_RESIZE_VERIFY=1`) -- the same training run, and in addition:
  1. Every `gpu_direct` switch compares the new runtime with the old one before the old one is dropped:
     checksums of every parameter, optimizer tensor and buffer, and the scalar state (`step`,
     `param_groups`, LR scheduler) entry by entry. A difference fails the run.
  2. verl logs everything: `VERL_LOGGING_LEVEL` defaults to `DEBUG` (a value the caller exports is
     kept). That includes the trainer's `[one-step-off][resize]` lines with the transfer statistics of
     every rank and the workers' memory lines.
  3. The resize trace is on (`trainer.dynamic_resize.trace`): `$CKPT_DIR/resize_trace.jsonl` gets two
     records per step, `resize_observation` and `resize_schedule`, with all `resize/*` metrics of the
     step.

More switches are added as further stages with the same shape as `stage0` in the yaml:

```bash
S=trainer.dynamic_resize.schedule
TOTAL_TRAINING_STEPS=8 bash .../run_dynamic_resize.sh \
  +$S.stage3.step=6 +$S.stage3.actor_pool.mode=split +$S.stage3.actor_pool.from_pool=shared_pool \
  +$S.stage3.actor_pool.index=0 +$S.stage3.actor_pool.size=[6,2] \
  +$S.stage3.rollout_pool.mode=split +$S.stage3.rollout_pool.from_pool=shared_pool \
  +$S.stage3.rollout_pool.index=1 +$S.stage3.rollout_pool.size=[6,2] +$S.stage3.release_old=false
```

## Environment requirements and test commands

Run everything from the repository root of a clean checkout, with `export PYTHONPATH=$PWD`. What each
check proves is described under *Verification*.

| Check | Needs | Reference environment |
|---|---|---|
| CPU tests | Python 3.12 and the packages listed under *CPU tests*; no accelerator | Python 3.12.3, torch 2.8.0+cpu, ray 2.49.2, transformers 4.57.1, tensordict 0.10.0, numpy 2.5.4, pytest 9.2.0 |
| Transfer test | one node with 6 CUDA GPUs; a CUDA build of torch with FSDP2 and NCCL | 6 of the 8 GPUs below, same software |
| Optimizer check | the same, and a causal LM whose fp32 training step fits on the GPUs | 6 of the 8 GPUs below, same software; Qwen3-0.6B |
| End to end | one node with 8 CUDA GPUs (the launcher's default pool); vLLM and Ray; `flash_attn` when a critic is trained; a causal LM, a GSM8K parquet pair and a writable `CKPT_DIR` | 8x RTX 5090 (32 GB), NVIDIA driver 580.82.09; Python 3.12.12, torch 2.9.0+cu129 (CUDA 12.9, NCCL 2.27.5), vLLM 0.12.0, Ray 2.53.0, transformers 4.57.5, tensordict 0.10.0, hydra-core 1.3.2, numpy 1.26.4, flash_attn 2.8.1; Qwen3-0.6B, GSM8K in verl format |

### CPU tests

In a Python 3.12 virtual environment. The first command and the packages up to `psutil` are what the
`v1 integration` workflow installs for the three files it runs; the packages after `psutil` are imported
by the other files. Keep the second command in one piece: installed in a separate step, `datasets`
upgrades `huggingface-hub` beyond what `transformers==4.57.1` accepts.

```bash
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install pytest pytest-asyncio hydra-core numpy packaging tensordict==0.10.0 transformers==4.57.1 \
  ray==2.49.2 psutil codetiming peft datasets torchdata cachetools pillow uvicorn fastapi
```

```bash
python -m pytest -q -rs tests/experimental/one_step_off_policy \
  tests/workers/test_training_worker_resize_snapshot_on_cpu.py \
  tests/utils/test_fsdp_resize_on_cpu.py tests/utils/test_fsdp_load_scope_on_cpu.py \
  tests/workers/rollout/test_replica_on_cpu.py
```

No test may fail; one is skipped because it needs an accelerator.

### Transfer test (6 GPUs)

```bash
torchrun --standalone --nproc-per-node=6 tests/special_distributed/run_fsdp2_resize_transfer.py
```

Rank 0 prints `FSDP2 GPU-direct resize transfer 4->6->2->6 passed (3 scalar-state faults caught)`.

### Optimizer check (6 GPUs)

```bash
MODEL_PATH=<model directory> torchrun --standalone --nproc-per-node=6 \
  tests/special_distributed/run_fsdp2_resize_optimizer.py
```

Rank 0 prints every expectation as a `PASS` or `FAIL` line, twelve of them for the default sizes
`4,6,2`. The run passes when it exits with code 0 and its last line is
`FSDP2 GPU-direct resize optimizer check 4->6->2 passed`; any `FAIL` makes it exit non-zero.
`RESIZE_SEQUENCE=4,6,4` selects other sizes (and needs as many processes as the largest of them),
`OUT_DIR=<directory>` also writes everything that was measured to `<directory>/optimizer_check.json`.

### End to end (8 GPUs)

```bash
export MODEL_PATH=<model directory> TRAIN_FILE=<gsm8k train.parquet> VAL_FILE=<gsm8k test.parquet>
export PROFILER_ENABLE=false TRAIN_BATCH_SIZE=96 PPO_MINI_BATCH_SIZE=48
export TORCH_CUDA_ARCH_LIST=12.0   # RTX 5090 (sm_120) only, as in the reference environment
L=verl/experimental/one_step_off_policy/shell/run_dynamic_resize.sh
G="algorithm.adv_estimator=grpo actor_rollout_ref.rollout.n=5 actor_rollout_ref.actor.entropy_coeff=0"
```

`$G` selects GRPO, which trains no critic; without it the launcher runs GAE with the token-classification
critic. The two batch sizes keep a run short. `$RUN` is a new, empty directory for each run.
`VERL_RESIZE_VERIFY=1` selects the verification version of the launcher (see *Running the single-node
example*); the second command is the plain version of the first.

```bash
# resize: [4,4] (actor, rollout) -> [6,2] at step 2 -> [2,6] at step 4, six steps
VERL_RESIZE_VERIFY=1 CKPT_DIR=$RUN/ckpt bash $L $G
# the same run with the plain version
CKPT_DIR=$RUN/ckpt bash $L $G
# the same with GAE and the critic
VERL_RESIZE_VERIFY=1 CKPT_DIR=$RUN/ckpt bash $L
# back to an earlier topology: [4,4] -> [6,2] -> [4,4]
VERL_RESIZE_VERIFY=1 SECOND_SPLIT_PLAN="[4,4]" CKPT_DIR=$RUN/ckpt bash $L $G
# the two default switches through the checkpoint round trip
HOST_STAGING_BACKEND=disk_fallback CKPT_DIR=$RUN/ckpt bash $L $G
# dynamic resize off: separate actor and rollout workers on 4 + 4 GPUs
CKPT_DIR=$RUN/ckpt bash $L $G actor_rollout_ref.hybrid_engine=false trainer.dynamic_resize.enable=false
```

A run passes when the launcher exits with code 0 after its last step line (`step:6`) and every switch
line (`step:2` and `step:4`) reports `resize/hard_switch_success:1.0`; with resize off every step line
reports `resize/enabled:0.0` instead. In addition:

* Verification version: every switch line reports `resize/gpu_direct_verified:1.0`, with the critic also
  `resize/gpu_direct_critic_verified:1.0`. Before each switch line the log has
  `[one-step-off][resize] actor runtime moved [0, 1, 2, 3] -> [0, 1, 2, 3, 4, 5] ranks on GPU: ...
  verified=1 verified_scalars=330` with the ranks of that switch (330 entries for the Qwen3-0.6B actor;
  with the critic a `critic runtime moved` line as well) and one
  `[one-step-off][resize] actor rank N: {...}` line per GPU of the pool.
  The move at start-up (pool -> initial split, see the end of *The hard switch, step by step*) is
  logged the same way before the first step, as `[0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 2, 3]` with
  `verified=1` and fewer scalar entries: no optimizer step has been taken yet.
  `$RUN/ckpt/resize_trace.jsonl` has two lines per step.
* Plain version: no `[one-step-off][resize]` line and no `resize_trace.jsonl`; the switch lines report
  `resize/gpu_direct_verified:0.0`.
* `gpu_direct` leaves no `global_step_*` directory in `CKPT_DIR`; `disk_fallback` leaves one per switch.

## The hard switch, step by step

`OneStepOffRayTrainer._apply_dynamic_resize_hard_switch` with `handoff.backend=gpu_direct`, for a switch
to `(actor_target, rollout_target)`. Nothing is written to or read from disk, and nothing is staged on the
host; the training state moves GPU to GPU.

1. **Quiesce rollout** (`_destroy_dynamic_resize_runtime`): abort in-flight requests on every replica
   (`sleep_replicas` is a no-op under the recipe's `free_cache_engine=False`) and kill the agent-loop
   workers. The vLLM server actors stay up for now.
2. **Move the training state** (`_rebuild_hybrid_training_runtime` -> `resize_actor_runtime` on every
   physical rank): build the new FSDP2 runtime on the new ranks next to the live one, fill it shard to
   shard with one `all_to_all_single` per tensor, swap, destroy the old groups. The critic goes through the
   same `resize_runtime` call right after the actor. Details under *Training state: shard-to-shard
   transfer*.
3. **Re-point the control plane**: `actor_wg` / `critic_wg` become `RayWorkerGroup.subgroup(new actor ranks)`
   views (no Ray actors are created or destroyed), the rollout view becomes the suffix ranks, and the
   hybrid workers drop their cached server handles (`reset_rollout_server_handle`).
4. **Rebuild rollout**: `LLMServerManager.rebuild_hybrid_runtime` kills every vLLM server actor and
   launches new ones on the new rollout ranks, then rebuilds the load balancer and the agent-loop manager.
   This is the dominant cost of a switch (about 34 s of vLLM engine start-up for Qwen3-0.6B).
5. **Weight sync**: `reconfigure_checkpoint_engine(group_name, rebuild_group=True)` finalizes the old NCCL
   weight-sync engine and creates a new one on every rank; `CheckpointEngineManager.update_weights`
   pushes the current actor weights to the replicas and resumes generation.
6. The trainer records the new topology and launches the next step's generation on the new rollout ranks.

At startup the actor is still built on all physical ranks (upstream `init_model`) and then moved onto the
initial actor ranks with the same transfer (`_rebuild_hybrid_training_runtime(old_ranks=range(pool))`),
so a default run shows three transfers: pool -> initial split, then the two scheduled switches.

## Training state: shard-to-shard transfer

### Layers

| Layer | File | Responsibility |
|---|---|---|
| trainer | `one_step_off_policy/ray_trainer.py` | decides the topology; `_resize_actor_runtime_gpu_direct` / `_resize_critic_runtime_gpu_direct` call the workers, check that exactly `actor_target` ranks report `member` with identical tensor counts, aggregate `resize/gpu_direct_*` metrics |
| worker | `verl/workers/engine_workers.py` | `TrainingWorker.resize_runtime(old_ranks, new_ranks)` runs on *every* physical rank (`ONE_TO_ALL`): group creation, prepare, verdicts, transfer, commit, teardown; `ActorRolloutRefWorker.resize_actor_runtime` wraps it for actor and co-located ref |
| engine | `verl/workers/engine/fsdp/transformer_impl.py` | `FSDPEngine.prepare_resize_target` / `transfer_resize_state` / `commit_resize` / `abort_resize`; `check_gpu_direct_resize_supported`; `_build_without_weights` branches of `_build_module` / `_build_fsdp_module` |
| transfer | `verl/utils/fsdp_resize.py` | `FSDPResizePlanner` (row intervals), `transfer_fsdp2_runtime` / `_Transfer` (manifest, `all_to_all_single`, broadcasts, optimizer/scheduler metadata, checksums, scalar-state comparison), `raise_if_any_failed` |

### What moves

`_build_manifest` on `old_ranks[0]` lists everything; the manifest is broadcast and every rank checks its
old and new runtime against it (`_check_structure`) before any data moves.

| Item | How | Why it is needed |
|---|---|---|
| parameters with `Shard(0)` placement | one `all_to_all_single` per tensor, old local shard as send buffer, new local shard as receive buffer | the weights themselves; tied weights are moved once (`named_parameters()` de-duplicates) |
| replicated parameters / plain tensors | `broadcast` from `old_ranks[0]` | same |
| **all** `named_buffers()`, including non-persistent ones such as rotary `inv_freq` | `broadcast` | the new module is built on the meta device and `to_empty`'d, so buffers are uninitialised too; `state_dict()` would miss the non-persistent ones |
| optimizer state tensors (`exp_avg`, `exp_avg_sq`) | same planner and `all_to_all_single` as the parameter (same shape, same sharding); the new side first allocates `zeros_like(param)` via `_ensure_optimizer_state` | exact training continuation; without them Adam restarts from zero moments |
| optimizer scalars (`step`), `param_groups` hyper-parameters (minus `params`), `lr_scheduler.state_dict()` | metadata in the manifest (`broadcast_object_list`), applied on the new side | bias correction, current LR / weight decay, warm-up position |

Not moved, and not needed: gradients (the switch happens after `optimizer.step()` and `zero_grad`), FSDP2's
own bookkeeping (recreated by `fully_shard`), RNG state (training has no dropout; sampling lives in vLLM),
dataloader position (lives in the trainer process).

### Why one `all_to_all_single` per tensor is enough

FSDP2 shards every parameter along dim 0 with `torch.chunk` semantics (`Shard._local_shard_size_and_offset`),
so each rank's local shard is a contiguous row interval and the intervals are ordered by mesh rank. With
`old_ranks` and `new_ranks` both sorted:

* the rows an old rank must send to successive new ranks are consecutive sub-intervals of its own shard, in
  order -- `FSDPResizePlanner.send_row_splits` partitions the old shard;
* the rows a new rank receives from successive old ranks are consecutive sub-intervals of its new shard, in
  order -- `recv_row_splits` partitions the new shard.

`all_to_all_single(recv, send, output_split_sizes, input_split_sizes, group=old ∪ new)` therefore uses
`old_param.to_local().view(-1)` and `new_param.to_local().view(-1)` directly (`to_local()` returns FSDP2's
padded sharded storage minus the padding, which is the tensor's permanent home). Send and receive sizes
differ in general; the only constraints are `send.numel() == sum(input_split_sizes)` and
`recv.numel() == sum(output_split_sizes)`, both asserted. Ranks on one side only pass a zero-length buffer
and an all-zero split vector. Rows a rank keeps are NCCL's self chunk, an on-device copy
(`resize/gpu_direct_self_copy_bytes`); every other row crosses the wire exactly once.

### Worker protocol (`TrainingWorker.resize_runtime`)

Every physical rank executes the same sequence; non-members "accompany" the collectives that need WORLD.

1. `barrier`; `new_group(new_ranks)` and `new_group(old ∪ new)` in the same order on all ranks.
2. New ranks: `engine.prepare_resize_target(new_pg)` -- save the live runtime attributes, switch
   `process_group`, set `_build_without_weights=True`, `_init_device_mesh()` (`DeviceMesh.from_group`),
   `_build_model_optimizer()` (`from_config` on the meta device -> `fully_shard` -> `module.to_empty(device)`,
   optimizer and scheduler with empty state), park the result in `_resize_target`, restore the live runtime.
   Old and new runtimes now coexist in the process.
3. Second verdict exchange: every new rank's target exists, or everyone raises.
4. `old ∪ new` ranks: `engine.transfer_resize_state(old_ranks, new_ranks, group)` ->
   `transfer_fsdp2_runtime`: `reshard()` both modules, manifest, structure check, parameters, optimizer
   state, buffers, optional verification (checksums and scalar state), `synchronize` + `barrier`. Status
   per rank: `member` (in the new group), `sender` (old only), `non_member`.
5. Record `peak_alloc_bytes` (old and new shards coexist), `engine.commit_resize()` (shut the live runtime
   down, install the target, rebuild `FSDPCheckpointManager`), destroy the old training group and the
   transfer group, `clear_dtensor_dispatch_caches()` on every rank, refresh dispatch info, `barrier`.

Any exception -> `engine.abort_resize()` drops the half-built target and re-raises. There is no fallback
path: a failed switch fails the run rather than silently training on wrong state.

### Preconditions (`check_gpu_direct_resize_supported`, checked at startup)

* `actor.strategy=fsdp2` with `fsdp_size=-1` (one-dimensional mesh, `Shard(0)` only);
* parameters and optimizer state on GPU (no `param_offload` / `optimizer_offload` / FSDP2 `CPUOffloadPolicy`);
* no LoRA, `ulysses_sequence_parallel_size=1`, CUDA;
* the critic, if any, is an `AutoModelForTokenClassification` model (trl value-head critics are rejected);
* no reference model. A separate reference worker group is rejected. One co-located in
  `ActorRolloutRefWorker` (`use_kl_loss=true`) is rejected as well with the shipped reference
  configuration: its engine is forward-only, FSDP2 gives such an engine `CPUOffloadPolicy`, and that
  fails the condition on offloading above. `resize_actor_runtime` would move a reference runtime that
  keeps its parameters on GPU right after the actor, but no test has reached that code;
* every actor and rollout size in the schedule is a multiple of the replica size (`_validate_hybrid_split`).

## Process groups and DTensor caches across resizes

Every hard switch destroys the training subgroup and the weight-sync engine and creates new ones. Two
things break on the *second* switch back to a topology that existed before:

* `DeviceMesh.__eq__` / `__hash__` compare content (rank list, shape, dim names) and ignore the process
  group, which the mesh references by name. DTensor's sharding propagator and redistribute planner cache
  results keyed on specs that contain the mesh. After a subgroup is destroyed and a later switch recreates
  one with identical ranks, the cache hits and returns a spec whose mesh resolves to the dead group:
  `Could not resolve the process group registered under the name N` at the first DTensor collective
  (gradient clipping is the first). `clear_dtensor_dispatch_caches()` (`verl/utils/distributed.py`) clears
  `propagate_op_sharding` (a `LocalLRUCache` in torch >= 2.9, hence `.cache`), `_propagate_tensor_meta` and
  `_gen_transform_infos`; `TrainingWorker.destroy_process_group` and `resize_runtime` call it on **every** rank, members or not.
* `NCCLCheckpointEngine.finalize()` runs twice with `rebuild_group=True` (after each `update_weights`, and
  again from `reconfigure_checkpoint_engine` when a resize retires the engine) and compared `None >= 0`.
  It now treats `rank is None` as "never joined".

Rules the hybrid runtime imposes: all `new_group` calls happen on every rank in the same order, every
worker method that touches groups is `ONE_TO_ALL` with barriers, and non-member ranks still make the calls.

## Verification

Always on:

* **structure** -- manifest from `old_ranks[0]`; every rank checks names, shapes, dtypes and placements of its
  old and new runtime (and optimizer state) against it; verdicts are exchanged so all ranks raise together.
* **sizes** -- the receive buffer is the real FSDP2 shard, so `numel == sum(splits)` checks that the planner
  and FSDP2's chunking agree; the trainer checks that every member reports the same tensor counts.

Opt-in, `VERL_RESIZE_VERIFY=1` (metric `resize/gpu_direct_verified`, `resize/gpu_direct_critic_verified`):

* `_shard_digest` computes, per tensor, two int64 sums over the raw 32-bit words of the local shard:
  `sum(v_i)` and `sum(v_i * ((global_index_i mod 2^20) + 1))`. Because the weight is the *global* flat index,
  shards of one tensor combine by plain addition however the tensor is cut, so each side reduces its digests
  inside its own mesh with one `all_reduce` -- no all-gather, no full tensors. The old side computes before
  the transfer, the new side after; the old digest is broadcast and compared element-wise. The second sum
  catches rows that landed in the wrong place; integer arithmetic on raw bits makes the comparison exact.
  Any mismatch names the tensors and raises on every rank. Cost: about 0.3 s for Qwen3-0.6B.
* **scalar state** -- the checksums vouch for tensors, but `step`, the `param_groups` hyper-parameters and
  the LR-scheduler state travel in the manifest and are applied on the new side by code. So every old rank
  reads them from its live runtime before the transfer and every new rank from its live runtime after it
  (`_scalar_state`: one `path -> exact text` entry per value, 330 for Qwen3-0.6B). All ranks exchange
  these reports, and each one has to equal that of `old_ranks[0]` entry by entry; a difference names the
  entry, both values and both ranks, and raises on every rank. The optimizer check below shows what an
  unnoticed `step + 1` costs while every checksum matches: 6 % of the next update.

`resize/gpu_direct_verified` is 1.0 only if both comparisons passed on every rank; the trainer's
`[one-step-off][resize] ... runtime moved ...` line reports the same as `verified=1 verified_scalars=N`.
The launcher's verification version turns all of this on together with the logs and the trace (*Running
the single-node example*).

Offline:

* `tests/utils/test_fsdp_resize_on_cpu.py` -- planner against DTensor's own chunking, expand / shrink /
  non-prefix rank sets, uneven and empty shards, row conservation, invalid input.
* `tests/special_distributed/run_fsdp2_resize_transfer.py` (`torchrun --nproc-per-node=6`) -- 4 -> 6 -> 2 -> 6
  with AdamW state: bitwise equality against the old runtime, tied weight and non-persistent buffer
  restored, and training *continues* on the new mesh against a single-device golden model. The last check
  matters because `full_tensor()` reads `_local_tensor` through DTensor's own all-gather and would pass even
  if FSDP2's `_sharded_param_data` had not been written; training exercises FSDP2's all-gather path.
  At the end three manifests that lie about `step`, the learning rate or the scheduler position each have
  to make every rank raise and name the entry.
* `tests/special_distributed/run_fsdp2_resize_optimizer.py` (`torchrun --nproc-per-node=6`) -- the
  optimizer on the real model, fp32 AdamW with an LR schedule, after two real training steps. *Next
  update*: after each transfer (4 -> 6, 6 -> 2) the old runtime is still alive, and parameters, `exp_avg`,
  `exp_avg_sq`, `step`, `param_groups` and scheduler state are compared bitwise (1240 tensors for
  Qwen3-0.6B). Then both runtimes are given the same gradient -- a function of the global element index,
  hence bit-identical however a tensor is sharded -- and take one optimizer and scheduler step, and
  everything is compared bitwise again. *Faults*: an optimizer tensor that is not moved and a manifest
  that reports `step + 1` both have to be caught by name; the second one, repeated without verification,
  changes nothing but `step`, and the next update is then off by 6.1 %, the ratio of Adam's bias
  corrections for steps 5 and 4. *Trajectories*: eight language-model steps on fixed batches. The run that
  is transferred after two steps has to stay on the baseline that is not (relative loss difference below
  1e-5; the two meshes add the same numbers up in a different order), and the run that loses its optimizer
  state in the transfer has to leave it (1.1e-2 at the next step).
* `tests/experimental/one_step_off_policy/test_dynamic_resize_control_on_cpu.py` -- trainer-level flow,
  metrics and log lines with mocked worker groups.
* `tests/experimental/one_step_off_policy/test_dynamic_resize_launcher_on_cpu.py` -- runs the launcher
  with a stand-in `python3` that records every command: both versions start training and nothing else,
  and the verification version adds the trace and the logging level to the same command.

Why not just watch the loss: in RL the loss depends on sampled rollouts (which change with replica count,
batch composition and RNG stream), it is a single scalar, and optimizer state is invisible to it. It
catches only gross corruption.

## Configuration, environment, metrics

| Setting | Default | Effect |
|---|---|---|
| `trainer.dynamic_resize.handoff.backend` | `gpu_direct` (example yaml) | `gpu_direct`: this document; `disk_fallback`: checkpoint round trip (optimizer state not restored across world sizes) |
| `VERL_RESIZE_VERIFY` | unset | `1`: checksum and scalar-state comparison after every transfer; in the launcher, the verification version |
| `VERL_LOGGING_LEVEL` | `WARN` (`DEBUG` in the launcher's verification version) | `INFO` shows the resize lines of the trainer (`[one-step-off][resize] ... runtime moved ...`); `DEBUG` adds the transfer statistics of every rank and the workers' memory lines |

Metrics reported on a resize step (`step:N - ... - resize/...` line):

| Metric | Meaning |
|---|---|
| `resize/gpu_direct_status` | `moved` on success |
| `resize/gpu_direct_tensor_count`, `resize/gpu_direct_optimizer_tensor_count` | parameters / optimizer tensors moved (Qwen3-0.6B: 310 / 620) |
| `resize/gpu_direct_net_recv_bytes`, `resize/gpu_direct_self_copy_bytes` | bytes received over NCCL / copied on-device; network bytes equal the lower bound of the shard layout (4 -> 6: 3/4 of the state, 6 -> 2 or 2 -> 6: 5/6) |
| `resize/gpu_direct_transfer_seconds`, `resize/gpu_direct_peak_alloc_bytes` | transfer wall time; peak device memory while old and new shards coexist |
| `resize/gpu_direct_verified` | 1.0 when `VERL_RESIZE_VERIFY=1` and all checksums and scalar-state entries matched, otherwise 0.0 |
| `resize/gpu_direct_critic_*` | the same for the critic |
| `resize/active_actor_size`, `resize/active_rollout_size` | topology after the switch |

`CKPT_DIR` stays empty with `gpu_direct`; a `global_step_*` directory means the run used `disk_fallback`.

## Measured (8x RTX 5090, PCIe, single node)

* Qwen3-0.6B GRPO, 4 -> 6 -> 2 -> 6 with `VERL_RESIZE_VERIFY=1`: every transfer of the whole fp32 training
  state (parameters + two Adam moments, ~6 GB) takes 1.0-1.4 s and verifies; network bytes are exactly at
  the lower bound of the shard layout; training continues normally after each switch.
* The whole hard switch is about 45 s, of which about 34 s is vLLM engine start-up for the relaunched
  replicas; the training-state move is no longer on the critical path.

## Limitations and open items

* Validated on one node only. The transfer has no single-node assumption (`old ∪ new` is an arbitrary
  subgroup of WORLD), but multi-node transport and overlap with compute have not been exercised.
* Startup builds the actor on the whole pool and moves it to the initial actor ranks with one extra
  transfer (and one extra full-size model build).
* Peak device memory during a transfer is old + new shards; a model that fits in training may not fit
  twice.
* Every switch kills and relaunches all vLLM server actors (~34 s for Qwen3-0.6B).
* `prepare_resize_target` relies on FSDP2's `FSDPModule._apply` -> `reset_sharded_param()` making the
  `to_empty`'d storage the sharded parameter's home (the same mechanism as official meta-device init).
* `clear_dtensor_dispatch_caches` touches private DTensor caches whose names may change between torch
  versions (torch 2.9 verified).
* `disk_fallback` is kept as the alternative backend; it does not restore optimizer state across a world-size
  change.
