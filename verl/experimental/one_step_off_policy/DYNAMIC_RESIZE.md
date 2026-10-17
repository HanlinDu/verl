# Dynamic resize in the hybrid one-step-off trainer

This document describes how `verl.experimental.one_step_off_policy` moves GPUs between rollout and
training while a run is in progress ("dynamic resize"): what a resize is, how to run the single-node
example, what happens during a hard switch, and how the result is verified. It is the reader's guide to
the code; the commit messages carry the per-change rationale.

Related files:

* `README.md` -- the one-step-off recipe itself (pipeline, parameter sync, configuration).
* `DYNAMIC_RESIZE_DIRECT_RESHARD_PLAN.md` -- the design plan for direct (checkpoint-free) resharding.
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
| `USE_KL_LOSS` | false | GRPO without a reference model by default |
| `HARD_SWITCH_ENABLE` | true | actually move resources |
| `HOST_STAGING_ENABLE` / `HOST_STAGING_BACKEND` | true / `gpu_direct` | `handoff.enable` / `handoff.backend` |
| `HYSTERESIS_*`, `BUDGET_PROTECTION_ENABLE`, `COMM_CACHE_ENABLE` | off | optional controller, memory gate, communicator cache |
| `PROFILER_ENABLE` | true | `global_profiler.enabled`; set `false` for quick iterations |

The yaml's `stage0`, the initial split, is `[4,4]`. A pool or an initial split other than 4 + 4 therefore
also needs `trainer.dynamic_resize.schedule.stage0.actor_pool.size=[actor,rollout]` and
`...stage0.rollout_pool.size=[actor,rollout]` on the command line, next to `NUM_GPUS` and `N_GPUS_ROLLOUT`.

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

Run everything from the repository root of a clean checkout, with `export PYTHONPATH=$PWD`.

| Check | Needs | Reference environment |
|---|---|---|
| CPU tests | Python 3.12 and the packages listed under *CPU tests*; no accelerator | Python 3.12.3, torch 2.8.0+cpu, ray 2.49.2, transformers 4.57.1, tensordict 0.10.0, numpy 2.5.4, pytest 9.2.0 |
| Transfer test | one node with 6 CUDA GPUs; a CUDA build of torch with FSDP2 and NCCL | 6 of the 8 GPUs below, same software |
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

`tests/experimental/one_step_off_policy/test_dynamic_resize_launcher_on_cpu.py` runs the launcher with a
stand-in `python3` that records every command: it has to start training, with the arguments of the caller
last.

### Transfer test (6 GPUs)

```bash
torchrun --standalone --nproc-per-node=6 tests/special_distributed/run_fsdp2_resize_transfer.py
```

Rank 0 prints `FSDP2 GPU-direct resize transfer 4->6 passed`.

### End to end (8 GPUs)

```bash
export MODEL_PATH=<model directory> TRAIN_FILE=<gsm8k train.parquet> VAL_FILE=<gsm8k test.parquet>
export PROFILER_ENABLE=false TRAIN_BATCH_SIZE=96 PPO_MINI_BATCH_SIZE=48
export TORCH_CUDA_ARCH_LIST=12.0   # RTX 5090 (sm_120) only, as in the reference environment
L=verl/experimental/one_step_off_policy/shell/run_dynamic_resize.sh
G="algorithm.adv_estimator=grpo actor_rollout_ref.rollout.n=5 actor_rollout_ref.actor.entropy_coeff=0"
```

`$G` selects GRPO, which trains no critic; without it the launcher runs GAE, which trains one. The two
batch sizes keep a run short. `$RUN` is a new, empty directory for each run.

```bash
# one switch: [4,4] (actor, rollout) -> [6,2] at step 2, three steps
CKPT_DIR=$RUN/ckpt bash $L $G trainer.total_training_steps=3
# the same switch through the checkpoint round trip
HOST_STAGING_BACKEND=disk_fallback CKPT_DIR=$RUN/ckpt bash $L $G trainer.total_training_steps=3
# dynamic resize off: separate actor and rollout workers on 4 + 4 GPUs
CKPT_DIR=$RUN/ckpt bash $L $G trainer.total_training_steps=3 \
  actor_rollout_ref.hybrid_engine=false trainer.dynamic_resize.enable=false
```

A run passes when the launcher exits with code 0 after its last step line (`step:3`) and the switch line
(`step:2`) reports `resize/hard_switch_success:1.0`; with resize off every step line reports
`resize/enabled:0.0` instead.

## The hard switch, step by step

`OneStepOffRayTrainer._apply_dynamic_resize_hard_switch` currently does, for a switch to
`(actor_target, rollout_target)`:

1. Save a full checkpoint of actor (and critic) and, if `handoff.stage_optimizer`, page each rank's
   optimizer state to the host staging directory.
2. Quiesce rollout: abort in-flight requests on every replica, kill the agent-loop workers.
3. `handoff.backend=gpu_direct` and an *expansion*: `FSDPEngine.stage_model_for_resize` sends each old
   rank's parameter shards over a temporary NCCL group into a staging buffer on the new ranks
   (`verl/utils/fsdp_resize.py`). Shrinks, and any staging failure, fall back to the checkpoint.
4. Rebuild the training runtime: destroy the training subgroup, create the new one, rebuild the FSDP2
   engine on the new ranks from the pretrained weights, then either copy the staged shards in or
   `load_checkpoint`. Optimizer files are per world size, so across a world-size change the optimizer
   restarts from a fresh state (the plan document calls this "effectively weights_only").
5. Kill and relaunch every vLLM server actor on the new rollout ranks, rebuild the load balancer and the
   agent-loop manager.
6. `reconfigure_checkpoint_engine` with a fresh group name, then `update_weights` pushes the actor
   weights to the new replicas and resumes generation.
