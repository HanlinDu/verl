#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
cd "${REPO_ROOT}"

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_STRATEGY=${ACTOR_STRATEGY:-"fsdp2"}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-/file_system/common-models/Qwen/Qwen2.5-1.5B-Instruct}
RESOLVED_MODEL_PATH="${MODEL_PATH}"

TRAIN_FILE=${TRAIN_FILE:-/file_system/common-data/new_gsm8k/train.parquet}
VAL_FILE=${VAL_FILE:-/file_system/common-data/new_gsm8k/test.parquet}

n_gpus_rollout=${N_GPUS_ROLLOUT:-4}
n_gpus_training=$((NUM_GPUS - n_gpus_rollout))
resize_step=${RESIZE_STEP:-2}
second_resize_step=$((resize_step + 2))
shared_pool_gpus=${SHARED_POOL_GPUS:-${NUM_GPUS}}
first_split_plan=${FIRST_SPLIT_PLAN:-${SPLIT_PLAN:-"[6,2]"}}
second_split_plan=${SECOND_SPLIT_PLAN:-"[2,6]"}
total_training_steps=${TOTAL_TRAINING_STEPS:-$((second_resize_step + 2))}
CKPT_DIR=${CKPT_DIR:-/file_system/dhl/save_ckpt/dynamic-resize}
RAY_TMPDIR=${RAY_TMPDIR:-}

# profile
ENABLE_PROFILING=${ENABLE_PROFILING:-1}
PROFILE_TOOL=${PROFILE_TOOL:-torch}
PROFILE_DIR=${PROFILE_DIR:-${CKPT_DIR}/profiler}
PROFILE_ALL_RANKS=${PROFILE_ALL_RANKS:-false}
PROFILE_RANKS=${PROFILE_RANKS:-[0]}
PROFILE_MINI_STEP_START=${PROFILE_MINI_STEP_START:-0}
PROFILE_MINI_STEP_END=${PROFILE_MINI_STEP_END:-1}

if [[ -z "${PROFILE_STEPS:-}" ]]; then
  PROFILE_STEPS="[$(seq -s, 1 "${total_training_steps}")]"
fi

if [[ -n "${RAY_TMPDIR}" ]]; then
  mkdir -p "${RAY_TMPDIR}"
  export RAY_TMPDIR
fi

mkdir -p "${CKPT_DIR}"
mkdir -p "${PROFILE_DIR}"

args=(
  --config-path=config
  --config-name=one_step_off_ppo_trainer_dynamic_resize.yaml
  data.train_files="${TRAIN_FILE}"
  data.val_files="${VAL_FILE}"
  actor_rollout_ref.model.path="${RESOLVED_MODEL_PATH}"
  trainer.nnodes=1
  trainer.n_gpus_per_node=${n_gpus_training}
  rollout.nnodes=1
  rollout.n_gpus_per_node=${n_gpus_rollout}
  trainer.default_local_dir="${CKPT_DIR}"
  trainer.total_training_steps=${total_training_steps}
  trainer.save_freq=-1
  trainer.dynamic_resize.shared_pool.n_gpus_per_node=${shared_pool_gpus}
  +trainer.dynamic_resize.schedule.stage1.step=${resize_step}
  +trainer.dynamic_resize.schedule.stage1.actor_pool.mode=split
  +trainer.dynamic_resize.schedule.stage1.actor_pool.from_pool=shared_pool
  +trainer.dynamic_resize.schedule.stage1.actor_pool.index=0
  +trainer.dynamic_resize.schedule.stage1.actor_pool.size=${first_split_plan}
  +trainer.dynamic_resize.schedule.stage1.rollout_pool.mode=split
  +trainer.dynamic_resize.schedule.stage1.rollout_pool.from_pool=shared_pool
  +trainer.dynamic_resize.schedule.stage1.rollout_pool.index=1
  +trainer.dynamic_resize.schedule.stage1.rollout_pool.size=${first_split_plan}
  +trainer.dynamic_resize.schedule.stage1.release_old=false
  +trainer.dynamic_resize.schedule.stage2.step=${second_resize_step}
  +trainer.dynamic_resize.schedule.stage2.actor_pool.mode=split
  +trainer.dynamic_resize.schedule.stage2.actor_pool.from_pool=shared_pool
  +trainer.dynamic_resize.schedule.stage2.actor_pool.index=0
  +trainer.dynamic_resize.schedule.stage2.actor_pool.size=${second_split_plan}
  +trainer.dynamic_resize.schedule.stage2.rollout_pool.mode=split
  +trainer.dynamic_resize.schedule.stage2.rollout_pool.from_pool=shared_pool
  +trainer.dynamic_resize.schedule.stage2.rollout_pool.index=1
  +trainer.dynamic_resize.schedule.stage2.rollout_pool.size=${second_split_plan}
  +trainer.dynamic_resize.schedule.stage2.release_old=false
  trainer.test_freq=-1
  trainer.resume_mode=disable
  trainer.logger='["console"]'
  actor_rollout_ref.hybrid_engine=false
  actor_rollout_ref.actor.use_torch_compile=false
  actor_rollout_ref.actor.strategy=${ACTOR_STRATEGY}
  critic.strategy=${ACTOR_STRATEGY}
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
  critic.ppo_micro_batch_size_per_gpu=4
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.enforce_eager=true
  actor_rollout_ref.rollout.max_num_seqs=64
  actor_rollout_ref.rollout.max_num_batched_tokens=2048
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3
  critic.model.path="${RESOLVED_MODEL_PATH}"
  critic.model.tokenizer_path="${RESOLVED_MODEL_PATH}"
)

if [[ "${ENABLE_PROFILING}" == "1" || "${ENABLE_PROFILING}" == "true" || "${ENABLE_PROFILING}" == "True" ]]; then
  args+=(
    global_profiler.tool=${PROFILE_TOOL}
    global_profiler.steps=${PROFILE_STEPS}
    global_profiler.save_path="${PROFILE_DIR}"
    actor_rollout_ref.actor.profiler.enable=true
    actor_rollout_ref.actor.profiler.tool=${PROFILE_TOOL}
    actor_rollout_ref.actor.profiler.save_path="${PROFILE_DIR}/actor"
    actor_rollout_ref.actor.profiler.all_ranks=${PROFILE_ALL_RANKS}
    actor_rollout_ref.actor.profiler.ranks=${PROFILE_RANKS}
    actor_rollout_ref.rollout.profiler.enable=true
    actor_rollout_ref.rollout.profiler.tool=${PROFILE_TOOL}
    actor_rollout_ref.rollout.profiler.save_path="${PROFILE_DIR}/rollout"
    actor_rollout_ref.rollout.profiler.all_ranks=${PROFILE_ALL_RANKS}
    actor_rollout_ref.rollout.profiler.ranks=${PROFILE_RANKS}
    critic.profiler.enable=true
    critic.profiler.tool=${PROFILE_TOOL}
    critic.profiler.save_path="${PROFILE_DIR}/critic"
    critic.profiler.all_ranks=${PROFILE_ALL_RANKS}
    critic.profiler.ranks=${PROFILE_RANKS}
  )

  if [[ "${PROFILE_TOOL}" == "torch" ]]; then
    args+=(
      actor_rollout_ref.actor.profiler.tool_config.torch.step_start=${PROFILE_MINI_STEP_START}
      actor_rollout_ref.actor.profiler.tool_config.torch.step_end=${PROFILE_MINI_STEP_END}
      +actor_rollout_ref.actor.profiler.tool_config.torch.manual_save=true
      actor_rollout_ref.rollout.profiler.tool_config.torch.step_start=${PROFILE_MINI_STEP_START}
      actor_rollout_ref.rollout.profiler.tool_config.torch.step_end=${PROFILE_MINI_STEP_END}
      +actor_rollout_ref.rollout.profiler.tool_config.torch.manual_save=true
      critic.profiler.tool_config.torch.step_start=${PROFILE_MINI_STEP_START}
      critic.profiler.tool_config.torch.step_end=${PROFILE_MINI_STEP_END}
      +critic.profiler.tool_config.torch.manual_save=true
    )
  fi
fi

python3 -m verl.experimental.one_step_off_policy.main_ppo "${args[@]}"
