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
shared_pool_gpus=${SHARED_POOL_GPUS:-${NUM_GPUS}}
split_plan=${SPLIT_PLAN:-"[6,2]"}

python3 -m verl.experimental.one_step_off_policy.main_ppo \
  --config-path=config \
  --config-name=one_step_off_ppo_trainer_dynamic_resize.yaml \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  actor_rollout_ref.model.path="${RESOLVED_MODEL_PATH}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=${n_gpus_training} \
  rollout.nnodes=1 \
  rollout.n_gpus_per_node=${n_gpus_rollout} \
  trainer.total_training_steps=5 \
  trainer.save_freq=-1 \
  trainer.dynamic_resize.shared_pool.n_gpus_per_node=${shared_pool_gpus} \
  +trainer.dynamic_resize.schedule.stage1.step=${resize_step} \
  +trainer.dynamic_resize.schedule.stage1.actor_pool.mode=split \
  +trainer.dynamic_resize.schedule.stage1.actor_pool.from_pool=shared_pool \
  +trainer.dynamic_resize.schedule.stage1.actor_pool.index=0 \
  +trainer.dynamic_resize.schedule.stage1.actor_pool.size=${split_plan} \
  +trainer.dynamic_resize.schedule.stage1.rollout_pool.mode=split \
  +trainer.dynamic_resize.schedule.stage1.rollout_pool.from_pool=shared_pool \
  +trainer.dynamic_resize.schedule.stage1.rollout_pool.index=1 \
  +trainer.dynamic_resize.schedule.stage1.rollout_pool.size=${split_plan} \
  +trainer.dynamic_resize.schedule.stage1.release_old=false \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.total_training_steps=4 \
  trainer.logger='["console"]' \
  actor_rollout_ref.hybrid_engine=false \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.strategy=${ACTOR_STRATEGY} \
  critic.strategy=${ACTOR_STRATEGY} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  critic.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  critic.model.path="${RESOLVED_MODEL_PATH}" \
  critic.model.tokenizer_path="${RESOLVED_MODEL_PATH}"
