#!/usr/bin/env bash
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_STRATEGY=${ACTOR_STRATEGY:-"fsdp2"}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}

TRAIN_FILE=${TRAIN_FILE:-${HOME}/data/gsm8k/train.parquet}
VAL_FILE=${VAL_FILE:-${HOME}/data/gsm8k/test.parquet}

n_gpus_rollout=${N_GPUS_ROLLOUT:-4}
n_gpus_training=$((NUM_GPUS - n_gpus_rollout))
resize_step=${RESIZE_STEP:-1}
shared_pool_gpus=${SHARED_POOL_GPUS:-${NUM_GPUS}}
split_plan=${SPLIT_PLAN:-"[6,2]"}
actor_split_index=${ACTOR_SPLIT_INDEX:-0}
rollout_split_index=${ROLLOUT_SPLIT_INDEX:-1}
split_from_pool=${SPLIT_FROM_POOL:-shared_pool}

python3 -m verl.experimental.one_step_off_policy.main_ppo \
  --config-path=config \
  --config-name=one_step_off_ppo_trainer_dynamic_resize.yaml \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=${n_gpus_training} \
  rollout.nnodes=1 \
  rollout.n_gpus_per_node=${n_gpus_rollout} \
  trainer.total_training_steps=5 \
  trainer.save_freq=1 \
  trainer.dynamic_resize.schedule[0].step=${resize_step} \
  trainer.dynamic_resize.shared_pool.n_gpus_per_node=${shared_pool_gpus} \
  trainer.dynamic_resize.schedule[0].actor_pool.mode=split \
  trainer.dynamic_resize.schedule[0].actor_pool.size=${split_plan} \
  trainer.dynamic_resize.schedule[0].actor_pool.index=${actor_split_index} \
  trainer.dynamic_resize.schedule[0].actor_pool.from_pool=${split_from_pool} \
  trainer.dynamic_resize.schedule[0].rollout_pool.mode=split \
  trainer.dynamic_resize.schedule[0].rollout_pool.size=${split_plan} \
  trainer.dynamic_resize.schedule[0].rollout_pool.index=${rollout_split_index} \
  trainer.dynamic_resize.schedule[0].rollout_pool.from_pool=${split_from_pool} \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  actor_rollout_ref.actor.strategy=${ACTOR_STRATEGY} \
  critic.strategy=${ACTOR_STRATEGY}
