#!/usr/bin/env bash
# Real trainer/Ray/engine acceptance: two consecutive actor expansions with a
# weight sync between them.
#
#   stage0 actor/rollout = 2/6   (initial split, overrides the config's 4/4)
#   stage1 at step 2      = 4/4
#   stage2 at step 4      = 6/2
#
# Both steps are expansions, because the 4d branch rejects actor shrink. The
# run reaches step 6, so at least one weight sync happens after each resize.
# GPU execution requires approval of the enclosing command.
set -euo pipefail

RESIZE_TRACE=${RESIZE_TRACE:-/results/resize_trace.jsonl}
RESIZE_SUMMARY=${RESIZE_SUMMARY:-/results/two_resize_summary.json}

command=(
  python3 -m verl.experimental.one_step_off_policy.main_ppo
  --config-path=config --config-name=one_step_off_ppo_trainer_dynamic_resize.yaml
  hydra.run.dir=/results/hydra hydra.sweep.dir=/results/hydra_sweep
  data.train_files=/dataset/train.parquet data.val_files=/dataset/test.parquet
  data.train_batch_size=96 data.train_max_samples=384 data.val_max_samples=16
  data.max_prompt_length=512 data.max_response_length=64
  data.filter_overlong_prompts=true data.truncation=error
  actor_rollout_ref.model.path=/model actor_rollout_ref.model.use_shm=false
  actor_rollout_ref.model.use_remove_padding=false
  actor_rollout_ref.hybrid_engine=true actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.use_torch_compile=false
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=false
  actor_rollout_ref.actor.fsdp_config.model_dtype=fp32
  actor_rollout_ref.actor.fsdp_config.param_offload=false
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
  actor_rollout_ref.actor.fsdp_config.offload_policy=false
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.ppo_mini_batch_size=48
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
  actor_rollout_ref.actor.use_dynamic_bsz=false
  actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.entropy_coeff=0
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]'
  'actor_rollout_ref.actor.checkpoint.load_contents=[model,extra]'
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.n=2 actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false
  actor_rollout_ref.rollout.enforce_eager=true
  actor_rollout_ref.rollout.max_num_seqs=32
  actor_rollout_ref.rollout.max_num_batched_tokens=1024
  actor_rollout_ref.rollout.max_model_len=576
  actor_rollout_ref.rollout.gpu_memory_utilization=0.1
  actor_rollout_ref.rollout.agent.num_workers=1
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
  trainer.nnodes=1 trainer.n_gpus_per_node=2 rollout.nnodes=1 rollout.n_gpus_per_node=6
  trainer.total_training_steps=6 trainer.total_epochs=1
  trainer.save_freq=-1 trainer.test_freq=-1 trainer.val_before_train=false
  trainer.resume_mode=disable trainer.logger='[console]'
  trainer.default_local_dir=/results/checkpoints
  trainer.project_name=fsdp2-direct-acceptance trainer.experiment_name=actor2to4to6
  trainer.dynamic_resize.enable=true trainer.dynamic_resize.mode=schedule
  +trainer.dynamic_resize.hard_switch.enable=true
  trainer.dynamic_resize.shared_pool.n_gpus_per_node=8
  trainer.dynamic_resize.handoff.enable=true trainer.dynamic_resize.handoff.backend=gpu_direct
  trainer.dynamic_resize.handoff.stage_optimizer=false
  trainer.dynamic_resize.handoff.async_optimizer_preload=false
  trainer.dynamic_resize.handoff.optimizer_full_state_restore=false
  trainer.dynamic_resize.communicator_cache.enable=false
  trainer.dynamic_resize.trace.enable=true trainer.dynamic_resize.trace.flush=true
  trainer.dynamic_resize.trace.path="${RESIZE_TRACE}"
  trainer.dynamic_resize.trace.run_name=actor2to4to6-direct-acceptance
  trainer.dynamic_resize.schedule.stage0.actor_pool.size='[2,6]'
  trainer.dynamic_resize.schedule.stage0.rollout_pool.size='[2,6]'
  +trainer.dynamic_resize.schedule.stage1.step=2
  +trainer.dynamic_resize.schedule.stage1.actor_pool.mode=split
  +trainer.dynamic_resize.schedule.stage1.actor_pool.from_pool=shared_pool
  +trainer.dynamic_resize.schedule.stage1.actor_pool.index=0
  '+trainer.dynamic_resize.schedule.stage1.actor_pool.size=[4,4]'
  +trainer.dynamic_resize.schedule.stage1.rollout_pool.mode=split
  +trainer.dynamic_resize.schedule.stage1.rollout_pool.from_pool=shared_pool
  +trainer.dynamic_resize.schedule.stage1.rollout_pool.index=1
  '+trainer.dynamic_resize.schedule.stage1.rollout_pool.size=[4,4]'
  +trainer.dynamic_resize.schedule.stage1.release_old=false
  +trainer.dynamic_resize.schedule.stage2.step=4
  +trainer.dynamic_resize.schedule.stage2.actor_pool.mode=split
  +trainer.dynamic_resize.schedule.stage2.actor_pool.from_pool=shared_pool
  +trainer.dynamic_resize.schedule.stage2.actor_pool.index=0
  '+trainer.dynamic_resize.schedule.stage2.actor_pool.size=[6,2]'
  +trainer.dynamic_resize.schedule.stage2.rollout_pool.mode=split
  +trainer.dynamic_resize.schedule.stage2.rollout_pool.from_pool=shared_pool
  +trainer.dynamic_resize.schedule.stage2.rollout_pool.index=1
  '+trainer.dynamic_resize.schedule.stage2.rollout_pool.size=[6,2]'
  +trainer.dynamic_resize.schedule.stage2.release_old=false
  ray_kwargs.ray_init.num_cpus=48 +ray_kwargs.ray_init.num_gpus=8
  +ray_kwargs.ray_init.include_dashboard=false
  +ray_kwargs.ray_init.object_store_memory=4294967296
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DEBUG=INFO
  "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DEBUG_SUBSYS='INIT,ENV,GRAPH,P2P,SHM,NET'"
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_DEBUG_FILE=/results/nccl.%p.log
  +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_LEVEL=SYS
)

case "${1:-}" in
  --print-argv)
    python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${command[@]}"
    exit 0 ;;
  '') ;;
  *) printf '%s\n' 'Only --print-argv (CPU only) or approved GPU execution is supported.' >&2; exit 2 ;;
esac

[[ ! -e "${RESIZE_TRACE}" && ! -e "${RESIZE_SUMMARY}" ]] || {
  printf '%s\n' 'Fresh trace and summary paths required; never reuse a prior acceptance.' >&2; exit 1;
}
[[ ! -e /results/checkpoints ]] || {
  printf '%s\n' 'Fresh checkpoint directory required.' >&2; exit 1;
}
mkdir -p /results/checkpoints

exec "${command[@]}"
