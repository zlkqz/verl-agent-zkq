#!/usr/bin/env bash

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd -- "$repo_root" || exit 1

# Reduce glibc arena proliferation and return releasable heap memory sooner.
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export MALLOC_TRIM_THRESHOLD_=${MALLOC_TRIM_THRESHOLD_:-131072}

if [[ ${DRY_RUN:-0} != 1 ]]; then
    set -x
fi

# 使用镜像：`verlai/verl:vllm011.latest`

# export http_proxy=http://124.16.138.148:7890
# export https_proxy=http://124.16.138.148:7890
# export all_proxy=socks5://124.16.138.148:7890
# export HTTP_PROXY=http://124.16.138.148:7890
# export HTTPS_PROXY=http://124.16.138.148:7890
# export ALL_PROXY=socks5://124.16.138.148:7890
# export NO_PROXY_LIST=localhost,127.0.0.1,127.0.1.1,::1,0.0.0.0,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,169.254.0.0/16,.local,.lan,.internal,.cluster.local,kubernetes.default.svc
# export no_proxy="$NO_PROXY_LIST"
# export NO_PROXY="$NO_PROXY_LIST"

export http_proxy=http://100.98.216.198:9000
export https_proxy=http://100.98.216.198:9000
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy

# wandb login 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
if [[ ${DRY_RUN:-0} != 1 ]]; then
    /usr/local/bin/wandb login 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
fi

ENGINE=${1:-vllm}
# vLLM 0.11's XFORMERS paged-KV path can fail with its 16-token cache blocks.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_DEDUP_LOGS=0
export ALFWORLD_DATA="$repo_root/data/alfworld"

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=16
val_data_size=128
group_size=8
train_temperature=0.8
train_top_p=0.95
max_prompt_length=3072
max_response_length=3072
lr=1e-6
val_before_train=False
project_name=p2o_agent_exp
micro_bs_per_gpu=8
# Dedicated ablation scripts override only these experiment settings.
mini_batch_size=${mini_batch_size:-256}
entropy_coeff=${entropy_coeff:-0.0}
use_kl_loss=${use_kl_loss:-False}
kl_loss_coef=${kl_loss_coef:-0.001}
grpo_reward_grouping=${grpo_reward_grouping:-step}
grpo_invalid_penalty_mode=${grpo_invalid_penalty_mode:-joint}
# model_path=/ceph_home/arknet/hf_models/Qwen/Qwen3-8B
model_path=/mnt/shared-storage-user/gpfs1-agent-security/luxinyu1/cot-p2o-space/Qwen3-8B

EXP_NAME="p2o_alfworld_qwen3_8b_think_bs_${train_data_size}x${group_size}_mini_batch_${mini_batch_size}_temperature${train_temperature}_length${max_prompt_length}+${max_response_length}_lr${lr}${EXP_NAME_SUFFIX:-}"
# save_path=/ceph_home/zhangkaiqi2024/saved_models/p2o-agent/${EXP_NAME}
save_path=/mnt/shared-storage-user/gpfs1-agent-security/luxinyu1/cot-p2o-space/saved_models/p2o-agent/${EXP_NAME}


echo $EXP_NAME
echo "save_path=$save_path"

# We only use data preparation to indicate the modality and the data size.
# python3 -m examples.data_preprocess.prepare \
#     --mode 'text' \
#     --local_dir "$repo_root/data/alfworld" \
#     --train_data_size $train_data_size \
#     --val_data_size $val_data_size

cmd=(python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.grpo.reward_grouping=$grpo_reward_grouping \
    algorithm.grpo.invalid_penalty_mode=$grpo_invalid_penalty_mode \
    data.train_files="$repo_root/data/alfworld/text/train.parquet" \
    data.val_files="$repo_root/data/alfworld/text/test.parquet" \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.checkpoint.contents='[model,optimizer,extra,hf_model]' \
    critic.checkpoint.contents='[model,optimizer,extra,hf_model]' \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_bs_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_bs_per_gpu \
    actor_rollout_ref.rollout.max_num_batched_tokens=300000 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.temperature=$train_temperature \
    actor_rollout_ref.rollout.top_p=$train_top_p \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_bs_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=30 \
    +env.progress.enable=True \
    +env.progress.sample_count=2 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$EXP_NAME \
    trainer.default_local_dir=$save_path \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=400 \
    trainer.resume_mode=auto \
    trainer.keep_latest_full_checkpoint=True \
    trainer.val_before_train=$val_before_train "${@:2}"
)

if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi
mkdir -p logs/
"${cmd[@]}" > "logs/${EXP_NAME}.log" 2>&1
