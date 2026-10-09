set -x

# 使用镜像：`verlai/verl:vllm011.latest`

export http_proxy=http://124.16.138.148:7890
export https_proxy=http://124.16.138.148:7890
export all_proxy=socks5://124.16.138.148:7890
export HTTP_PROXY=http://124.16.138.148:7890
export HTTPS_PROXY=http://124.16.138.148:7890
export ALL_PROXY=socks5://124.16.138.148:7890
export NO_PROXY_LIST=localhost,127.0.0.1,127.0.1.1,::1,0.0.0.0,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,169.254.0.0/16,.local,.lan,.internal,.cluster.local,kubernetes.default.svc
export no_proxy="$NO_PROXY_LIST"
export NO_PROXY="$NO_PROXY_LIST"

wandb login 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

ENGINE=${1:-vllm}
# vLLM 0.11's XFORMERS paged-KV path can fail with its 16-token cache blocks.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_DEDUP_LOGS=0
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export ALFWORLD_DATA="$repo_root/data/alfworld"

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=16
val_data_size=128
group_size=8
train_temperature=0.8
train_top_p=0.95
max_prompt_length=2048
max_response_length=2048
lr=1e-6
val_before_train=False
project_name=p2o_agent_exp
micro_bs_per_gpu=8

EXP_NAME="p2o_alfworld_qwen3_8b_think_bs_${train_data_size}x${group_size}_temperature${train_temperature}_length${max_prompt_length}+${max_response_length}_lr${lr}"
save_path=/ceph_home/zhangkaiqi2024/saved_models/p2o-agent/${EXP_NAME}

mkdir -p logs/

echo $EXP_NAME
echo "save_path=$save_path"

# We only use data preparation to indicate the modality and the data size.
# python3 -m examples.data_preprocess.prepare \
#     --mode 'text' \
#     --local_dir "$repo_root/data/alfworld" \
#     --train_data_size $train_data_size \
#     --val_data_size $val_data_size

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$repo_root/data/alfworld/text/train.parquet" \
    data.val_files="$repo_root/data/alfworld/text/test.parquet" \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/ceph_home/arknet/hf_models/Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_bs_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_bs_per_gpu \
    actor_rollout_ref.rollout.max_num_batched_tokens=400000 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.temperature=$train_temperature \
    actor_rollout_ref.rollout.top_p=$train_top_p \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
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
    env.max_steps=50 \
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
    trainer.total_epochs=200 \
    trainer.val_before_train=$val_before_train "${@:3}" > "logs/${EXP_NAME}.log" 2>&1
