#!/bin/bash

cd ../../

export HYDRA_FULL_ERROR=1

BASE_AGENT=$1
ALGO=$2
USE_WANDB=$3
GN=$4

if [ "$BASE_AGENT" != "TD3" ] && [ "$BASE_AGENT" != "DDPG" ]; then
    echo "Invalid BASE_AGENT. Please choose TD3 or DDPG."
    exit 1
fi

ALGO=${ALGO:-"null"}
USE_WANDB=${USE_WANDB:-"false"}

EXTRA_ARGS=()

ENVS="cnr" # continuous noisy rewards
ENV_NAME="ContinuosNoisyRewards"

max_skip=3
num_decision=64
ensemble_size=1

if [ "$ALGO" == "UTE" ]; then
    ensemble_size=10
fi

use_lr_decay=true
use_eval_render=false

warmup_steps=10000
total_training_steps=40000
buffer_size=40000
skip_buffer_size=40000 

e_greedy_type=linear
e_decay=30000

traj_log_interval=300
eval_interval=300
num_eval_episodes=1

lr=0.001
tau=0.005

hidden_dim=32
batch_size=64
use_data_aug=false

cutoff=1.0
n_sample=30
max_alpha=2.0
min_alpha=1.0
use_es_target=false # whether to use expected sarsa target for skip q value update, only for RARe

if [ "$ALGO" != "null" ]; then
    EXTRA_ARGS+=("base_agent/algo=$ALGO")
    EXTRA_ARGS+=("base_agent.algo.ensemble_size=$ensemble_size")
    EXTRA_ARGS+=("base_agent.algo.max_skip=$max_skip")
    EXTRA_ARGS+=("base_agent.algo.skip_buffer_size=$skip_buffer_size")
    EXTRA_ARGS+=("base_agent.algo.use_data_aug=$use_data_aug")
    if [ "$ALGO" == "RARe" ]; then
        EXTRA_ARGS+=("base_agent.algo.cutoff=$cutoff")
        EXTRA_ARGS+=("base_agent.algo.max_alpha=$max_alpha")
        EXTRA_ARGS+=("base_agent.algo.min_alpha=$min_alpha")
        EXTRA_ARGS+=("base_agent.algo.use_es_target=$use_es_target")
    fi
fi

if [ "$USE_WANDB" != "false" ]; then
    GN=${GN:-"test"}
    EXTRA_ARGS+=("use_wandb=true")
    EXTRA_ARGS+=("group_name=$GN")
fi

for SEED in {16..19};
do
    ARGS=(
        "base_agent=$BASE_AGENT"
        "envs=$ENVS"
        "seed=$SEED"
        "envs.name=$ENV_NAME"
        "envs.num_decision=$num_decision"
        "eval_interval=$eval_interval"
        "traj_log_interval=$traj_log_interval"
        "num_eval_episodes=$num_eval_episodes"
        "warmup_steps=$warmup_steps"
        "total_training_steps=$total_training_steps"
        "use_eval_render=$use_eval_render"
        "common.e_greedy_type=$e_greedy_type"
        "common.use_lr_decay=$use_lr_decay"
        "common.e_decay=$e_decay"
        "common.buffer_size=$buffer_size" 
        "common.hidden_dim=$hidden_dim"
        "common.batch_size=$batch_size"
        "common.lr=$lr"
        "common.tau=$tau"
        ${EXTRA_ARGS[@]}
    )

    python main.py "${ARGS[@]}" 
done