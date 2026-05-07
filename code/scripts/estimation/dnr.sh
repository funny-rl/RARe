#!/bin/bash

cd ../../

export HYDRA_FULL_ERROR=1

ALGO=$1
USE_WANDB=$2
GN=$3

ALGO=${ALGO:-"null"}
USE_WANDB=${USE_WANDB:-"false"}

EXTRA_ARGS=()

BASE_AGENT="MAXMINQ"

ENVS="grid" 
ENV_NAME="DiscreteNoisyRewards"

max_skip=5
reward_mean=0.0

if [ "$ALGO" == "UTE" ]; then
    ensemble_size=5
fi

use_lr_decay=true
warmup_steps=10000
total_training_steps=30000
eval_render_interval=1000000
buffer_size=30000
skip_buffer_size=30000 

e_greedy_type=linear
e_decay=20000

traj_log_interval=400
eval_interval=400
num_eval_episodes=1

lr=0.001
tau=0.005

hidden_dim=32
batch_size=64
use_data_aug=false

num_decision=8 # 4
n_target=1 # 1

cutoff=1.0
max_alpha=0.2
min_alpha=0.1
use_es_target=false
expected_ensemble_size=1 
expected_ensemble_reduction=min

EXTRA_ARGS+=("base_agent.n_target=$n_target")
if [ "$ALGO" != "null" ]; then
    EXTRA_ARGS+=("base_agent/algo=$ALGO")
    EXTRA_ARGS+=("base_agent.algo.max_skip=$max_skip")
    EXTRA_ARGS+=("base_agent.algo.skip_buffer_size=$skip_buffer_size")
    EXTRA_ARGS+=("base_agent.algo.use_data_aug=$use_data_aug")
    if [ "$ALGO" == "RARe" ]; then
        EXTRA_ARGS+=("base_agent.algo.cutoff=$cutoff")
        EXTRA_ARGS+=("base_agent.algo.max_alpha=$max_alpha")
        EXTRA_ARGS+=("base_agent.algo.min_alpha=$min_alpha")
        EXTRA_ARGS+=("base_agent.algo.use_es_target=$use_es_target")
        EXTRA_ARGS+=("base_agent.algo.expected_ensemble_size=$expected_ensemble_size")
        EXTRA_ARGS+=("base_agent.algo.expected_ensemble_reduction=$expected_ensemble_reduction")
    fi
    if [ "$ALGO" == "UTE" ]; then
        EXTRA_ARGS+=("base_agent.algo.ensemble_size=$ensemble_size")
    fi
fi

if [ "$USE_WANDB" != "false" ]; then
    GN=${GN:-"test"}
    EXTRA_ARGS+=("use_wandb=true")
    EXTRA_ARGS+=("group_name=$GN")
fi

for SEED in {15..19};
do
    ARGS=(
        "base_agent=$BASE_AGENT"
        "envs=$ENVS"
        "seed=$SEED"
        "envs.name=$ENV_NAME"
        "envs.skip_mdp.num_decision=$num_decision"
        "envs.skip_mdp.reward_mean=$reward_mean"
        "eval_interval=$eval_interval"
        "traj_log_interval=$traj_log_interval"
        "num_eval_episodes=$num_eval_episodes"
        "warmup_steps=$warmup_steps"
        "total_training_steps=$total_training_steps"
        "eval_render_interval=$eval_render_interval"
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