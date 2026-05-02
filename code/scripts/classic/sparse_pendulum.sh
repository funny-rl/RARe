#!/bin/bash

cd ../../

export HYDRA_FULL_ERROR=1


ALGO=$1
USE_WANDB=$2
GN=$3

ALGO=${ALGO:-"null"}
USE_WANDB=${USE_WANDB:-"false"}

EXTRA_ARGS=()

BASE_AGENT=DDPG
ENVS="classic"
ENV_NAME="Sparse_Pendulum-v1"
custom_reward=true

max_skip=10
ensemble_size=1 

if [ "$ALGO" == "UTE" ]; then
    ensemble_size=10
fi

use_lr_decay=true
warmup_steps=2000
total_training_steps=22000
eval_render_interval=22000
buffer_size=22000
skip_buffer_size=44000 
e_greedy_type=linear
e_decay=20000

traj_log_interval=200
eval_interval=200
num_eval_episodes=10

lr=0.001
tau=0.005

hidden_dim=64
batch_size=64
use_data_aug=true

# RARe
n_sample=20
max_alpha=0.1
min_alpha=0.0
use_es_target=false # whether to use expected sarsa target for skip q value update, only for RARe

# UTE
use_adaptive_lambda=true # whether to use adaptive lambda for UTE, which adjusts the lambda based on the uncertainty of the value estimation

if [ "$ALGO" != "null" ]; then
    EXTRA_ARGS+=("base_agent/algo=$ALGO")
    if [ "$ALGO" != "TAAC" ]; then
        EXTRA_ARGS+=("base_agent.algo.use_data_aug=$use_data_aug")
        EXTRA_ARGS+=("base_agent.algo.max_skip=$max_skip")
        EXTRA_ARGS+=("base_agent.algo.ensemble_size=$ensemble_size")
        EXTRA_ARGS+=("base_agent.algo.skip_buffer_size=$skip_buffer_size")
    fi
    if [ "$ALGO" == "RARe" ]; then
        EXTRA_ARGS+=("base_agent.algo.n_sample=$n_sample")
        EXTRA_ARGS+=("base_agent.algo.max_alpha=$max_alpha")
        EXTRA_ARGS+=("base_agent.algo.min_alpha=$min_alpha")
        EXTRA_ARGS+=("base_agent.algo.use_es_target=$use_es_target")
    fi
    if [ "$ALGO" == "UTE" ]; then
        EXTRA_ARGS+=("base_agent.algo.use_adaptive_lambda=$use_adaptive_lambda")
    fi

fi

if [ "$USE_WANDB" != "false" ]; then
    GN=${GN:-"test"}
    EXTRA_ARGS+=("use_wandb=true")
    EXTRA_ARGS+=("group_name=$GN")
fi

for SEED in {0..9}; 
do
    ARGS=(
        "base_agent=$BASE_AGENT"
        "envs=$ENVS"
        "seed=$SEED"
        "envs.name=$ENV_NAME"
        "envs.pendulum.custom_reward=$custom_reward"
        "eval_interval=$eval_interval"
        "traj_log_interval=$traj_log_interval"
        "num_eval_episodes=$num_eval_episodes"
        "warmup_steps=$warmup_steps"
        "total_training_steps=$total_training_steps"
        "common.e_greedy_type=$e_greedy_type"
        "common.use_lr_decay=$use_lr_decay"
        "eval_render_interval=$eval_render_interval"
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