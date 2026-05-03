#!/bin/bash

cd ../../

export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1

ALGO=$1
USE_WANDB=$2
GN=$3


ALGO=${ALGO:-"null"}
USE_WANDB=${USE_WANDB:-"false"}

EXTRA_ARGS=()

BASE_AGENT=DDPG
ENVS="safe"
ENV_NAME="SafetyRacecarGoal1-v0"

max_skip=3

if [ "$ALGO" == "UTE" ]; then
    ensemble_size=5
fi

use_lr_decay=true
warmup_steps=5000
total_training_steps=35000
eval_render_interval=1500
buffer_size=35000
skip_buffer_size=100000 
e_greedy_type=linear
e_decay=30000

traj_log_interval=300
eval_interval=300
num_eval_episodes=3

lr=0.001
tau=0.005

hidden_dim=128
batch_size=64
use_data_aug=true

# RARe
cutoff=1.0
n_sample=20
max_alpha=0.05
min_alpha=0.0
use_es_target=true # whether to use expected sarsa target for skip q value update, only for RARe
expected_ensemble_size=1
expected_ensemble_reduction=min
# UTE
use_adaptive_lambda=true # whether to use adaptive lambda for UTE, which adjusts the lambda based on the uncertainty of the value estimation

if [ "$ALGO" != "null" ]; then
    EXTRA_ARGS+=("base_agent/algo=$ALGO")
    if [ "$ALGO" != "TAAC" ]; then
        EXTRA_ARGS+=("base_agent.algo.use_data_aug=$use_data_aug")
        EXTRA_ARGS+=("base_agent.algo.max_skip=$max_skip")
        EXTRA_ARGS+=("base_agent.algo.skip_buffer_size=$skip_buffer_size")
    fi
    if [ "$ALGO" == "RARe" ]; then
        EXTRA_ARGS+=("base_agent.algo.cutoff=$cutoff")
        EXTRA_ARGS+=("base_agent.algo.n_sample=$n_sample")
        EXTRA_ARGS+=("base_agent.algo.max_alpha=$max_alpha")
        EXTRA_ARGS+=("base_agent.algo.min_alpha=$min_alpha")
        EXTRA_ARGS+=("base_agent.algo.use_es_target=$use_es_target")
        EXTRA_ARGS+=("base_agent.algo.expected_ensemble_size=$expected_ensemble_size")
        EXTRA_ARGS+=("base_agent.algo.expected_ensemble_reduction=$expected_ensemble_reduction")
    fi
    if [ "$ALGO" == "UTE" ]; then
        EXTRA_ARGS+=("base_agent.algo.ensemble_size=$ensemble_size")
        EXTRA_ARGS+=("base_agent.algo.use_adaptive_lambda=$use_adaptive_lambda")
    fi

fi

if [ "$USE_WANDB" != "false" ]; then
    GN=${GN:-"test"}
    EXTRA_ARGS+=("use_wandb=true")
    EXTRA_ARGS+=("group_name=$GN")
fi

for SEED in {0..19};
do
    ARGS=(
        "base_agent=$BASE_AGENT"
        "envs=$ENVS"
        "seed=$SEED"
        "envs.name=$ENV_NAME"
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