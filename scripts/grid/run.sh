#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
    echo "Usage: run.sh <environment> <algorithm> <fixed|random> [training options...]" >&2
    exit 2
fi

grid_env="$1"
grid_algorithm="$2"
grid_mode="$3"
shift 3

case "$grid_env" in
    CliffWalking|Bridge|ZigZag) ;;
    *) echo "Unknown grid environment: $grid_env" >&2; exit 2 ;;
esac
case "$grid_algorithm" in
    DDQN|TempoRL|UTE|RARe|TAACDiscrete) ;;
    *) echo "Unknown grid algorithm: $grid_algorithm" >&2; exit 2 ;;
esac
mode_args=()
grid_project="RARe-$grid_env"
case "$grid_mode" in
    fixed) ;;
    random)
        case "$grid_env" in
            CliffWalking) grid_steps=50000 ;;
            Bridge) grid_steps=150000 ;;
            ZigZag) grid_steps=200000 ;;
        esac
        mode_args=(--random --total-steps "$grid_steps")
        if (( grid_steps >= 100000 )); then
            mode_args+=(--replay-capacity 100000)
            if [[ "$grid_algorithm" != DDQN && "$grid_algorithm" != TAACDiscrete ]]; then
                mode_args+=(--skip-replay-capacity 300000)
            fi
        fi
        grid_project="RARe-R$grid_env"
        ;;
    *) echo "Unknown grid mode: $grid_mode" >&2; exit 2 ;;
esac

# Keep each named script tied to its environment, algorithm and W&B project.
for option in "$@"; do
    case "$option" in
        --env|--env=*|--algorithm|--algorithm=*|--random|--random=*|--wandb-project|--wandb-project=*)
            echo "This script fixes environment, algorithm, random mode and W&B project. Choose another script to change them." >&2
            exit 2
            ;;
    esac
done

grid_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$grid_root"
exec python -m experiments.train --env "$grid_env" --algorithm "$grid_algorithm" --wandb-project "$grid_project" "${mode_args[@]}" "$@"
