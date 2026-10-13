#!/usr/bin/env bash
set -euo pipefail

# Run four matched layout/initial-wind blocks on the permitted GPUs only.
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$root"
campaign="${1:-runs/campaigns/windshift-visible-60k}"
mkdir -p "$campaign"

algorithms=(DDQN TempoRL UTE TAACDiscrete RARe-se RARe-sh)
layouts=(descending descending ascending ascending)
initial_winds=(north south north south)
seed_ranges=(0:5 5:10 10:15 15:20)

for shifts in 0 2 4; do
    case "$shifts" in
        0) wind_schedule=(--wind-switch-interval 0) ;;
        2) wind_schedule=(--wind-switch-steps 20000,40000) ;;
        4) wind_schedule=(--wind-switch-steps 20000,30000,40000,50000) ;;
    esac
    for label in "${algorithms[@]}"; do
        pids=()
        for gpu in 0 1 2 3; do
            layout="${layouts[gpu]}"
            initial="${initial_winds[gpu]}"
            seeds="${seed_ranges[gpu]}"
            run_dir="$campaign/shift${shifts}/${label}/${layout}-${initial}"
            log="$campaign/shift${shifts}/${label}/${layout}-${initial}.log"
            mkdir -p "$(dirname -- "$run_dir")"
            if [[ -e "$run_dir/run.json" ]]; then
                if python -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["status"] != "completed")' "$run_dir/run.json"; then
                    echo "Already completed: $run_dir"
                    continue
                fi
                echo "Incomplete run requires inspection: $run_dir" >&2
                exit 1
            fi
            algorithm="$label"
            mode=()
            if [[ "$label" == RARe-* ]]; then
                algorithm=RARe
                if [[ "$label" == RARe-se ]]; then mode=(--rare-q-mode separate)
                else mode=(--rare-q-mode shared); fi
            fi
            python -m experiments.train \
                --env WindShift --algorithm "$algorithm" "${mode[@]}" \
                "${wind_schedule[@]}" --wind-probability 0.5 \
                --wind-layout "$layout" --wind-initial "$initial" \
                --wind-observation visible --seeds "$seeds" --gpu "$gpu" \
                --total-steps 60000 --warmup-steps 3000 \
                --eval-interval-steps 1000 --eval-episodes 20 \
                --model-save-interval 0 --wandb-mode online \
                --wandb-group "$label" \
                --wandb-name "${label}-WindShift-shift${shifts}-${layout}-${initial}" \
                --output "$run_dir" >"$log" 2>&1 &
            pids+=("$!")
            echo "Started: shift=$shifts algorithm=$label layout=$layout initial=$initial gpu=$gpu seeds=$seeds"
        done
        failed=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then failed=1; fi
        done
        if (( failed )); then
            echo "Failed: shift=$shifts algorithm=$label; inspect $campaign/shift${shifts}/${label}/*.log" >&2
            exit 1
        fi
        echo "Completed: shift=$shifts algorithm=$label"
    done
done
