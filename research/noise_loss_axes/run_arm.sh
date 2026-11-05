#!/usr/bin/env bash
# One independent tmux session per arm; real training only, no reservation jobs.
set -u
cd /home/t-lab01/sunrun
study=research/noise_loss_axes
arm="$1"
threads="${2:-2}"
mkdir -p "$study/logs" "$study/results/$arm"
for seed in 42 43 44; do
    out="$study/results/$arm/seed$seed"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        .venv/bin/python -u "$study/experiment.py" --arm "$arm" --seed "$seed" \
        --threads "$threads" --out "$out" > "$study/logs/${arm}_seed${seed}.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "$out/process.exit"
    if [ "$code" -ne 0 ]; then
        printf '%s\n' "$code" > "$study/results/$arm/process.exit"
        exit "$code"
    fi
done
printf '0\n' > "$study/results/$arm/process.exit"
