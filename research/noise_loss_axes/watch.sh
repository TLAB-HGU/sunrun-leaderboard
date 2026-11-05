#!/usr/bin/env bash
set -u
cd /home/t-lab01/sunrun
study=research/noise_loss_axes
mkdir -p "$study/logs"
CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    .venv/bin/python -u "$study/controller.py" --interval 60 > "$study/logs/watch.log" 2>&1
code=$?
printf '%s\n' "$code" > "$study/controller.exit"
exit "$code"
