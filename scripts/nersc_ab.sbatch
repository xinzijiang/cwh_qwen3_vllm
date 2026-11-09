#!/bin/bash
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -t 02:00:00
#SBATCH --job-name=qwen3-cwh-ab
#SBATCH --output=qwen3_cwh_%j.log
# Requirements: image already shifterimg-pulled, data/model already in $SCRATCH.
# Full BF16 Qwen3 load twice can exceed requested allocation; adjust job walltime.
set -euo pipefail
IMAGE=${IMAGE:-docker:vllm/vllm-openai:v0.13.0}
srun -n 1 -G 4 --gpu-bind=none \
  shifter --image="$IMAGE" --module=gpu --volume="$SCRATCH:/work" \
  bash /work/cwh_qwen3_vllm/scripts/run_ab_inside_container.sh
