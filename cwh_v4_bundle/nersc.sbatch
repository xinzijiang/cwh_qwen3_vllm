#!/bin/bash
#SBATCH -J CWH_V4
#SBATCH -A m4431_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -t 10:00:00
#SBATCH -o CWH_v4_%j.out
#SBATCH -e CWH_v4_%j.err
set -Eeuo pipefail
bash "$SCRATCH/cwh_qwen3_vllm/cwh_v4_bundle/launch.sh" "${1:-focus}" "${2:?tag required}" 10000
