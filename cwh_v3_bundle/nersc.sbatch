#!/bin/bash
#SBATCH -J CWH_V3
#SBATCH -A m4431_g
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -t 12:00:00
#SBATCH -o CWH_v3_%j.out
#SBATCH -e CWH_v3_%j.err
set -Eeuo pipefail
bash "$SCRATCH/cwh_qwen3_vllm/cwh_v3_bundle/launch.sh" "${1:-focus}" "${2:?tag required}" 690
