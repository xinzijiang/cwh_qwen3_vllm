#!/bin/bash
set -Eeuo pipefail
TAG="${1:?Usage: bash analyze.sh TAG}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
shifter --image=docker:vllm/vllm-openai:v0.13.0 --volume="$SCRATCH:/work" \
 bash -c 'set -e; export PYTHONPATH=/work/cwh_qwen3_vllm/cwh_v4_bundle; unset PYTHONHOME; cd /work/cwh_qwen3_vllm/cwh_v4_bundle; python3 tools/audit_configs.py "/work/cwh_qwen3_vllm/runs/cwh_v4_$1"; python3 tools/summarize_v4.py "/work/cwh_qwen3_vllm/runs/cwh_v4_$1"' _ "$TAG"
