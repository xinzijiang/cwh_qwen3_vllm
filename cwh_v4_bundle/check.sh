#!/bin/bash
set -Eeuo pipefail
: "${SCRATCH:?}"
# CPU validation inside the image; no model serving or GPU allocation on login.
shifter --image=docker:vllm/vllm-openai:v0.13.0 --volume="$SCRATCH:/work" \
 bash -c 'set -e; export PYTHONPATH=/work/cwh_qwen3_vllm/cwh_v4_bundle; unset PYTHONHOME CWH_PORT_FIX; cd /work/cwh_qwen3_vllm/cwh_v4_bundle; python3 -m unittest discover -s tests -v; python3 tools/preflight.py'
