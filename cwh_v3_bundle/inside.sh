#!/bin/bash
set -Eeuo pipefail
ROOT=/work/cwh_qwen3_vllm
PACK="$ROOT/cwh_v3_bundle"
export PYTHONPATH="$PACK:$ROOT/vendor:$ROOT"
export VLLM_PLUGINS=cwh_v2 VLLM_USE_V1=1 TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$PACK"
python3 "$ROOT/runtime_fix/run_session.py" --stage "${CWH_STAGE:-focus}" --tag "${CWH_TAG:?tag missing}" \
  --budget-minutes "${CWH_BUDGET:-220}" --port "${CWH_PORT:-18765}" ${CWH_REPEAT:+--repeat "$CWH_REPEAT"}
