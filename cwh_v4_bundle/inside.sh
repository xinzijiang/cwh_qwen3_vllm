#!/bin/bash
set -Eeuo pipefail
ROOT=/work/cwh_qwen3_vllm
PACK="$ROOT/cwh_v4_bundle"
# No host/vendor/sitecustomize path: use exactly the pinned image and this plugin.
export PYTHONPATH="$PACK"
unset PYTHONHOME CWH_PORT_FIX
export VLLM_PLUGINS=cwh_v4 VLLM_USE_V1=1 TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# Remove stale experiment overrides copied from earlier interactive sessions.
unset CWH_FRACTION CWH_CAPTURE_DIR CWH_CAPTURE_MARKER CWH_POLICY CWH_MODE CWH_BACKEND VLLM_TORCH_PROFILER_DIR
cd "$PACK"
extra=()
if [[ -n "${CWH_IMPORT_V3:-}" ]]; then extra+=(--import-v3 "$CWH_IMPORT_V3"); fi
python3 tools/session.py --stage "${CWH_STAGE:-focus}" --tag "${CWH_TAG:?}" \
  --budget-minutes "${CWH_BUDGET:-230}" "${extra[@]}"
