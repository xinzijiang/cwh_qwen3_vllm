#!/usr/bin/env bash
# Run INSIDE vLLM 0.13.0 container on an allocated 4xA100 GPU node.
# Expects /work/cwh_qwen3_vllm and /work/qwen3_real/models/Qwen3-30B-A3B.
set -euo pipefail

ROOT=${ROOT:-/work/cwh_qwen3_vllm}
MODEL=${MODEL:-/work/qwen3_real/models/Qwen3-30B-A3B}
DATASET=${DATASET:-/work/openbookqa/validation.jsonl}
OUT=${OUT:-/work/cwh_qwen3_vllm/runs/ab_$(date +%Y%m%d_%H%M%S)}
export CWH_POLICY="$ROOT/data/cwh_policy_48layers.json"
export CWH_AUDIT_DIR="$OUT/weight_audits"
export PYTHONPATH="$ROOT/vendor:$ROOT:${PYTHONPATH:-}"
export NCCL_NET=${NCCL_NET:-Socket}
export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export VLLM_USE_V1=1
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export CWH_TRACE_ROUTING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$OUT"

python3 - <<'PY'
import importlib.metadata as im,os
ver=im.version('vllm')
print('vLLM VERSION',ver,flush=True)
assert ver.startswith('0.13.0'),f'Prototype requires v0.13.0, got {ver}'
import torch
print('CUDA GPUS',torch.cuda.device_count(),flush=True)
assert torch.cuda.device_count()==4
from cwh_vllm.policy import read_policy
print('POLICY LAYERS',len(read_policy(os.environ['CWH_POLICY'])))
from importlib.metadata import entry_points
plugins=[ep.name for ep in entry_points(group='vllm.general_plugins')]
assert 'cwh_model' in plugins,('CWH plugin not installed in',plugins)
print('CWH PLUGIN REGISTERED',flush=True)
PY
if [ ! -f "$DATASET" ];then echo "Missing $DATASET. Prepare OpenBookQA first." >&2;exit 2;fi
if [ ! -f "$MODEL/config.json" ];then echo "Missing model dir $MODEL" >&2;exit 2;fi

server_pid=''
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=''
  fi
}
trap cleanup EXIT INT TERM

wait_ready() {
  python3 - "$server_pid" <<'PY'
import sys,time,urllib.request,os
pid=int(sys.argv[1]);url='http://127.0.0.1:8000/v1/models'
for _ in range(480):
    try:
        with urllib.request.urlopen(url,timeout=3) as response:
            if response.status==200: print('SERVER READY',flush=True);sys.exit(0)
    except Exception:
        pass
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        print('Server exited before ready',file=sys.stderr);sys.exit(1)
    time.sleep(2)
print('Server never became ready',file=sys.stderr);sys.exit(1)
PY
}

run_mode() {
  local mode=$1
  shift
  echo "===== START $mode =====" | tee -a "$OUT/trace.txt"
  export VLLM_PLUGINS=cwh_model
  vllm serve "$MODEL" \
    --served-model-name qwen3-moe \
    --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 4 --enable-expert-parallel \
    --attention-backend FLASH_ATTN --dtype bfloat16 --safetensors-load-strategy eager --max-model-len 4096 \
    --max-num-seqs 16 --gpu-memory-utilization 0.85 --enforce-eager \
    --disable-log-requests \
    "$@" >"$OUT/${mode}_server.log" 2>&1 &
  server_pid=$!
  if ! wait_ready;then tail -80 "$OUT/${mode}_server.log";exit 1;fi
  python3 "$ROOT/scripts/bench_vllm_streaming.py" \
    --base-url http://127.0.0.1:8000 --served-model qwen3-moe \
    --tokenizer "$MODEL" --dataset-jsonl "$DATASET" \
    --mode "$mode" --out "$OUT/$mode" \
    --concurrencies 1,4,16 --output-tokens 128 --ignore-eos --repeats 3 --warmup 2
  python3 "$ROOT/scripts/bench_openbookqa_mc_accuracy.py" \
    --base-url http://127.0.0.1:8000 --served-model qwen3-moe \
    --tokenizer "$MODEL" --dataset-jsonl "$DATASET" --out "$OUT/$mode" \
    --mode "$mode" --split validation --limit 500 --concurrency 16
  cleanup
  # Port must no longer be served by old process before next mode.
  python3 - <<'PY'
import time,urllib.request
for _ in range(120):
    try: urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=1)
    except Exception: break
    time.sleep(1)
else: raise RuntimeError('Old vLLM server still listening; stop it before custom run')
PY
  echo "===== END $mode =====" | tee -a "$OUT/trace.txt"
}

# run_mode eplb --enable-eplb --eplb-config '{"num_redundant_experts":4,"window_size":256,"step_interval":100}'
run_mode custom_full_method --hf-overrides '{"architectures":["CWHQwen3MoeForCausalLM"]}'

# Fail if custom engine did not reach the expected CWH initialization path.
python3 - "$CWH_AUDIT_DIR" <<'PY'
from pathlib import Path
import json,sys
root=Path(sys.argv[1]); files=list(root.glob('cwh_weights_*.json'))
assert len(files)>=4,f'Expected at least 4 rank audits; got {len(files)}'
for f in files:
    obj=json.loads(f.read_text())
    assert obj['expert_weight_tensors_aliased']==144
    assert obj['original_c_tensors_skipped']==144
print('CWH CHECKPOINT ALIAS AUDITS PASS',len(files),flush=True)
PY
python3 "$ROOT/scripts/compare_bench.py" \
  --baseline "$OUT/eplb/summary.csv" \
  --custom "$OUT/custom_full_method/summary.csv" \
  --out "$OUT/comparison.csv"
python3 - "$OUT" <<'PYACC'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);a=json.loads((root/'eplb/accuracy_summary.json').read_text());b=json.loads((root/'custom_full_method/accuracy_summary.json').read_text())
print('OPENBOOKQA ACCURACY',a['accuracy_pct'], '->', b['accuracy_pct'], 'change_pp=',b['accuracy_pct']-a['accuracy_pct'])
(root/'accuracy_comparison.json').write_text(json.dumps({'eplb_accuracy_pct':a['accuracy_pct'],'custom_accuracy_pct':b['accuracy_pct'],'change_pp':b['accuracy_pct']-a['accuracy_pct'],'questions':a['questions']},indent=2))
PYACC
echo "RESULTS $OUT/comparison.csv and $OUT/accuracy_comparison.json"
