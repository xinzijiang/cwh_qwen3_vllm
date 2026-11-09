#!/bin/bash
set -Eeuo pipefail
STAGE="${1:-focus}"
TAG="${2:?Usage: bash launch.sh STAGE TAG [minutes]}"
BUDGET="${3:-230}"
: "${SCRATCH:?SCRATCH is not set}"
: "${SLURM_JOB_ID:?First obtain four A100-80GB GPUs or submit nersc.sbatch}"
PACK_HOST="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ "$PACK_HOST" == "$SCRATCH/cwh_qwen3_vllm/cwh_v4_bundle" ]] || { echo 'Place this bundle at $SCRATCH/cwh_qwen3_vllm/cwh_v4_bundle'; exit 2; }
export CWH_IMAGE=docker:vllm/vllm-openai:v0.13.0
export CWH_JOB_RECORD="$(scontrol show job "$SLURM_JOB_ID" -o)" CWH_REQUEST_MINUTES="$BUDGET"
# Host Python 3.6 supported. Slurm timestamp interpreted in the host timezone.
export CWH_SAFE_DEADLINE="$(python3 - <<'PY'
import os,re,time
now=time.time()
m=re.search(r'\bEndTime=([^ ]+)',os.environ['CWH_JOB_RECORD'])
if not m or m.group(1) in ('Unknown','None'):raise SystemExit('Cannot determine allocation EndTime')
end=time.mktime(time.strptime(m.group(1),'%Y-%m-%dT%H:%M:%S'))
deadline=min(now+int(os.environ['CWH_REQUEST_MINUTES'])*60,end-300)
if deadline<=now+120:raise SystemExit('Insufficient remaining allocation time')
print(int(deadline))
PY
)"
[[ "$CWH_SAFE_DEADLINE" =~ ^[0-9]+$ ]] || exit 3
unset CWH_JOB_RECORD CWH_REQUEST_MINUTES
export CWH_STAGE="$STAGE" CWH_TAG="$TAG" CWH_BUDGET="$BUDGET"
srun -n 1 -G 4 --gpu-bind=none \
  shifter --image="$CWH_IMAGE" --module=gpu --volume="$SCRATCH:/work" \
  bash /work/cwh_qwen3_vllm/cwh_v4_bundle/inside.sh
