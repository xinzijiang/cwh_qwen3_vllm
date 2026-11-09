#!/usr/bin/env python3
"""Run in the original container, before allocating model weights."""
import argparse,hashlib,importlib.metadata,inspect,json,os,platform,shutil,subprocess,sys
from pathlib import Path
from durable import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');p.add_argument('--out',type=Path)
    p.add_argument('--model',type=Path,default=Path('/work/qwen3_real/models/Qwen3-30B-A3B'))
    p.add_argument('--source',type=Path,default=Path('/work/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json'))
    p.add_argument('--qa',type=Path,default=Path('/work/openbookqa/validation.jsonl'));a=p.parse_args()
    if sys.version_info<(3,10):raise RuntimeError('Run inside the pinned Shifter image; host Python is too old')
    import torch,triton,numpy,safetensors,transformers,vllm
    if vllm.__version__!='0.13.0':raise RuntimeError('Require vLLM 0.13.0; do not pip-upgrade the image')
    for path in (a.source,a.qa,a.model/'config.json',a.model/'model.safetensors.index.json'):
        if not path.is_file():raise FileNotFoundError(str(path))
    index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    for shard in set(index.values()):
        if not (a.model/shard).is_file():raise FileNotFoundError(shard)
    mc=json.loads((a.model/'config.json').read_text())
    if (mc.get('num_hidden_layers'),mc.get('num_experts'),mc.get('num_experts_per_tok'))!=(48,128,8):
        raise RuntimeError('Expected Qwen3-30B-A3B 48-layer 128-expert Top-8 config')
    if mc.get('quantization_config'):raise RuntimeError('Require original BF16 checkpoint')
    from evaluate_openbookqa import parse_record
    path=a.qa
    if path.suffix=='.jsonl':rows=[json.loads(line) for line in path.open() if line.strip()]
    else:
        obj=json.loads(path.read_text());rows=obj.get('data',[]) if isinstance(obj,dict) else obj
    if len(rows)!=500:raise RuntimeError('Expected 500 OpenBookQA questions')
    for row in rows:parse_record(row)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True)
    for tid,letter in zip([362,425,356,422],'ABCD'):
        if tok.decode([tid]).strip()!=letter:raise RuntimeError('Tokenizer letter IDs mismatch')
    from cwh_v4.plugin import register
    register()
    sig='GPU API check deferred to allocated node'
    if a.gpu:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        sig=inspect.signature(fused_experts)
        for kw in ('expert_map','global_num_experts'):
            if kw not in sig.parameters:raise RuntimeError('Incompatible fused_experts API: '+str(sig))
    result=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                triton=triton.__version__,vllm=vllm.__version__,transformers=transformers.__version__,
                image=os.getenv('CWH_IMAGE',''),gpu_count=torch.cuda.device_count() if a.gpu else None,
                model=str(a.model),source=str(a.source),qa=str(a.qa),fused_experts_signature=str(sig))
    if a.gpu:
        if torch.cuda.device_count()!=4:raise RuntimeError('Require exactly four visible GPUs')
        result['gpus']=[dict(name=torch.cuda.get_device_name(i),memory=torch.cuda.get_device_properties(i).total_memory) for i in range(4)]
        if any('A100' not in x['name'] or x['memory']<75*1024**3 for x in result['gpus']):raise RuntimeError('Request gpu&hbm80g')
        result['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,power.limit','--format=csv'],text=True)
    if a.out:atomic_json(a.out,result)
    print(json.dumps(result,indent=2));print('PREFLIGHT_OK',flush=True)
if __name__=='__main__':main()
