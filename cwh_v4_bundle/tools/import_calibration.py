"""Verified reuse of V3 calibration only. Never imports V3 timing/QA as V4 results."""
import json,os
from pathlib import Path
from durable import atomic_json,atomic_copy,sha
from check_artifacts import check_capture

def import_v3(source,target,model,qa,data):
    spec=json.loads((source/'study_spec.json').read_text())
    for key,path in [('source',data),('qa',qa),('model_config',model/'config.json'),('model_index',model/'model.safetensors.index.json')]:
        if spec[key]!=sha(path):raise RuntimeError('V3 calibration input mismatch: '+key)
    out=target/'calibration'/'imported';out.mkdir(parents=True,exist_ok=True)
    hashes={};sources={}
    for split in ('train','validation'):
        done=json.loads((source/'calibration'/('.done_'+split+'.json')).read_text())
        for name,digest in done['files'].items():
            if sha(source/'calibration'/name)!=digest:raise RuntimeError('V3 completed artifact changed')
        summary=json.loads((source/'calibration'/(split+'.json')).read_text())
        if summary['success_count']!=800 or summary['error_count'] or summary['total_output_tokens']!=204800:
            raise RuntimeError('Incomplete V3 calibration')
        cap=Path(done['attempt'])/'native_probe'/'capture';check_capture(cap)
        for l in range(48):
            path=cap/(split+f'_layer{l:02d}.jsonl')
            phases=set()
            for line in path.open():
                if not line.strip():continue
                row=json.loads(line)
                if len(row['ids'])!=len(row['decode']):raise RuntimeError('Invalid capture dimensions')
                phases.update(row['decode'])
            if phases!={0,1}:raise RuntimeError('V3 capture lacks real prefill/decode')
            for original in [path]+([cap/f'hidden_layer{l:02d}.npyseq'] if split=='train' else []):
                dest=out/original.name;digest=sha(original)
                if not dest.exists() or sha(dest)!=digest:atomic_copy(original,dest)
                hashes[str(dest.relative_to(target))]=digest;sources[original.name]=str(original)
    # Preserve old source and exact workload evidence, but make new held-out data.
    receipt=dict(source=str(source),source_spec=spec,files=hashes,sources=sources,
                 note='Calibration only; no old performance result is reused')
    atomic_json(target/'calibration/import_receipt.json',receipt)
    return out
