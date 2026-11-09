#!/usr/bin/env python3
"""Validate checkpoint tensor naming and all 48 C/H triples before GPU allocation."""
import argparse,json
from pathlib import Path

def validate(model_path,policy_path):
    model=Path(model_path)
    p=model/'model.safetensors.index.json'
    if not p.exists():
        raise FileNotFoundError(f'{p} missing: expected official BF16 sharded checkpoint')
    index=json.loads(p.read_text()).get('weight_map',{})
    cfg=json.loads(Path(policy_path).read_text())
    missing=[]
    for row in cfg['layers']:
        l=row['layer']
        for e in (row['cold'],row['hot']):
            for mat in ('gate_proj','up_proj','down_proj'):
                key=f'model.layers.{l}.mlp.experts.{e}.{mat}.weight'
                if key not in index:missing.append(key)
    if missing:
        raise RuntimeError(f'Official Qwen3 checkpoint naming mismatch! First missing: {missing[:5]}. '
                           'Do not run GPU load until resolved.')
    return {'layers':len(cfg['layers']),'verified_h_c_weight_keys':len(cfg['layers'])*6}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--policy',required=True)
    x=a.parse_args();print('CHECKPOINT INDEX OK',validate(x.model,x.policy))
