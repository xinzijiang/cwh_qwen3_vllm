#!/usr/bin/env python3
"""Generate a deterministic Qwen3 EP4 fixed-slot policy from calibration-only traces."""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict

def build(trace, steps=8):
    counts=defaultdict(lambda:[0]*128)
    with open(trace,newline='') as f:
        reader=csv.DictReader(f)
        for row in reader:
            step,layer,e,n=map(int,(row['step'],row['layer'],row['expert'],row['token_count']))
            if step < steps: counts[layer][e]+=n
    out=[]
    for l in range(48):
        a=counts[l];avg=sum(a)/128
        if not sum(a): raise RuntimeError(f'No data for layer {l}')
        pool=[e for e in range(64,96) if a[e]>0]
        if not pool: raise RuntimeError(f'Layer {l} lacks nonzero cold candidate')
        c=min(pool,key=lambda e:(abs(a[e]-max(1,0.25*avg)),a[e],e))
        w_pool=[e for e in range(32,64) if a[e]>a[c]]
        if not w_pool: raise RuntimeError(f'No eligible W layer {l}')
        w=min(w_pool,key=lambda e:(abs(a[e]-avg),e))
        h=max(range(0,32),key=lambda e:(a[e],-e))
        out.append({'layer':l,'cold':c,'warm':w,'hot':h,
                    'cal_c_count':a[c],'cal_w_count':a[w],'cal_h_count':a[h],
                    'cal_mean':avg})
    return {'model':'Qwen3-30B-A3B','method':'fixed_C_slot_reused_as_H_replica',
            'calibration':f'prefill first {steps} prompts; no held-out evaluation data',
            'ep_size':4,'placement':'linear','expert_slots_per_rank_per_layer':32,
            'layers':out}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--trace',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--cal-prompts',type=int,default=8)
    a=p.parse_args()
    out=build(a.trace,a.cal_prompts)
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2))
    print('WROTE POLICY',path,'48 layers; EP0=H, EP1=W, EP2=C')
    print('EXAMPLE',out['layers'][0])
