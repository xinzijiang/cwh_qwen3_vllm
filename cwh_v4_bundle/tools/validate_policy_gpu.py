#!/usr/bin/env python3
"""Conservative measured veto after cost-model search, before serving.

Replay held-out routing shapes through local fused_experts + redirect on each
GPU. Uses synthetic BF16 weights/activations; excludes attention/collectives.
This is a kernel proxy and MUST NOT be reported as end-to-end throughput.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from durable import atomic_json,sha
from calibrate_cost import cuda_ns

def admissible(rows):
    if not rows:return False
    # Independent timing passes; demand a consistent gain, not a lucky median.
    for repeat in range(3):
        for phase in ('decode','prefill_mixed'):
            items=[r for r in rows if r['phase']==phase]
            if not items:return False
            native=np.array([max(r['native_ns'][repeat]) for r in items])
            changed=np.array([max(r['cost_ns'][repeat]) for r in items])
            gain=1-changed.sum()/native.sum()
            if gain < (.02 if phase=='decode' else -.01):return False
            if phase=='decode' and np.percentile(changed/native-1,95)>.05:return False
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--deadline',type=float,default=0);a=p.parse_args()
    import torch
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from cwh_v4.routing import redirect
    candidate=json.loads(a.candidate.read_text());checkpoint=a.out.with_suffix('.replay.partial.json')
    state=json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(candidate_sha=sha(a.candidate),layers=[])
    if state['candidate_sha']!=sha(a.candidate):raise RuntimeError('Replay candidate changed')
    mc=json.loads((a.model/'config.json').read_text());hd=mc['hidden_size'];inter=mc['moe_intermediate_size']
    for cfg in candidate['layers'][len(state['layers']):]:
        if a.deadline and time.time()+240>a.deadline:raise SystemExit(75)
        cfg=dict(cfg);layer=cfg['layer'];rows=[]
        if cfg['enabled']:
            records=[json.loads(s) for s in (a.capture/f'validation_layer{layer:02d}.jsonl').open() if s.strip()]
            samples=[]
            for decode,limit in ((True,6),(False,2)):
                pool=[r for r in records if bool(all(r['decode']))==decode]
                for ix in np.linspace(0,len(pool)-1,min(len(pool),limit),dtype=int):
                    r=pool[ix];samples.append(r)
                    rows.append(dict(call=r['call'],phase='decode' if decode else 'prefill_mixed',
                                     native_ns=[[] for _ in range(3)],cost_ns=[[] for _ in range(3)]))
            for device in range(4):
                with torch.cuda.device(device),torch.inference_mode():
                    torch.manual_seed(1000+device)
                    w1=torch.randn(32,2*inter,hd,device=device,dtype=torch.bfloat16)*.01
                    w2=torch.randn(32,hd,inter,device=device,dtype=torch.bfloat16)*.01
                    emap=torch.full((128,),-1,device=device,dtype=torch.int32)
                    emap[device*32:(device+1)*32]=torch.arange(32,device=device,dtype=torch.int32)
                    for row,r in zip(rows,samples):
                        arr=r['ids']+r.get('padding_ids',[]);n=len(arr)
                        ids=torch.tensor(arr,device=device,dtype=torch.int32)
                        x=torch.randn(n,hd,device=device,dtype=torch.bfloat16)*.1
                        weights=torch.full((n,8),.125,device=device,dtype=torch.float32)
                        logits=torch.empty(n,128,device=device,dtype=torch.float32)
                        def native():
                            return fused_experts(x,w1,w2,weights,ids,inplace=False,global_num_experts=128,expert_map=emap)
                        def changed():
                            out=redirect(ids,logits,cfg,'cost','triton',call=r['call'])
                            return fused_experts(x,w1,w2,weights,out,inplace=False,global_num_experts=128,expert_map=emap)
                        for repeat in range(3):
                            order=[('native_ns',native),('cost_ns',changed)]
                            if (repeat+device)%2:order.reverse()
                            for name,fn in order:row[name][repeat].append(cuda_ns(fn,repeat=20))
                    del w1,w2,emap
                    torch.cuda.empty_cache()
            ok=admissible(rows)
            cfg.update(enabled=ok,measured_replay=rows,
                       reason='accepted measured local replay' if ok else 'measured local replay veto')
        state['layers'].append(cfg);atomic_json(checkpoint,state)
        print('MEASURED_VETO',layer,cfg['enabled'],cfg['reason'],flush=True)
    candidate['layers']=state['layers'];candidate['active_layers']=sum(r['enabled'] for r in state['layers'])
    candidate['measured_veto']=dict(candidate_sha=state['candidate_sha'],decode_min_gain=.02,prefill_max_regression=.01,
       timing_passes=3,scope='max of sequential per-rank local replay; no collectives/attention; synthetic weights',
       caveat='Validation veto only. Correlated samples; not a confidence interval or end-to-end guarantee.')
    atomic_json(a.out,candidate)
if __name__=='__main__':main()
