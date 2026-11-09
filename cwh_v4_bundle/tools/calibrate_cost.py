#!/usr/bin/env python3
"""Calibrate real-shape vLLM local fused-expert cost, on all four GPUs.

Synthetic activations/weights, real train routing IDs. This is a kernel cost
model, not a model-quality measurement or an end-to-end speed result.
"""
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
from durable import atomic_json,sha
from cwh_v4.cost import features,bucket_for,profile_bucket

def nnls_small(x,y):
    best=None
    for mask in itertools.product((0,1),repeat=x.shape[1]):
        ix=np.flatnonzero(mask);beta=np.zeros(x.shape[1])
        if len(ix):
            b=np.linalg.lstsq(x[:,ix],y,rcond=None)[0]
            if (b<0).any():continue
            beta[ix]=b
        error=float(np.square(x@beta-y).mean())
        if best is None or error<best[0]:best=(error,beta)
    return best[1]

def cuda_ns(fn,repeat=20):
    import torch
    for _ in range(3):fn()
    torch.cuda.synchronize()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):fn()
    end.record();end.synchronize()
    return start.elapsed_time(end)*1e6/repeat

def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--deadline',type=float,default=0);a=p.parse_args()
    import torch
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from cwh_v4.routing import redirect
    from cwh_v4.cost_cuda import redirect_cost
    mc=json.loads((a.model/'config.json').read_text());hd=mc['hidden_size'];inter=mc['moe_intermediate_size']
    result=json.loads(a.out.read_text()) if a.out.exists() else dict(schema=4,buckets={},scope='local fused_experts; synthetic BF16 weights; no collectives',model_config_sha=sha(a.model/'config.json'))
    groups={}
    for l in (0,8,16,24,32,40,47):
        with (a.capture/f'train_layer{l:02d}.jsonl').open() as f:
            for line in f:
                if not line.strip():continue
                r=json.loads(line);ids=np.asarray(r['ids']+r.get('padding_ids',[]),dtype=np.int32)
                if not len(ids):continue
                b=profile_bucket(len(ids))
                if b>32768:raise RuntimeError('Capture exceeds bounded 32768-row cost calibration')
                if len(groups.setdefault(b,[]))<24:groups[b].append(ids)
    # Ensure every supported bucket has samples independent of train coverage.
    rng=np.random.default_rng(20261108)
    for b in sorted(set(groups)|{1,2,4,8,16,32,64,128,256,512}):
        while len(groups.setdefault(b,[]))<24:
            logits=rng.normal(size=(b,128));logits[:,0:16]+=rng.uniform(0,4)
            groups[b].append(np.argsort(logits,axis=1)[:,-8:].astype(np.int32))
    for bucket,arrays in sorted(groups.items()):
        if str(bucket) in result['buckets']:continue
        if a.deadline and time.time()+240>a.deadline:raise SystemExit(75)
        measurements=[];overheads={k:[] for k in ('cw','ratio','dynamic','cost')}
        for device in range(4):
            with torch.cuda.device(device),torch.inference_mode():
                torch.manual_seed(100+device)
                w1=torch.randn(32,inter*2,hd,device=device,dtype=torch.bfloat16)*.01
                w2=torch.randn(32,hd,inter,device=device,dtype=torch.bfloat16)*.01
                emap=torch.full((128,),-1,device=device,dtype=torch.int32)
                emap[device*32:(device+1)*32]=torch.arange(32,device=device,dtype=torch.int32)
                for sample,arr in enumerate(arrays):
                    ids=torch.tensor(arr,device=device);n=len(arr)
                    x=torch.randn(n,hd,device=device,dtype=torch.bfloat16)*.1
                    weights=torch.full((n,8),.125,device=device,dtype=torch.float32)
                    fn=lambda:fused_experts(x,w1,w2,weights,ids,inplace=False,global_num_experts=128,expert_map=emap)
                    measured=cuda_ns(fn)
                    counts=np.bincount(arr.ravel(),minlength=128)[device*32:(device+1)*32]
                    measurements.append(dict(sample=sample,device=device,counts=counts.tolist(),ns=measured))
                cfg=dict(layer=0,enabled=True,cold=71,warm=72,hot=3,replica_fraction=.5,
                         cost_coeff_ns=[1000,100,10],cost_tile=16,online_gain_bp=100)
                route_arr=max(arrays,key=len)
                logits=torch.zeros(len(route_arr),128,device=device)
                ids=torch.tensor(route_arr,device=device)
                for mode in overheads:
                    if mode=='cost' and len(ids)>512:continue
                    fn=(lambda:redirect_cost(ids,cfg)) if mode=='cost' else (lambda m=mode:redirect(ids,logits,cfg,m,'triton'))
                    overheads[mode].append(cuda_ns(fn,repeat=100))
                del w1,w2,x,ids,weights,emap
                torch.cuda.empty_cache()
        # Held-out synthetic/routing shapes test fit; not held-out application data.
        train=[v for v in measurements if v['sample']%3!=0]
        val=[v for v in measurements if v['sample']%3==0]
        best=None
        for tile in (8,16,32,64):
            x=np.array([[1]+features(v['counts'],tile) for v in train],dtype=float)
            y=np.array([v['ns'] for v in train]);beta=nnls_small(x,y)
            xv=np.array([[1]+features(v['counts'],tile) for v in val],dtype=float)
            yv=np.array([v['ns'] for v in val]);pred=xv@beta
            err=np.abs(pred-yv)/np.maximum(yv,1)
            score=float(np.percentile(err,95))
            if best is None or score<best[0]:best=(score,tile,beta,float(np.median(err)))
        score,tile,beta,med=best
        result['buckets'][str(bucket)]=dict(cost_tile=tile,cost_coeff_ns=[int(round(x)) for x in beta[1:]],
            intercept_ns=int(round(beta[0])),fit_p95_relative_error=score,fit_median_relative_error=med,
            usable=bool(score<=.25 and sum(beta[1:])>0),
            route_ns={k:int(round(max(v))) if v else None for k,v in overheads.items()},measurements=measurements)
        atomic_json(a.out,result);print('COST_BUCKET',bucket,'p95_error',round(score,3),flush=True)
    result['complete']=True;atomic_json(a.out,result)
if __name__=='__main__':main()
