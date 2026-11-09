#!/usr/bin/env python3
"""GPU/CPU oracle comparison and all-rank replica semantics, before real tests."""
import json,random
import torch
from cwh_v4.cost import reference
from cwh_v4.cost_cuda import redirect_cost

def main():
    rng=random.Random(55);checks=0;nonzero=0
    for device in range(4):
        with torch.cuda.device(device):
            for n in (1,2,4,16,17,64,128,512):
                for src,dst in ((0,1),(1,2),(2,3),(3,0),(1,3),(2,0)):
                    h=src*32+3;c=dst*32+7;w=((src+dst)%4)*32+9
                    coeff=([1200,200,10],[1200,0,0],[0,200,10],[0,0,10])[checks%4]
                    cfg=dict(cold=c,warm=w,hot=h,cost_coeff_ns=coeff,cost_tile=(8,16,32,64)[checks%4],
                             online_gain_bp=100 if checks%3 else 0,online_margin_ns=100000 if checks%7==0 else 0)
                    rows=[rng.sample(range(128),8) for _ in range(n)]
                    # Force source load skew without requiring unique logical routes.
                    for row in rows:row[:4]=[h,src*32+1,src*32+2,src*32+4]
                    ids=torch.tensor(rows,device=device,dtype=torch.int32);old=ids.clone()
                    want,m=reference(rows,cfg);nonzero+=m>0
                    got=redirect_cost(ids,cfg)
                    if got.cpu().tolist()!=want:raise AssertionError((device,n,src,dst))
                    shadow=redirect_cost(ids,cfg,True)
                    if shadow.cpu().tolist()!=reference(rows,cfg,True)[0]:raise AssertionError('shadow mismatch')
                    if not torch.equal(ids,old):raise AssertionError('Input mutated')
                    checks+=1
    if not nonzero:raise AssertionError('No nonzero routing case exercised')
    print(json.dumps(dict(cost_oracle_checks=checks,nonzero_cases=nonzero,all_four_gpus=True)))
if __name__=='__main__':main()
