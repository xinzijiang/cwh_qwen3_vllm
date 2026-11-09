#!/usr/bin/env python3
"""Required CUDA gate, automatic before formal runs. No model weights needed."""
import json
import torch
from cwh_v4.policy import reference_ids
from cwh_v4.routing import redirect


def main():
    torch.manual_seed(21)
    cfg=dict(layer=17,enabled=True,cold=71,warm=34,hot=16,replica_fraction=.5)
    checks=0
    for n in (1,4,16,17,902,2048):
        logits=torch.randn(n,128,device='cuda')
        ids=torch.topk(logits,8,dim=-1).indices.to(torch.int32)
        # Explicit collision: C,W,H all selected in a row.
        ids[0,:3]=torch.tensor([71,34,16],device='cuda',dtype=ids.dtype)
        old=ids.clone()
        for p in (0,.25,.5,.75,1):
            for call in (1,12345):
                for mode in ('cw','ratio','legacy'):
                    a=redirect(ids,logits,cfg,mode,'torch',p,call)
                    b=redirect(ids,logits,cfg,mode,'triton',p,call)
                    if not torch.equal(a,b):
                        raise AssertionError(f'Fused/reference mismatch {n,p,call,mode}')
                    if mode != 'legacy':
                        oracle=reference_ids(ids.cpu().tolist(),cfg,mode,p,call)
                        if a.cpu().tolist()!=oracle:
                            raise AssertionError('Python oracle mismatch')
                    if not torch.equal(ids,old):
                        raise AssertionError('Input IDs were modified in-place')
                    checks+=1
        for mode in ('legacy_shadow','shadow'):
            got=redirect(ids,logits,cfg,mode,'torch',.5,1)
            cw=redirect(ids,logits,cfg,'cw','torch',.5,1)
            if not torch.equal(got,cw):raise AssertionError('Shadow changes dispatch')
    from cwh_v4.dynamic import dynamic_reference
    dynamic_checks=0
    for n in (1,16,512,513,902,4096):
        # Strong H skew exercises nonzero quotas and block-boundary prefixes.
        ids=torch.randint(0,128,(n,8),device='cuda',dtype=torch.int32)
        ids[:,0]=16;ids[:,1]=17;ids[:,2]=18
        logit=torch.empty(n,128,device='cuda')
        for bp in (0,100,10000):
            oracle=dynamic_reference(ids.cpu().tolist(),cfg,bp)
            out=redirect(ids,logit,cfg|dict(online_gain_bp=bp),'dynamic','triton')
            if out.cpu().tolist()!=oracle:raise AssertionError(f'Dynamic CUDA quota/prefix mismatch {n,bp}')
            dynamic_checks+=1
    for source in range(4):
        for dest in range(4):
            if source==dest:continue
            for wg in range(4):
                cc=dict(layer=0,enabled=True,hot=source*32+3,cold=dest*32+7,
                        warm=wg*32+9,replica_fraction=.5)
                ids=torch.randint(0,128,(16,8),device='cuda',dtype=torch.int32)
                ids[:,:3]=torch.tensor([cc['hot'],source*32+4,source*32+5],device='cuda')
                ids[0,3]=cc['cold']
                oracle=dynamic_reference(ids.cpu().tolist(),cc)
                out=redirect(ids,logit[:16],cc,'dynamic','triton')
                if out.cpu().tolist()!=oracle:raise AssertionError(f'Dynamic owner mapping mismatch {source,dest,wg}')
                dynamic_checks+=1
    # Validate actual vLLM fused expert duplicate handling and replica equivalence.
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    n,h,inter=16,128,128
    x=torch.randn(n,h,device='cuda',dtype=torch.bfloat16)*.1
    w1=torch.randn(128,2*inter,h,device='cuda',dtype=x.dtype)*.05
    w2=torch.randn(128,h,inter,device='cuda',dtype=x.dtype)*.05
    w1[71]=w1[16];w2[71]=w2[16]
    # Low-level function is pinned to this vLLM version. Fail on incompatible API.
    ids=torch.randint(0,128,(n,8),device='cuda',dtype=torch.int32)
    ids[:,0]=16;ids[:,1]=16  # explicit duplicate routes (preserve both weights)
    weights=torch.softmax(torch.randn(n,8,device='cuda'),dim=-1)
    y=fused_experts(x,w1,w2,weights,ids,inplace=False)
    rid=ids.clone();rid[:,0]=71
    z=fused_experts(x,w1,w2,weights,rid,inplace=False)
    expected=torch.zeros_like(x,dtype=torch.float32)
    for t in range(n):
        for j in range(8):
            e=int(ids[t,j])
            gu=torch.nn.functional.linear(x[t].float(),w1[e].float())
            g,u=gu.chunk(2)
            v=torch.nn.functional.linear(torch.nn.functional.silu(g)*u,w2[e].float())
            expected[t]+=weights[t,j]*v
    torch.testing.assert_close(y.float(),expected,atol=3e-3,rtol=.05)
    torch.testing.assert_close(y.float(),z.float(),atol=3e-3,rtol=.05)
    torch.cuda.synchronize()
    print(json.dumps(dict(routing_checks=checks,dynamic_checks=dynamic_checks, fused_duplicate_and_replica_checks=True,
                          torch=torch.__version__, GPU=torch.cuda.get_device_name())))


if __name__=='__main__':main()
