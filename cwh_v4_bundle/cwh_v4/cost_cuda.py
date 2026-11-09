"""One CTA: expert histogram, five quotas, cost gate, prefix and ID remap.

No host sync, state across calls, collectives, or weight relocation. Large calls
use the explicitly calibrated static fallback in routing.py. Eager EP4 only.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def cost_dispatch(IDS,OUT,DIAG,N,C:tl.constexpr,W:tl.constexpr,H:tl.constexpr,
                  ACTIVE:tl.constexpr,TILES:tl.constexpr,TOKENS:tl.constexpr,
                  TILE:tl.constexpr,MARGIN:tl.constexpr,GAIN_BP:tl.constexpr,
                  SHADOW:tl.constexpr,B:tl.constexpr):
    i=tl.arange(0,B)
    valid=i<N
    e=tl.load(IDS+i,valid,other=128)
    # 256 bins include an out-of-range sentinel, avoiding masked padding in bin 0.
    raw=tl.histogram(e,256)
    ex=tl.arange(0,256)
    nc=tl.sum(tl.where(ex==C,raw,0),0)
    nh=tl.sum(tl.where(ex==H,raw,0),0)
    base=tl.where(ex==C,0,raw+tl.where(ex==W,nc,0))
    costs=(base>0).to(tl.int64)*ACTIVE+((base+TILE-1)//TILE).to(tl.int64)*TILES+base.to(tl.int64)*TOKENS
    l0=tl.sum(tl.where(ex<32,costs,0),0)
    l1=tl.sum(tl.where((ex>=32)&(ex<64),costs,0),0)
    l2=tl.sum(tl.where((ex>=64)&(ex<96),costs,0),0)
    l3=tl.sum(tl.where((ex>=96)&(ex<128),costs,0),0)
    src=l0 if H//32==0 else l1 if H//32==1 else l2 if H//32==2 else l3
    dst=l0 if C//32==0 else l1 if C//32==1 else l2 if C//32==2 else l3
    other=tl.maximum(tl.maximum(l0 if H//32!=0 and C//32!=0 else 0,l1 if H//32!=1 and C//32!=1 else 0),
                     tl.maximum(l2 if H//32!=2 and C//32!=2 else 0,l3 if H//32!=3 and C//32!=3 else 0))
    baseline=tl.maximum(tl.maximum(l0,l1),tl.maximum(l2,l3))
    original_h=(nh>0).to(tl.int64)*ACTIVE+((nh+TILE-1)//TILE).to(tl.int64)*TILES+nh.to(tl.int64)*TOKENS
    best=baseline
    quota=tl.full((),0,tl.int32)
    for fraction in tl.static_range(1,5):
        m=nh*fraction//4
        left=nh-m
        a=(left>0).to(tl.int64)*ACTIVE+((left+TILE-1)//TILE).to(tl.int64)*TILES+left.to(tl.int64)*TOKENS
        b=(m>0).to(tl.int64)*ACTIVE+((m+TILE-1)//TILE).to(tl.int64)*TILES+m.to(tl.int64)*TOKENS
        candidate=tl.maximum(other,tl.maximum(src-original_h+a,dst+b))
        better=candidate<best
        quota=tl.where(better,m,quota)
        best=tl.minimum(best,candidate)
    margin=tl.maximum(MARGIN,(baseline*GAIN_BP+9999)//10000)
    quota=tl.where(baseline-best>margin,quota,0)
    ordinal=tl.cumsum((valid&(e==H)).to(tl.int32),0)
    output=tl.where(e==C,W,e)
    if not SHADOW:
        output=tl.where((e==H)&(ordinal<=quota),C,output)
    else:
        # Observable output keeps the histogram and decision alive in shadow runs.
        tl.store(DIAG,quota)
    tl.store(OUT+i,output,valid)

def redirect_cost(ids,cfg,shadow=False):
    if not ids.is_cuda or not ids.is_contiguous() or ids.ndim!=2:
        raise RuntimeError('Expected contiguous CUDA routing IDs')
    if ids.numel()==0:return ids
    out=torch.empty_like(ids)
    scratch=torch.empty((),device=ids.device,dtype=torch.int32) if shadow else out
    a,b,c=map(int,cfg['cost_coeff_ns'])
    cost_dispatch[(1,)](ids,out,scratch,ids.numel(),cfg['cold'],cfg['warm'],cfg['hot'],
        a,b,c,int(cfg['cost_tile']),int(cfg.get('online_margin_ns',0)),int(cfg.get('online_gain_bp',100)),
        shadow,triton.next_power_of_2(ids.numel()),num_warps=4 if ids.numel()<=1024 else 8)
    return out
