"""GPU-only online dispatch; gathered EP4 callback, no additional collective.

Small batches: one CTA handles the entire routing call (one extra launch).
Large batches: one partial-count launch and one prefix/remap launch.
Each original H assignment gets a global ordinal; exactly quota assignments move.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def quota_gpu(a,b,other,hot,peak,GAIN_BP:tl.constexpr):
    target=tl.maximum(tl.maximum(other,(a+b+1)//2),tl.maximum(a-hot,b))
    m=tl.maximum(0,a-target)
    benefit=peak-target
    return tl.where((benefit>=1)&(benefit.to(tl.int64)*10000>=GAIN_BP*peak.to(tl.int64)),m,0)


@triton.jit
def dynamic_small(IDS,OUT,N,C:tl.constexpr,W:tl.constexpr,H:tl.constexpr,
                  GAIN_BP:tl.constexpr,B:tl.constexpr):
    i=tl.arange(0,B)
    valid=i<N
    e=tl.load(IDS+i,valid,other=-1)
    base=tl.where(e==C,W,e)
    g=base//32
    l0=tl.sum((valid&(g==0)).to(tl.int32),0)
    l1=tl.sum((valid&(g==1)).to(tl.int32),0)
    l2=tl.sum((valid&(g==2)).to(tl.int32),0)
    l3=tl.sum((valid&(g==3)).to(tl.int32),0)
    a=l0 if H//32==0 else l1 if H//32==1 else l2 if H//32==2 else l3
    b=l0 if C//32==0 else l1 if C//32==1 else l2 if C//32==2 else l3
    o0=l0 if H//32!=0 and C//32!=0 else 0
    o1=l1 if H//32!=1 and C//32!=1 else 0
    o2=l2 if H//32!=2 and C//32!=2 else 0
    o3=l3 if H//32!=3 and C//32!=3 else 0
    other=tl.maximum(tl.maximum(o0,o1),tl.maximum(o2,o3))
    hot=valid&(e==H)
    nh=tl.sum(hot.to(tl.int32),0)
    peak=tl.maximum(tl.maximum(l0,l1),tl.maximum(l2,l3))
    m=quota_gpu(a,b,other,nh,peak,GAIN_BP)
    ordinal=tl.cumsum(hot.to(tl.int32),0)
    out=tl.where(hot&(ordinal<=m),C,base)
    tl.store(OUT+i,out,valid)


@triton.jit
def dynamic_counts(IDS,COUNTS,N,C:tl.constexpr,W:tl.constexpr,H:tl.constexpr,B:tl.constexpr):
    pid=tl.program_id(0)
    i=pid*B+tl.arange(0,B)
    valid=i<N
    e=tl.load(IDS+i,valid,other=-1)
    base=tl.where(e==C,W,e)
    g=base//32
    tl.store(COUNTS+pid*5+0,tl.sum((valid&(g==0)).to(tl.int32),0))
    tl.store(COUNTS+pid*5+1,tl.sum((valid&(g==1)).to(tl.int32),0))
    tl.store(COUNTS+pid*5+2,tl.sum((valid&(g==2)).to(tl.int32),0))
    tl.store(COUNTS+pid*5+3,tl.sum((valid&(g==3)).to(tl.int32),0))
    tl.store(COUNTS+pid*5+4,tl.sum((valid&(e==H)).to(tl.int32),0))


@triton.jit
def dynamic_large(IDS,COUNTS,OUT,N,NPART,C:tl.constexpr,W:tl.constexpr,H:tl.constexpr,
                  GAIN_BP:tl.constexpr,B:tl.constexpr,P:tl.constexpr):
    pid=tl.program_id(0)
    p=tl.arange(0,P)
    good=p<NPART
    v0=tl.load(COUNTS+p*5+0,good,other=0)
    v1=tl.load(COUNTS+p*5+1,good,other=0)
    v2=tl.load(COUNTS+p*5+2,good,other=0)
    v3=tl.load(COUNTS+p*5+3,good,other=0)
    vh=tl.load(COUNTS+p*5+4,good,other=0)
    l0=tl.sum(v0,0);l1=tl.sum(v1,0);l2=tl.sum(v2,0);l3=tl.sum(v3,0)
    nh=tl.sum(vh,0)
    prior=tl.sum(tl.where(p<pid,vh,0),0)
    a=l0 if H//32==0 else l1 if H//32==1 else l2 if H//32==2 else l3
    b=l0 if C//32==0 else l1 if C//32==1 else l2 if C//32==2 else l3
    o0=l0 if H//32!=0 and C//32!=0 else 0
    o1=l1 if H//32!=1 and C//32!=1 else 0
    o2=l2 if H//32!=2 and C//32!=2 else 0
    o3=l3 if H//32!=3 and C//32!=3 else 0
    other=tl.maximum(tl.maximum(o0,o1),tl.maximum(o2,o3))
    peak=tl.maximum(tl.maximum(l0,l1),tl.maximum(l2,l3))
    m=quota_gpu(a,b,other,nh,peak,GAIN_BP)
    i=pid*B+tl.arange(0,B)
    valid=i<N
    e=tl.load(IDS+i,valid,other=-1)
    base=tl.where(e==C,W,e)
    hot=valid&(e==H)
    ordinal=prior+tl.cumsum(hot.to(tl.int32),0)
    tl.store(OUT+i,tl.where(hot&(ordinal<=m),C,base),valid)


def redirect_dynamic(ids,cfg,gain_bp=100):
    if not ids.numel() or not cfg['enabled']:return ids
    if not ids.is_cuda or not ids.is_contiguous() or ids.ndim!=2:
        raise RuntimeError('Dynamic dispatch requires contiguous gathered CUDA [T,K] IDs')
    n=ids.numel();out=torch.empty_like(ids)
    c,w,h=(int(cfg[k]) for k in ('cold','warm','hot'))
    if n<=4096:
        dynamic_small[(1,)](ids,out,n,c,w,h,gain_bp,triton.next_power_of_2(n),num_warps=4 if n<=1024 else 8)
    else:
        b=256;parts=triton.cdiv(n,b)
        counts=torch.empty((parts,5),device=ids.device,dtype=torch.int32)
        dynamic_counts[(parts,)](ids,counts,n,c,w,h,b,num_warps=4)
        dynamic_large[(parts,)](ids,counts,out,n,parts,c,w,h,gain_bp,b,triton.next_power_of_2(parts),num_warps=4)
    return out
