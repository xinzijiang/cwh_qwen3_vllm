"""Integer cost proxy shared by the offline planner and CUDA routing oracle."""
import math

FRACTIONS = (0, 1, 2, 3, 4)
MAX_DYNAMIC_ROWS = 512

def bucket_for(rows):
    return min(512, 1 << max(0, (int(rows) - 1).bit_length()))

def profile_bucket(rows):
    return 1 << max(0, (int(rows) - 1).bit_length())

def features(counts, tile=16):
    return [sum(n > 0 for n in counts), sum((int(n)+tile-1)//tile for n in counts), sum(counts)]

def gpu_cost(counts, coeff, tile=16):
    return sum(a*b for a,b in zip(features(counts,tile),coeff))

def loads(counts, cfg):
    return [gpu_cost(counts[g*32:(g+1)*32], cfg['cost_coeff_ns'], cfg['cost_tile']) for g in range(4)]

def choose_move(counts, cfg):
    """Compare five quotas including whole-H relocation; ties keep fewer moves.

    Exact minimum on this five-point grid, not a global makespan optimum.
    Input counts are original logical expert counts, including padding.
    """
    c,w,h=(cfg[k] for k in ('cold','warm','hot'))
    after=list(map(int,counts)); after[w]+=after[c]; after[c]=0
    nh=after[h]; baseline=max(loads(after,cfg)); best=baseline; quota=0
    for numerator in FRACTIONS[1:]:
        m=(nh*numerator)//4
        moved=after.copy();moved[h]-=m;moved[c]+=m
        value=max(loads(moved,cfg))
        if value<best:best=value;quota=m
    margin=max(int(cfg.get('online_margin_ns',0)), (baseline*int(cfg.get('online_gain_bp',100))+9999)//10000)
    if baseline-best<=margin:quota=0
    return quota

def reference(ids,cfg,shadow=False):
    counts=[0]*128
    for row in ids:
        for e in row:counts[e]+=1
    quota=choose_move(counts,cfg)
    out=[];ordinal=0
    c,w,h=(cfg[k] for k in ('cold','warm','hot'))
    for row in ids:
        dst=[]
        for e in row:
            if e==h:ordinal+=1
            dst.append(w if e==c else c if e==h and ordinal<=quota and not shadow else e)
        out.append(dst)
    return out,quota
