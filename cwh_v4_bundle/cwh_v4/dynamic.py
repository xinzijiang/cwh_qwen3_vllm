"""Closed-form min-max quota for ONE resident H replica, and pure Python oracle.

No latency optimum is claimed. Loads are integer expert assignments AFTER C->W.
Without the gain guard, this is exact min-max, with the fewest moved assignments
among all peak-minimizing solutions. Resident C/W/H placement remains load-time.
"""


def optimal_quota(loads, source, destination, hot_count, gain_bp=100):
    if source == destination or len(loads) != 4:
        raise ValueError('Need distinct EP4 source/destination')
    if any(int(v) != v or v < 0 for v in loads) or not 0 <= hot_count <= loads[source]:
        raise ValueError('Invalid assignment counts')
    a, b = int(loads[source]), int(loads[destination])
    other = max(int(v) for g,v in enumerate(loads) if g not in (source,destination))
    best_peak = max(other, (a+b+1)//2, a-int(hot_count), b)
    moved = max(0, a-best_peak)
    peak = max(loads)
    gain = peak-best_peak
    if gain < 1 or gain*10000 < int(gain_bp)*peak:
        moved = 0
    return moved


def dynamic_reference(ids, cfg, gain_bp=100):
    if not cfg.get('enabled',True):
        return [list(row) for row in ids]
    c,w,h=(cfg[k] for k in ('cold','warm','hot'))
    out=[[w if e == c else e for e in row] for row in ids]
    loads=[sum(e//32 == g for row in out for e in row) for g in range(4)]
    quota=optimal_quota(loads,h//32,c//32,sum(e==h for row in ids for e in row),gain_bp)
    ordinal=0
    for row,original in zip(out,ids):
        for j,e in enumerate(original):
            if e==h:
                if ordinal<quota:row[j]=c
                ordinal+=1
    return out
