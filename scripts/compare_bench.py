#!/usr/bin/env python3
"""Compute fair A/B metrics from same-client, same-workload summary CSVs."""
import argparse,csv,statistics,json
from pathlib import Path

def read(path):
    with open(path,newline='') as f:
        rows=list(csv.DictReader(f))
    return {(int(r['concurrency']),int(r['repeat'])):r for r in rows}

def val(row,k):
    v=row.get(k)
    return float(v) if v not in ('',None,'None') else None

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',required=True)
    p.add_argument('--custom',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args();b=read(a.baseline);c=read(a.custom)
    if set(b)!=set(c):raise SystemExit(f'Mismatched run keys: baseline-only={set(b)-set(c)} custom-only={set(c)-set(b)}')
    rows=[]
    for conc in sorted({x[0] for x in b}):
        keys=[k for k in b if k[0]==conc]
        def avg(d,metric):
            data=[val(d[k],metric) for k in keys]
            if any(x is None for x in data): return None
            return statistics.mean(data)
        rb=avg(b,'aggregate_output_tokens_per_s');rc=avg(c,'aggregate_output_tokens_per_s')
        tb=avg(b,'median_ttft_s');tc=avg(c,'median_ttft_s')
        pb=avg(b,'median_tpot_s_per_token');pc=avg(c,'median_tpot_s_per_token')
        if rb is None or rc is None:raise RuntimeError('Missing throughput')
        row={'concurrency':conc,'repeats':len(keys),
             'base_tok_s':rb,'custom_tok_s':rc,'throughput_speedup':rc/rb,
             'base_median_ttft_ms':tb*1000 if tb else None,
             'custom_median_ttft_ms':tc*1000 if tc else None,
             'base_median_tpot_ms':pb*1000 if pb else None,
             'custom_median_tpot_ms':pc*1000 if pc else None,
             'tpot_speedup':pb/pc if pb and pc else None}
        rows.append(row)
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))
