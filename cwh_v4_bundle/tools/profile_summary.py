#!/usr/bin/env python3
"""Summed per-file GPU kernel durations. Overlap means sums are not wall time."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();results=[]
    for path in a.root.rglob('*.pt.trace.json*'):
        with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
            obj=json.load(f)
        grouped=defaultdict(lambda:dict(calls=0,sum_gpu_ms=0.))
        for e in obj.get('traceEvents',[]):
            if e.get('cat')!='kernel' or e.get('ph')!='X':continue
            name=e.get('name','');low=name.lower()
            group=('routing_remap' if 'cwh_post_topk_redirect' in low else 'collective' if 'nccl' in low
                   else 'moe_gemm' if any(k in low for k in ('fused_moe','grouped_gemm'))
                   else 'other')
            grouped[group]['calls']+=1;grouped[group]['sum_gpu_ms']+=e.get('dur',0)/1000
        results.append(dict(file=str(path),groups=dict(grouped)))
    if not results:raise RuntimeError('No profiler trace files; inspect profiler output / permissions')
    a.out.write_text(json.dumps(dict(warning='Kernel-name heuristic; inspect raw timelines for GPU/stream overlap and critical path.',
                                  per_file=results),indent=2))


if __name__=='__main__':main()
