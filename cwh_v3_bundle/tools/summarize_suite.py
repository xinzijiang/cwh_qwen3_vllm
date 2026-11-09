#!/usr/bin/env python3
"""Independent server-start repeats are the experimental unit (NOT requests)."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics as S


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True,help='ONE phase root containing rep_0..rep_2')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();performance={};quality={};csvrows=[]
    for rep in range(3):
        folder=a.root/f'rep_{rep}'
        if not folder.is_dir():raise RuntimeError(f'Missing {folder}')
        for arm in sorted(folder.iterdir()):
            if not arm.is_dir():continue
            if not (arm/'COMPLETE').exists():raise RuntimeError(f'Incomplete arm {arm}')
            for work in ('short','long'):
                r=json.loads((arm/f'{work}.json').read_text())
                if r['error_count'] or r['success_count']!=800 or r['total_output_tokens']!=800*256:
                    raise RuntimeError(f'Invalid formal benchmark: {arm}/{work}')
                performance.setdefault((arm.name,work),{})[rep]=r['output_tokens_per_s']
            quality[rep,arm.name]={int(r['question_index']):r for r in csv.DictReader((arm/'quality.csv').open())}
    for (arm,work),byrep in performance.items():
        if set(byrep)!={0,1,2}:raise RuntimeError(f'Incomplete repeats: {arm}/{work}')
        values=list(byrep.values())
        baseline=performance['native',work]
        lr=[math.log(byrep[i]/baseline[i]) for i in range(3)]
        mean=S.mean(lr);se=S.stdev(lr)/math.sqrt(3);t=4.302652729911275
        csvrows.append(dict(arm=arm,workload=work,mean_tok_s=S.mean(values),sd_tok_s=S.stdev(values),
            paired_geomean_ratio=math.exp(mean),ratio_ci95_low=math.exp(mean-t*se),
            ratio_ci95_high=math.exp(mean+t*se)))
    pairs=[]
    arms=sorted({k[0] for k in performance})
    for rep in range(3):
        for left,right in itertools.combinations(arms,2):
            x,y=quality[rep,left],quality[rep,right]
            if x.keys()!=y.keys() or len(x)!=500:raise RuntimeError('Quality pairing failed')
            if any(x[i]['error'] or y[i]['error'] for i in x):raise RuntimeError('Quality request errors')
            b=sum(int(x[i]['correct']) and not int(y[i]['correct']) for i in x)
            c=sum(int(y[i]['correct']) and not int(x[i]['correct']) for i in x)
            n=b+c
            prob=min(1.,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.
            deltas=[]
            for i in x:
                lx=json.loads(x[i]['option_logprobs']);ly=json.loads(y[i]['option_logprobs'])
                # Compare only identical returned token strings (no token normalization assumptions).
                deltas.extend(abs(lx[k]-ly[k]) for k in lx.keys() & ly.keys())
            pairs.append(dict(repeat=rep,left=left,right=right,
                different_predictions=sum(x[i]['prediction']!=y[i]['prediction'] for i in x),
                left_only_correct=b,right_only_correct=c,exact_mcnemar_p=prob,
                max_option_logprob_delta=max(deltas) if deltas else None))
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    (a.out/'paired_quality.json').write_text(json.dumps(pairs,indent=2))
    (a.out/'interpretation.txt').write_text(
        'Three independent start/measurement blocks; log-ratio t intervals with df=2 are approximate and wide.\n'
        'Do not treat 800 requests as 800 independent speed repetitions.\n'
        'QA is the same 500 questions every repeat, not 1500 different questions.\n'
        'Option-logprob delta is NOT an intermediate-activation equivalence test.\n'
        'EPLB uses four additional slots per layer; custom CWH and Native use 128.\n')
    print(json.dumps(csvrows,indent=2))


if __name__=='__main__':main()
