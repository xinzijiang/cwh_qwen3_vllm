#!/usr/bin/env python3
"""Recompute original artifacts without repeating GPU runs. Handles mixed JSON logs."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import re
import statistics as S
import numpy as np


def records(path):
    decoder = json.JSONDecoder()
    for line in path.open(errors='replace'):
        pos = 0
        while True:
            marker = line.find('CWH_POSTREADY_TRACE ', pos)
            if marker < 0:
                break
            start = line.find('{', marker)
            if start < 0:
                break
            try:
                item, used = decoder.raw_decode(line[start:])
            except ValueError:
                pos = start + 1
                continue
            yield item
            pos = start + used


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = dict(performance={}, paired_quality={})
    for mode in ('native_ep', 'eplb4', 'cw_only', 'cwh_full'):
        rows = {}
        for work in ('short', 'long'):
            data = [json.loads(f.read_text()) for f in sorted((a.run / mode).glob(work+'_repeat*.json'))]
            if len(data) != 3 or any(x['error_count'] or x['success_count'] != 800 for x in data):
                raise RuntimeError(f'Incomplete original run: {mode}/{work}')
            xs = [x['output_tokens_per_s'] for x in data]
            rows[work] = dict(repeats=xs, mean=S.mean(xs), sample_sd=S.stdev(xs))
        rows['quality'] = json.loads((a.run / mode / 'accuracy.json').read_text())
        result['performance'][mode] = rows
    for left,right in itertools.combinations(result['performance'], 2):
        def load(mode):
            return {int(r['question_index']):r for r in csv.DictReader((a.run/mode/'accuracy.csv').open())}
        x,y = load(left),load(right)
        if x.keys() != y.keys():
            raise RuntimeError('Unpaired quality records')
        b = sum(int(x[i]['correct']) and not int(y[i]['correct']) for i in x)
        c = sum(int(y[i]['correct']) and not int(x[i]['correct']) for i in x)
        n = b + c
        import math
        prob = min(1., 2 * sum(math.comb(n,k) for k in range(min(b,c)+1)) / 2**n) if n else 1.
        result['paired_quality'][left+' vs '+right] = dict(
            predictions_differ=sum(x[i]['prediction'] != y[i]['prediction'] for i in x),
            left_only_correct=b, right_only_correct=c, exact_mcnemar_p=prob)
    groups = {}
    for x in records(a.run/'cwh_trace/server.log'):
        key = x['layer'], x['call']
        pid = x['pid']
        if pid in groups.setdefault(key, {}):
            raise RuntimeError('Duplicate original trace key')
        groups[key][pid] = x
    differing = []
    samples = []
    for k, v in groups.items():
        signatures = {json.dumps({k:z for k,z in x.items() if k != 'pid'},sort_keys=True) for x in v.values()}
        if len(v) != 4 or len(signatures) != 1:
            differing.append(k)
        samples.append(next(iter(v.values())))
    if differing:
        raise RuntimeError(f'Cannot deduplicate inconsistent ranks: {differing[:5]}')
    trace = dict(samples=len(samples), all_four_rank_records_equal=True,
                 h_to_replica=sum(x['h_to_replica'] for x in samples),
                 cold_redirect=sum(x['c_selected'] for x in samples),
                 duplicates=sum(x['duplicate_token_count'] for x in samples),
                 warning='Old trace has no scheduler phase metadata; row-count groups are proxies only.')
    for name, rows in [('all',samples),('rows_le16',[x for x in samples if x['token_rows']<=16]),
                       ('rows_gt16',[x for x in samples if x['token_rows']>16])]:
        before = np.array([x['before_gpu_assignment_counts'] for x in rows],dtype=float)
        full = np.array([x['after_gpu_assignment_counts'] for x in rows],dtype=float)
        cw = before.copy()
        for i,r in enumerate(rows):
            cw[i,r['cold']//32] -= r['c_selected']
            cw[i,r['warm']//32] += r['c_selected']
        trace[name] = {n:dict(aggregate_gpu_counts=v.sum(0).tolist(),
                            aggregate_skew=float(v.sum(0).max()/v.sum(0).mean()),
                            mean_per_layer_call_max=float(v.max(1).mean()),
                            mean_per_layer_call_skew=float((v.max(1)/v.mean(1)).mean()))
                       for n,v in [('native',before),('cw_counterfactual',cw),('full',full)]}
    result['trace'] = trace
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
