#!/usr/bin/env python3
"""Own decoupled baselines under the SAME C/W quality screens, not paper replicas."""
import argparse
import json
from pathlib import Path
import numpy as np
from durable import atomic_json
from optimize_policy import Replay


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();costs=json.loads(a.pairs.read_text())['layers']
    policies={s:dict(schema=2,selection=s+' decoupled baseline, same quality screens; not a published method reproduction',layers=[])
              for s in ('frequency','quality')}
    for l in range(48):
        train=Replay(a.capture/f'train_layer{l:02d}.jsonl')
        counts=train.counts['decode'].sum(0)
        candidates=[x for x in costs[str(l)] if x['relative_layer_rms_error']<=.10 and x['decode_mass_fraction']<=.01]
        for strategy,pol in policies.items():
            cfg=dict(layer=l,enabled=False,reason='No pair meeting identical quality screens')
            if candidates:
                pair=min(candidates,key=(lambda x:(counts[x['cold']],x['relative_layer_rms_error']))
                         if strategy=='frequency' else (lambda x:x['relative_layer_rms_error']))
                c,w=pair['cold'],pair['warm']
                h=max((e for e in range(128) if e!=w and e//32!=c//32),key=lambda e:counts[e])
                cfg=dict(layer=l,enabled=True,cold=c,warm=w,hot=h,replica_fraction=.5,
                         quality_proxy=pair,reason='Independent choice: C/W first; H by frequency; p=0.5')
            pol['layers'].append(cfg)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    for name,pol in policies.items():atomic_json(a.out_dir/f'{name}.json',pol)


if __name__=='__main__':main()
