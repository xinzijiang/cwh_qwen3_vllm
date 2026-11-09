#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
from cwh_v4.policy import read_policy


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--policy',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();cfg=read_policy(a.policy)
    result=dict(definition='same-call routing counterfactuals; NOT GPU execution times',layers=[])
    for l in range(48):
        byphase={p:dict(before=[],after=[],cold=0,hot=0,moved=0,duplicate_rows=0,rows=0) for p in ('decode','prefill')}
        for line in (a.capture/f'trace_layer{l:02d}.jsonl').read_text().split('\n'):
            if not line.strip(): continue
            r=json.loads(line);ids=np.array(r['ids']);out=np.array(r['after']);d=np.array(r['decode'],dtype=bool)
            for phase,sel in [('decode',d),('prefill',~d)]:
                if not sel.any():continue
                z=byphase[phase]
                z['before'].append(np.bincount((ids[sel]//32).ravel(),minlength=4))
                z['after'].append(np.bincount((out[sel]//32).ravel(),minlength=4))
                z['rows']+=int(sel.sum())
                z['duplicate_rows']+=int((np.diff(np.sort(out[sel],axis=1),axis=1)==0).any(1).sum())
                if cfg[l]['enabled']:
                    z['cold']+=int((ids[sel]==cfg[l]['cold']).sum())
                    z['hot']+=int((ids[sel]==cfg[l]['hot']).sum())
                    z['moved']+=int(((ids[sel]==cfg[l]['hot']) & (out[sel]==cfg[l]['cold'])).sum())
        for phase,z in byphase.items():
            for field in ('before','after'):
                v=np.asarray(z[field],dtype=float)
                z[field]=dict(calls=len(v),aggregate_gpu_counts=v.sum(0).tolist(),
                    mean_call_max=float(v.max(1).mean()),mean_call_skew=float((v.max(1)/v.mean(1)).mean())) if len(v) else None
            z['realized_replica_fraction']=z['moved']/z['hot'] if z['hot'] else None
        result['layers'].append(dict(layer=l,enabled=cfg[l]['enabled'],phases=byphase))
    a.out.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
