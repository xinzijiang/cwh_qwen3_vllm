#!/usr/bin/env python3
"""Verify parity of serving flags in completed or running units."""
import argparse,json
from pathlib import Path
KEYS=['--model','--tensor-parallel-size','--all2all-backend','--attention-backend','--dtype','--max-model-len','--gpu-memory-utilization','--seed','--block-size','--num-gpu-blocks-override']
FLAGS=['--enable-expert-parallel','--no-enable-prefix-caching','--no-enable-chunked-prefill','--enforce-eager']
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();reference=None;count=0
for path in sorted(a.root.rglob('server_config.json')):
 d=json.loads(path.read_text());cmd=d['command'];cfg={k:cmd[cmd.index(k)+1] if k in cmd else None for k in KEYS}
 cfg.update({k:k in cmd for k in FLAGS})
 if reference is None:reference=cfg
 elif cfg!=reference:raise RuntimeError('Nonmatching server configuration: '+str(path))
 count+=1
print(json.dumps(dict(configs_checked=count,common=reference,note='Ports, architecture, EPLB settings and sweep concurrency intentionally differ'),indent=2))
