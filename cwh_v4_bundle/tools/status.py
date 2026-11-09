#!/usr/bin/env python3
"""Python 3.6 compatible, read-only status command for login nodes."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
for f in sorted((a.root/'sessions').glob('*/state.json'))[-4:]:
 print('SESSION',f.parent.name,json.loads(f.read_text()))
policy=a.root/'policies/selected.json'
if policy.exists():
 obj=json.loads(policy.read_text());print('CWH ACTIVE LAYERS',sum(r['enabled'] for r in obj['layers']),'/48')
print('COMPLETED UNITS')
for f in sorted(a.root.rglob('.done_*.json')):print(f.relative_to(a.root))
print('TOTAL',len(list(a.root.rglob('.done_*.json'))))
