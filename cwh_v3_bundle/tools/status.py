#!/usr/bin/env python3
"""Read-only status, host Python standard library only; no GPUs needed."""
import argparse,json
from pathlib import Path
PACK=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
root=PACK.parent/'runs'/('cwh_v3_'+a.tag)
print('STUDY',root)
if not root.exists():raise SystemExit('Not started')
for phase,total in [('calibration',2),('formal',81),('diagnostic',90)]:
    done=list((root/phase).rglob('.done_*.json'))
    print(phase,len(done),'/',total,'completed units')
cost=root/'calibration/pair_costs.json'
if cost.exists():print('quality layers:',len(json.loads(cost.read_text())['layers']),'/ 48')
selected=root/'policies/selected.json';partial=root/'policies/selected.partial.json'
if selected.exists():
    rows=json.loads(selected.read_text())['layers'];print('selected layers:',sum(x['enabled'] for x in rows),'/ 48')
elif partial.exists():print('optimizer layers:',len(json.loads(partial.read_text())['layers']),'/ 48')
for path in sorted((root/'sessions').glob('*/state.json'))[-3:]:
    d=json.loads(path.read_text());print('SESSION',path.parent.name,d.get('status'),d.get('reason',''))
print('Incomplete attempt logs and request journals are retained under attempts/.')
