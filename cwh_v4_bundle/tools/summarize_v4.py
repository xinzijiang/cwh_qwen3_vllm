#!/usr/bin/env python3
"""Partial-progress tables and paired independent-start throughput intervals."""
import argparse,csv,json,math,statistics
from pathlib import Path
from durable import atomic_json,sha

def collect(root):
    rows=[]
    for marker in sorted(root.glob('*/c*/rep_*/*/.done_*.json')):
        suite,conc,rep,arm=marker.relative_to(root).parts[:4]
        which=marker.name[len('.done_'):-len('.json')]
        meta=json.loads(marker.read_text())
        for name,h in meta['files'].items():
            if sha(marker.parent/name)!=h:raise RuntimeError('Corrupt completed result '+str(marker))
        r=json.loads((marker.parent/(which+'.json')).read_text())
        rows.append(dict(suite=suite,concurrency=int(conc[1:]),repeat=int(rep[4:]),arm=arm,test=which,
            tok_s=r.get('output_tokens_per_s'),accuracy_pct=r.get('accuracy_pct'),
            success=r.get('success_count',r.get('questions')),p95_tpot_ms=r.get('p95_tpot_ms'),p95_ttft_ms=r.get('p95_ttft_ms')))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();rows=collect(a.root)
    out=a.root/'summary';out.mkdir(exist_ok=True)
    if rows:
        with (out/'units.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    grouped={};index={(r['suite'],r['concurrency'],r['repeat'],r['arm'],r['test']):r for r in rows}
    for r in rows:
        if r['tok_s'] is None:continue
        base=index.get((r['suite'],r['concurrency'],r['repeat'],'native',r['test']))
        if base:grouped.setdefault((r['suite'],r['concurrency'],r['arm'],r['test']),[]).append(math.log(r['tok_s']/base['tok_s']))
    ratios=[]
    for key,v in grouped.items():
        m=statistics.mean(v);n=len(v);crit=4.30265273 if n==3 else 12.70620474 if n==2 else None
        half=crit*statistics.stdev(v)/math.sqrt(n) if crit else None
        ratios.append(dict(suite=key[0],concurrency=key[1],arm=key[2],test=key[3],paired_repeats=n,
            ratio=math.exp(m),ci95_low=math.exp(m-half) if half is not None else None,ci95_high=math.exp(m+half) if half is not None else None))
    atomic_json(out/'ratios.json',ratios)
    # Pilot gates use only diagnostic performance, never test/QA labels.
    eligible=True;why=[]
    for rep in range(2):
        for test in ('short','long'):
            b=index.get(('pilot',16,rep,'native',test));c=index.get(('pilot',16,rep,'cwh_cost',test))
            if not b or not c:eligible=False;why.append('pilot incomplete');continue
            if c['tok_s']/b['tok_s']<1.02:eligible=False;why.append('pilot throughput gain below 2%')
            if c['p95_tpot_ms']>b['p95_tpot_ms']*1.05:eligible=False;why.append('pilot p95 TPOT regression')
    policy=a.root/'policies/selected.json'
    active=sum(x['enabled'] for x in json.loads(policy.read_text())['layers']) if policy.exists() else None
    if not active:eligible=False;why.append('no active CWH layers')
    atomic_json(out/'pilot_decision.json',dict(candidate='cwh_cost' if eligible else 'native',reasons=sorted(set(why)),active_layers=active,
        note='Performance-only deployment candidate. Full QA and held-out main remain required. Main always reports the unhidden CWH result.'))
    print('Active CWH layers:',active)
    for r in rows:print('{suite:10} c{concurrency:<3} rep{repeat} {arm:16} {test:7} tok/s={tok_s} accuracy={accuracy_pct}'.format(**r))
    print('Saved',out)
if __name__=='__main__':main()
