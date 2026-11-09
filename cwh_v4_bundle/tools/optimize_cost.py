#!/usr/bin/env python3
"""Train-only C/W/H search, one-candidate validation, bounded layer/quality budget."""
import argparse,json,time
from pathlib import Path
import numpy as np
from durable import atomic_json,sha
from cwh_v4.cost import bucket_for,choose_move,loads,profile_bucket
from cwh_v4.policy import reference_ids, salt_for
from optimize_policy import masks

def read_records(path,limit=96):
    rows=[json.loads(line) for line in path.open() if line.strip()]
    out=[]
    for decode in (True,False):
        eligible=[r for r in rows if bool(all(r['decode']))==decode]
        if not eligible:continue
        for i in np.linspace(0,len(eligible)-1,min(limit,len(eligible)),dtype=int):
            r=eligible[i];ids=r['ids']+r.get('padding_ids',[])
            out.append(dict(ids=ids,counts=np.bincount(np.array(ids).ravel(),minlength=128).tolist(),
                            call=r['call'],pure_decode=decode,
                            half_counts=np.bincount(np.array(ids)[masks(len(ids),salt_for(r['layer'],r['call']))[1]].ravel(),minlength=128).tolist()))
    return out

def evaluate(records,cfg,model,static=False):
    stats={'decode':[],'prefill_mixed':[]};moves=0
    for r in records:
        n=len(r['ids']);b=str(profile_bucket(n))
        if b not in model['buckets']:return None
        prof=model['buckets'][b];local=dict(cfg,**{k:v for k,v in prof.items() if k.startswith('cost_')})
        if not prof['usable']:
            return None
        before=max(loads(r['counts'],local))+prof['intercept_ns']
        if n<=512 and not static:
            m=choose_move(r['counts'],local);after=r['counts'].copy()
            after[cfg['warm']]+=after[cfg['cold']];after[cfg['cold']]=m;after[cfg['hot']]-=m
            overhead=prof['route_ns']['cost']
        else:
            fraction=cfg['large_fraction'];after=r['counts'].copy()
            m=0 if fraction==0 else r['half_counts'][cfg['hot']] if fraction==.5 else after[cfg['hot']]
            after[cfg['warm']]+=after[cfg['cold']];after[cfg['cold']]=m;after[cfg['hot']]-=m
            overhead=prof['route_ns']['ratio']
        final=max(loads(after,local))+prof['intercept_ns']+overhead
        stats['decode' if r['pure_decode'] else 'prefill_mixed'].append((before,final))
        moves+=m
    out={'moved':int(moves)}
    for phase,items in stats.items():
        if not items:return None
        x=np.array(items,dtype=float);out[phase+'_gain']=float(1-x[:,1].sum()/x[:,0].sum())
        out[phase+'_p95_regression']=float(np.percentile(x[:,1]/x[:,0]-1,95))
        out[phase+'_saved_ns']=float(np.mean(x[:,0]-x[:,1]))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--pairs',type=Path,required=True);p.add_argument('--cost',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--deadline',type=float,default=0)
    p.add_argument('--max-layers',type=int,default=8);p.add_argument('--max-pair-error',type=float,default=.05)
    p.add_argument('--error-budget',type=float,default=.15);p.add_argument('--min-gain',type=float,default=.03)
    a=p.parse_args();cost=json.loads(a.cost.read_text());pairs=json.loads(a.pairs.read_text())['layers']
    if not cost.get('complete'):raise RuntimeError('Cost calibration incomplete')
    partial=a.out.with_suffix('.partial.json')
    state=json.loads(partial.read_text()) if partial.exists() else dict(layers=[],cost_sha=sha(a.cost),pairs_sha=sha(a.pairs))
    if state['cost_sha']!=sha(a.cost) or state['pairs_sha']!=sha(a.pairs):raise RuntimeError('Planner input changed')
    for l in range(len(state['layers']),48):
        if a.deadline and time.time()+180>a.deadline:raise SystemExit(75)
        train=read_records(a.capture/f'train_layer{l:02d}.jsonl')
        valid=read_records(a.capture/f'validation_layer{l:02d}.jsonl')
        counts=np.sum([r['counts'] for r in train if r['pure_decode']],axis=0)
        hot=[g*32+int(np.argmax(counts[g*32:(g+1)*32])) for g in range(4)]
        eligible=[x for x in pairs[str(l)] if x['relative_layer_rms_error']<=a.max_pair_error and x['decode_mass_fraction']<=.01]
        # First shortlist by semantic error; compare one winner on validation.
        eligible=sorted(eligible,key=lambda x:x['relative_layer_rms_error'])[:12]
        best=None
        profiles={k:{kk:vv for kk,vv in v.items() if kk in ('cost_tile','cost_coeff_ns','usable')} for k,v in cost['buckets'].items() if int(k)<=512}
        for pair in eligible:
            c,w=pair['cold'],pair['warm']
            for h in hot:
                if h in (c,w) or h//32==c//32:continue
                for fraction in (0.,.5,1.):
                    cfg=dict(layer=l,enabled=True,cold=c,warm=w,hot=h,replica_fraction=fraction,
                             large_fraction=fraction,cost_buckets=profiles,online_gain_bp=100,
                             online_margin_ns=0,quality_proxy=pair)
                    metric=evaluate(train,cfg,cost)
                    if metric is None or metric['prefill_mixed_gain']<-.01 or metric['decode_p95_regression']>.05:continue
                    if best is None or metric['decode_saved_ns']>best['train_cost']['decode_saved_ns']:
                        best=dict(cfg,train_cost=metric)
        selected=dict(layer=l,enabled=False,reason='no cost/quality-admissible candidate')
        if best is not None:
            static_options=[]
            for fraction in (0.,.5,1.):
                probe=dict(best,large_fraction=fraction)
                metric=evaluate(train,probe,cost,static=True)
                if metric is not None:static_options.append((metric['decode_gain'],fraction))
            best['replica_fraction']=max(static_options)[1] if static_options else 0.
            metric=evaluate(valid,best,cost)
            ok=bool(metric and min(best['train_cost']['decode_gain'],metric['decode_gain'])>=a.min_gain and metric['prefill_mixed_gain']>=-.01 and metric['decode_p95_regression']<=.05)
            selected=dict(best,enabled=ok,validation_cost=metric,reason='accepted proxy' if ok else 'validation rejected')
        state['layers'].append(selected);atomic_json(partial,state)
        print('COST_SELECT',l,selected['enabled'],selected.get('validation_cost'),flush=True)
    # Bound cumulative approximation risk; sum of layer proxies is a heuristic,
    # not a theorem about end-to-end error (hidden states change downstream).
    accepted=sorted((r for r in state['layers'] if r['enabled']),key=lambda r:r['validation_cost']['decode_saved_ns'],reverse=True)
    keep=set();spent=0.
    for r in accepted:
        error=r['quality_proxy']['relative_layer_rms_error']
        if len(keep)<a.max_layers and spent+error<=a.error_budget:
            keep.add(r['layer']);spent+=error
    for r in state['layers']:
        if r['enabled'] and r['layer'] not in keep:r.update(enabled=False,reason='layer/error budget')
    state.update(schema=4,selection='train search; validation veto; layer and proxy error budget',
                 active_layers=len(keep),error_proxy_sum=spent,not_an_end_to_end_guarantee=True,
                 thresholds=vars(a),complete=True)
    atomic_json(a.out,state)
if __name__=='__main__':main()
