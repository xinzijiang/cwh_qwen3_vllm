#!/usr/bin/env python3
"""Resumable CWH V4 study. Completed performance units are never time-stitched."""
import argparse,fcntl,json,os,signal,sys,time,uuid
from pathlib import Path
from types import SimpleNamespace
from durable import atomic_json,atomic_copy,sha
import runtime
from runtime import ManagedServer,child,enough,Paused,fresh_port
from check_artifacts import check_capture
PACK=Path(__file__).resolve().parents[1];TOOLS=PACK/'tools';PY=sys.executable
ARMS={
 'native':(None,None,None,'selected'),
 'eplb0':('eplb0',None,None,'selected'),
 'eplb4':('eplb',None,None,'selected'),
 'cwh_cost':('cost','triton',None,'selected'),
 'cwh_cw':('cw','triton',0.,'selected'),
 'cwh_static':('ratio','triton',None,'selected'),
 'cwh_old_dynamic':('dynamic','triton',None,'selected'),
 'cwh_shadow':('cost_shadow','triton',None,'selected'),
 'identity':('identity','triton',None,'selected'),
 'v3_dynamic':('dynamic','triton',None,'v3_count'),
 'v3_static':('ratio','triton',None,'v3_count'),
}
SUITES={
 'pilot':['native','cwh_cost'],
 'main':['native','cwh_cost','eplb0','eplb4'],
 'ablation':['native','identity','cwh_cw','cwh_static','cwh_old_dynamic','cwh_shadow','cwh_cost'],
 'regression':['native','v3_static','v3_dynamic','cwh_cost'],
 'sweep':['native','cwh_cost','cwh_static'],
}

def valid_done(path,fingerprint):
    if not path.exists():return False
    d=json.loads(path.read_text())
    if d['fingerprint']!=fingerprint:raise RuntimeError('Unit fingerprint changed: '+str(path))
    for f,h in d['files'].items():
        if sha(path.parent/f)!=h:raise RuntimeError('Completed result modified: '+str(path.parent/f))
    return True

def publish(folder,prefix,which,fingerprint,attempt,requests,tokens):
    r=json.loads(prefix.with_suffix('.json').read_text())
    if r.get('error_count')!=0:raise RuntimeError('Invalid unit: HTTP errors')
    if which=='quality':
        if r.get('questions')!=500:raise RuntimeError('Expected 500 QA questions')
    elif r.get('success_count')!=requests or r.get('total_output_tokens')!=requests*tokens:
        raise RuntimeError('Performance unit request/token count mismatch')
    for ext in ('.json','.csv'):atomic_copy(prefix.with_suffix(ext),folder/(which+ext))
    files={which+ext:sha(folder/(which+ext)) for ext in ('.json','.csv')}
    atomic_json(folder/('.done_'+which+'.json'),dict(files=files,fingerprint=fingerprint,attempt=str(attempt),completed_epoch=time.time()))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['check','calibrate','pilot','main','focus','ablation','regression','sweep','profile','all'],default='focus')
    p.add_argument('--tag',required=True);p.add_argument('--budget-minutes',type=int,default=230)
    p.add_argument('--model',type=Path,default=Path('/work/qwen3_real/models/Qwen3-30B-A3B'))
    p.add_argument('--source',type=Path,default=Path('/work/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json'))
    p.add_argument('--qa',type=Path,default=Path('/work/openbookqa/validation.jsonl'))
    p.add_argument('--import-v3',type=Path);a=p.parse_args()
    if not a.tag.replace('_','').replace('-','').isalnum():raise ValueError('Simple tag required')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,runtime.interrupted)
    deadline=min(time.time()+a.budget_minutes*60,float(os.environ.get('CWH_SAFE_DEADLINE','inf')))
    root=PACK.parent/'runs'/('cwh_v4_'+a.tag);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'run.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another controller is running this tag')
    stamp=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    session=root/'sessions'/stamp;session.mkdir(parents=True)
    atomic_json(session/'state.json',dict(status='STARTED',job=os.getenv('SLURM_JOB_ID'),stage=a.stage,deadline=deadline))
    try:
        child([PY,TOOLS/'preflight.py','--gpu','--model',a.model,'--qa',a.qa,'--source',a.source,'--out',session/'environment.json'],deadline,session/'preflight.log')
        env=json.loads((session/'environment.json').read_text())
        spec=dict(version=4,model_config=sha(a.model/'config.json'),model_index=sha(a.model/'model.safetensors.index.json'),
            qa=sha(a.qa),source=sha(a.source),image=os.getenv('CWH_IMAGE',''),
            versions={k:env[k] for k in ('python','torch','cuda','vllm','triton','transformers')},
            code={str(f.relative_to(PACK)):sha(f) for f in sorted(PACK.rglob('*')) if f.is_file() and f.suffix in ('.py','.sh','.sbatch')},
            requests=800,output_tokens=256,import_v3=str(a.import_v3) if a.import_v3 else None,
            settings=dict(ep=4,tp=4,dtype='bfloat16',eager=True,kv_blocks=16384,block_size=16,prefix_cache=False,chunked_prefill=False))
        sp=root/'study_spec.json'
        if sp.exists() and json.loads(sp.read_text())!=spec:raise RuntimeError('Study code/data/environment changed; preserve this study and use a new tag')
        if not sp.exists():atomic_json(sp,spec)
        studyhash=sha(sp)
        child([PY,TOOLS/'gpu_check.py'],deadline,session/'gpu_check.log')
        child([PY,TOOLS/'gpu_check_cost.py'],deadline,session/'gpu_cost_check.log')
        if a.stage=='check':
            atomic_json(session/'state.json',dict(status='STAGE_COMPLETE',stage='check'));print('CHECK_PASSED');return
        work=root/'workloads';policies=root/'policies';policies.mkdir(exist_ok=True)
        if not (work/'manifest.json').exists():
            if work.exists():work.rename(root/('incomplete_workloads_'+stamp))
            child([PY,TOOLS/'prepare_splits.py','--source',a.source,'--model',a.model,'--out',work],deadline,session/'prepare.log')
        manifest=json.loads((work/'manifest.json').read_text())
        for name,meta in manifest.items():
            if sha(work/name)!=meta['sha256']:raise RuntimeError('Workload modified: '+name)
        # Preserve the original split seed so imported V3 train does not leak into V4 test.
        if a.import_v3:
            old_manifest=json.loads((a.import_v3/'workloads/manifest.json').read_text())
            if manifest!=old_manifest:raise RuntimeError('Imported calibration requires identical frozen split manifest')
        def fingerprint(value):
            import hashlib
            return hashlib.sha256((studyhash+sha(work/'manifest.json')+value).encode()).hexdigest()
        legacy=PACK/'legacy/cwh_policy_48layers.json'
        def unit(suite,rep,name,which,conc=16,calibration=False):
            folder=(root/'calibration') if calibration else root/suite/f'c{conc}'/f'rep_{rep}'/name
            folder.mkdir(parents=True,exist_ok=True)
            mode,backend,fraction,pname=('probe','torch',None,None) if calibration else ARMS[name]
            policy=legacy if calibration else policies/(pname+'.json')
            fp=fingerprint(f'{suite}/{conc}/{rep}/{name}/{which}/'+sha(policy))
            done=folder/('.done_'+which+'.json')
            if valid_done(done,fp):print('SKIP_COMPLETE',suite,conc,rep,name,which,flush=True);return
            n=800;tokens=256
            for old in sorted((folder/'attempts').glob(which+'_*'),reverse=True):
                pref=old/name/which;meta=old/'state.json'
                if not meta.exists() or not all(pref.with_suffix(e).exists() for e in ('.json','.csv')):continue
                if json.loads(meta.read_text()).get('fingerprint')!=fp:continue
                r=json.loads(pref.with_suffix('.json').read_text())
                if r.get('error_count') or (r.get('questions')!=500 if which=='quality' else r.get('success_count')!=800 or r.get('total_output_tokens')!=204800):continue
                if calibration:check_capture(old/name/'capture')
                publish(folder,pref,which,fp,old,n,tokens);print('RECOVER_COMPLETE',name,which,flush=True);return
            enough(deadline,900 if which!='quality' else 420)
            attempt=folder/'attempts'/(which+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);attempt.mkdir(parents=True)
            atomic_json(attempt/'state.json',dict(status='RUNNING',fingerprint=fp,session=stamp))
            params=SimpleNamespace(out=attempt,model=a.model,port=fresh_port(),legacy_policy=legacy,concurrency=conc,deadline=deadline)
            cap=attempt/name/'capture' if calibration else None
            server=ManagedServer(params,name,mode,backend,fraction,policy=policy,capture=cap)
            try:
                with server:
                    pref=server.target/which
                    url=f'http://127.0.0.1:{params.port}'
                    if which=='quality':
                        # Attempt-local journal: restarting partial QA avoids mixing changed
                        # batch composition for dynamic routing or EPLB state.
                        cmd=[PY,TOOLS/'evaluate_openbookqa.py','--dataset',a.qa,'--base-url',url,'--mode',name,'--output-prefix',pref,'--journal',server.target/'quality.questions.jsonl']
                    else:
                        split='diagnostic' if suite in ('pilot','ablation','regression','profile') else 'test'
                        workload=work/(which+'_mixed.jsonl') if calibration else work/(split+'_'+which+'.jsonl')
                        cmd=[PY,TOOLS/'benchmark_stream.py','--base-url',url,'--workload',workload,'--mode',name,'--repeat',rep,'--concurrency',conc,'--output-tokens',tokens,'--warmup',max(32,conc*2),'--output-prefix',pref]
                        if calibration:cmd+=['--trace-marker',server.env['CWH_CAPTURE_MARKER'],'--marker-value',which]
                        if not calibration:cmd+=['--warmup-rounds','8','--warmup-tokens','256']
                    child(cmd,deadline,attempt/'client.log')
                    # Snapshot metrics outside measured interval. Availability is optional.
                    try:
                        import urllib.request
                        with urllib.request.urlopen(url+'/metrics',timeout=10) as res:(server.target/'metrics_after.txt').write_bytes(res.read())
                    except Exception as e:(server.target/'metrics_unavailable.txt').write_text(str(e))
                if calibration:
                    check_capture(cap)
                    for artifact in cap.iterdir():
                        if artifact.is_file():
                            with artifact.open('rb') as stream:os.fsync(stream.fileno())
                publish(folder,pref,which,fp,attempt,n,tokens)
                atomic_json(attempt/'state.json',dict(status='COMPLETE',fingerprint=fp,session=stamp))
            except BaseException as e:
                atomic_json(attempt/'state.json',dict(status='PAUSED' if isinstance(e,Paused) else 'FAILED',reason=str(e),fingerprint=fp));raise
        calibration=root/'calibration';calibration.mkdir(exist_ok=True)
        if a.stage in ('calibrate','focus','all') or not (calibration/'COMPLETE.json').exists():
            receipt=calibration/'import_receipt.json'
            if a.import_v3:
                if not receipt.exists():
                    from import_calibration import import_v3
                    cap=import_v3(a.import_v3,root,a.model,a.qa,a.source)
                else:
                    for f,h in json.loads(receipt.read_text())['files'].items():
                        if sha(root/f)!=h:raise RuntimeError('Imported calibration modified: '+f)
                    cap=calibration/'imported'
            else:
                for split in ('train','validation'):unit('calibration',0,'native_probe',split,calibration=True)
                cap=calibration/'merged_capture';cap.mkdir(exist_ok=True)
                for split in ('train','validation'):
                    d=json.loads((calibration/('.done_'+split+'.json')).read_text());src=Path(d['attempt'])/'native_probe/capture'
                    for f in src.iterdir():
                        if f.name.startswith(split+'_layer') or split=='train' and f.name.startswith('hidden_layer'):
                            dest=cap/f.name
                            if not dest.exists():dest.symlink_to(os.path.relpath(f,cap))
            pairs=calibration/'pair_costs.json';cost=calibration/'hardware_cost.json'
            child([PY,TOOLS/'quality_pairs.py','--capture',cap,'--model',a.model,'--out',pairs,'--deadline',deadline],deadline,session/'pairs.log')
            child([PY,TOOLS/'calibrate_cost.py','--capture',cap,'--model',a.model,'--out',cost,'--deadline',deadline],deadline,session/'cost.log')
            if not (policies/'candidate.json').exists():
                child([PY,TOOLS/'optimize_cost.py','--capture',cap,'--pairs',pairs,'--cost',cost,'--out',policies/'candidate.json','--deadline',deadline],deadline,session/'optimize.log')
            if not (policies/'selected.json').exists():
                child([PY,TOOLS/'validate_policy_gpu.py','--capture',cap,'--model',a.model,'--candidate',policies/'candidate.json','--out',policies/'selected.json','--deadline',deadline],deadline,session/'replay.log')
            if not (policies/'v3_count.json').exists():
                child([PY,TOOLS/'optimize_policy.py','--capture',cap,'--pairs',pairs,'--out',policies/'v3_count.json','--dispatch','dynamic','--deadline',deadline],deadline,session/'v3_planner.log')
            atomic_json(calibration/'COMPLETE.json',dict(files={str(f.relative_to(root)):sha(f) for f in (pairs,cost,policies/'candidate.json',policies/'selected.json',policies/'v3_count.json')}))
        for f,h in json.loads((calibration/'COMPLETE.json').read_text())['files'].items():
            if sha(root/f)!=h:raise RuntimeError('Calibration/policy changed')
        suites=['pilot','main'] if a.stage=='focus' else ['pilot','main','ablation','regression','sweep'] if a.stage=='all' else [a.stage] if a.stage in SUITES else []
        for suite in suites:
            modes=SUITES[suite]
            for conc in ([1,4,16,32,64] if suite=='sweep' else [16]):
                for rep in range(2 if suite=='pilot' else 3):
                    ordered=modes[rep%len(modes):]+modes[:rep%len(modes)]
                    if rep%2:ordered=list(reversed(ordered))
                    for name in ordered:
                        for which in (('short','long') if rep%2==0 else ('long','short')):
                            unit(suite,rep,name,which,conc)
                        if suite!='pilot':
                            unit(suite,rep,name,'quality',conc)
                            atomic_json(root/suite/f'c{conc}'/f'rep_{rep}'/name/'COMPLETE',dict(complete=True))
        if a.stage=='profile':
            from profile_workflow import run_profiles
            run_profiles(root,a,deadline,policies,work)
        atomic_json(session/'state.json',dict(status='STAGE_COMPLETE',stage=a.stage,completed_epoch=time.time()))
        print('STAGE_COMPLETE',a.stage,root,flush=True)
    except Paused as e:
        atomic_json(session/'state.json',dict(status='PAUSED',reason=str(e),completed_epoch=time.time()))
        print('PAUSED_SAFELY',e,'Repeat SAME tag and stage in another allocation.',flush=True)
    except BaseException as e:
        atomic_json(session/'state.json',dict(status='FAILED',reason=str(e),completed_epoch=time.time()));raise
    finally:lock.close()
if __name__=='__main__':main()
