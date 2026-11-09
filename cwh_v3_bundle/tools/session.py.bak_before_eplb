#!/usr/bin/env python3
"""4-hour-friendly durable workflow. Rerun SAME stage/tag to continue.

Each timed 800-request unit has its OWN fresh server so an interrupted neighbour
cannot change EPLB state/warm history. Incomplete speed units are retained but
never stitched into throughput; quality resumes individual completed questions.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace
from durable import atomic_json,atomic_copy,sha
from run_suite import Server,DIAGNOSTIC

PACK=Path(__file__).resolve().parents[1];TOOLS=PACK/'tools';PY=sys.executable
FORMAL=[('native',None,None,None),('eplb_default','eplb',None,None),
        ('fixed_legacy_fused','legacy','triton',None),
        ('frequency_cwh','ratio','triton',None),('quality_cwh','ratio','triton',None),
        ('selective_cw','cw','triton',0),('selective_static','ratio','triton',None),
        ('selective_dynamic','dynamic','triton',None),('selective_static_torch','ratio','torch',None)]
CORE={'native','selective_cw','selective_static','selective_dynamic'}
STOP=False


class Paused(Exception):pass


class DurableServer(Server):
    def __enter__(self):
        probe=socket.socket()
        try:probe.bind(('127.0.0.1',self.a.port))
        finally:probe.close()
        self.log=(self.target/'server.log').open('w')
        atomic_json(self.target/'server_config.json',dict(command=self.cmd,
                    environment={k:v for k,v in self.env.items() if k.startswith(('CWH_','VLLM_'))}))
        self.p=subprocess.Popen(self.cmd,env=self.env,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            child([PY,TOOLS/'wait_ready.py','--url',f'http://127.0.0.1:{self.a.port}',
                   '--pid',self.p.pid,'--seconds','1800'],self.a.deadline,self.target/'readiness.log')
            if self.env['CWH_MODE'] not in ('v1','eplb') and '--hf-overrides' in self.cmd:
                child([PY,TOOLS/'check_artifacts.py','weights',self.target/'weight_audits'],
                      self.a.deadline,self.target/'weight_check.log')
            return self
        except BaseException:
            self.__exit__(None,None,None);raise


def interrupted(signum,frame):
    global STOP
    STOP=True


def child(command,deadline,log):
    """Single child, own process group, durable logs; deadlines also cover startup."""
    print('RUN',*map(str,command),flush=True)
    with Path(log).open('a') as f:
        p=subprocess.Popen(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                if STOP or time.time()>=deadline:
                    raise Paused('User stop / safe session deadline')
                time.sleep(.5)
            if p.returncode==75:raise Paused('Saved at a layer boundary')
            if p.returncode:raise RuntimeError(f'Child exit {p.returncode}; inspect {log}')
        finally:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL);p.wait()


def enough(deadline,estimate):
    if STOP or time.time()+estimate+90>=deadline:
        raise Paused(f'Not starting next unit: reserve about {estimate//60} minutes plus shutdown time')


def check_letters(model):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    for token,letter in zip([362,425,356,422],'ABCD'):
        if tok.decode([token]).strip()!=letter:raise RuntimeError('QA token IDs do not match this tokenizer')


def valid_done(path,fingerprint):
    if not path.exists():return False
    d=json.loads(path.read_text())
    if d['fingerprint']!=fingerprint:raise RuntimeError('Checkpoint belongs to different code/data/policy')
    for f,h in d.get('files',{}).items():
        if not (path.parent/f).is_file() or sha(path.parent/f)!=h:
            raise RuntimeError(f'Completed result missing/modified: {path.parent/f}')
    return True


def publish(folder,prefix,kind,fingerprint,attempt,elapsed=None):
    summary=json.loads(prefix.with_suffix('.json').read_text())
    if summary.get('error_count')!=0:raise RuntimeError('Failed requests: this unit is invalid')
    if kind=='performance':
        if summary['success_count']!=800 or summary['total_output_tokens']!=800*256:
            raise RuntimeError('Speed unit must contain exactly 800x256 successful output tokens')
    elif summary.get('questions')!=500:raise RuntimeError('QA unit must contain 500 questions')
    name='quality' if kind=='quality' else prefix.stem
    # Prefix stem is short/long/train/validation here.
    for ext in ('.json','.csv'):atomic_copy(prefix.with_suffix(ext),folder/(name+ext))
    files={name+ext:sha(folder/(name+ext)) for ext in ('.json','.csv')}
    atomic_json(folder/('.done_'+name+'.json'),dict(fingerprint=fingerprint,files=files,
                attempt=str(attempt),completed_epoch=time.time(),kind=kind,unit_elapsed_s=elapsed))


def merge_capture(root):
    merged=root/'calibration/merged_capture';merged.mkdir(parents=True,exist_ok=True)
    for split in ('train','validation'):
        d=json.loads((root/'calibration'/('.done_'+split+'.json')).read_text())
        cap=Path(d['attempt'])/'native_probe/capture'
        # Done paths are inside /work; links use relative paths and work on host too.
        for path in cap.iterdir():
            if path.name.startswith(split+'_layer') or split=='train' and path.name.startswith('hidden_layer'):
                link=merged/path.name
                if not link.exists():link.symlink_to(os.path.relpath(path,merged))
    return merged


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['calibrate','core','focus','formal','diagnostic','verify','profile','all'],default='focus')
    p.add_argument('--tag',required=True)
    p.add_argument('--budget-minutes',type=int,default=220)
    p.add_argument('--repeat',type=int,choices=[0,1,2])
    p.add_argument('--model',type=Path,default=Path('/work/qwen3_real/models/Qwen3-30B-A3B'))
    p.add_argument('--source',type=Path,default=Path('/work/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json'))
    p.add_argument('--qa',type=Path,default=Path('/work/openbookqa/validation.jsonl'))
    p.add_argument('--port',type=int,default=8765)
    a=p.parse_args()
    if not a.tag.replace('_','').replace('-','').isalnum():raise ValueError('Simple tag required')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
    deadline=min(time.time()+a.budget_minutes*60,float(os.getenv('CWH_SAFE_DEADLINE','inf')))
    root=PACK.parent/'runs'/('cwh_v3_'+a.tag);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'run.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Same study already running; do not run two sessions with the same tag')
    stamp=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    session=root/'sessions'/stamp;session.mkdir(parents=True)
    atomic_json(session/'state.json',dict(status='STARTED',job=os.getenv('SLURM_JOB_ID'),deadline=deadline))
    try:
        spec=dict(version=3,model=str(a.model),model_index=sha(a.model/'model.safetensors.index.json'),
                  model_config=sha(a.model/'config.json'),qa=sha(a.qa),source=sha(a.source),
                  code={str(f.relative_to(PACK)):sha(f) for f in sorted(PACK.rglob('*.py'))},
                  concurrency=16,output_tokens=256,requests=800)
        specpath=root/'study_spec.json'
        if specpath.exists() and json.loads(specpath.read_text())!=spec:
            raise RuntimeError('Study code/model/data changed. Keep old study and use a NEW tag.')
        if not specpath.exists():atomic_json(specpath,spec)
        studyhash=sha(specpath)
        enough(deadline,300)
        import torch,vllm
        if vllm.__version__!='0.13.0' or torch.cuda.device_count()!=4:
            raise RuntimeError('Use original vLLM 0.13.0 container and four GPUs')
        gpus=[dict(name=torch.cuda.get_device_name(i),bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(4)]
        if any('A100' not in x['name'] or x['bytes']<75*1024**3 for x in gpus):
            raise RuntimeError('Require four A100-80GB GPUs')
        check_letters(a.model)
        atomic_json(session/'hardware.json',dict(gpus=gpus,torch=torch.__version__,vllm=vllm.__version__,
                    node=socket.gethostname(),job=os.getenv('SLURM_JOB_ID')))
        child([PY,TOOLS/'gpu_check.py'],deadline,session/'cuda_correctness.log')
        work=root/'workloads'
        if not (work/'manifest.json').exists():
            enough(deadline,600)
            # An interrupted prepare has no accepted manifest and is archived.
            if work.exists():work.rename(root/('workloads_incomplete_'+stamp))
            child([PY,TOOLS/'prepare_splits.py','--source',a.source,'--model',a.model,'--out',work],deadline,session/'prepare.log')
        manifest=json.loads((work/'manifest.json').read_text())
        for name,md in manifest.items():
            if sha(work/name)!=md['sha256']:raise RuntimeError('Frozen workload modified: '+name)
        wh=sha(work/'manifest.json')
        policies=root/'policies';policies.mkdir(exist_ok=True)
        legacy=PACK/'legacy/cwh_policy_48layers.json'

        def fp(unit,policy=None):
            import hashlib
            return hashlib.sha256((studyhash+wh+unit+(sha(policy) if policy else '')).encode()).hexdigest()

        def unit(phase,rep,name,mode,backend,fraction,which,policy,split=None):
            iscal=phase=='calibration'
            folder=root/'calibration' if iscal else root/phase/f'rep_{rep}'/name
            folder.mkdir(parents=True,exist_ok=True)
            done=folder/('.done_'+which+'.json')
            fingerprint=fp(f'{phase}/{rep}/{name}/{which}',policy)
            if valid_done(done,fingerprint):
                print('SKIP_COMPLETE',phase,rep,name,which,flush=True);return
            # Recover a result saved just before its completion marker was lost.
            for old in sorted((folder/'attempts').glob(which+'_*'),reverse=True):
                meta=old/'state.json';prefix=old/name/which
                if not meta.exists() or not prefix.with_suffix('.json').exists() or not prefix.with_suffix('.csv').exists():continue
                if json.loads(meta.read_text()).get('fingerprint')!=fingerprint:continue
                r=json.loads(prefix.with_suffix('.json').read_text())
                good=(r.get('error_count')==0 and
                      (r.get('questions')==500 if which=='quality' else
                       r.get('success_count')==800 and r.get('total_output_tokens')==800*256))
                if not good:continue
                if iscal:
                    from check_artifacts import check_capture
                    check_capture(old/name/'capture')
                publish(folder,prefix,'quality' if which=='quality' else 'performance',fingerprint,old)
                print('RECOVER_FINISHED_CLIENT',phase,rep,name,which,flush=True);return
            estimate=2700 if iscal else 1800 if which in ('short','long') else 600
            if not iscal:
                prior=[]
                for f in root.glob('formal/rep_*/*/.done_*.json'):
                    r=json.loads(f.read_text())
                    if r.get('kind')==('quality' if which=='quality' else 'performance') and r.get('unit_elapsed_s'):
                        prior.append(r['unit_elapsed_s'])
                if prior:
                    import statistics
                    estimate=max(600 if which=='quality' else 900,120+1.4*statistics.median(prior))
            enough(deadline,estimate)
            attempt=folder/'attempts'/(which+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
            attempt.mkdir(parents=True)
            started=time.time()
            atomic_json(attempt/'state.json',dict(status='RUNNING',fingerprint=fingerprint,session=stamp,started_epoch=started))
            params=SimpleNamespace(out=attempt,model=a.model,port=a.port,legacy_policy=legacy,concurrency=16,deadline=deadline)
            cap=attempt/name/'capture' if iscal else None
            server=DurableServer(params,name,mode,backend,fraction,policy=policy,capture=cap)
            try:
                # Server readiness itself has a bounded timeout; outer budget reserves
                # the full 30-minute startup bound before timed units are attempted.
                with server:
                    if STOP or time.time()>=deadline:raise Paused('Stop after model readiness')
                    prefix=server.target/which
                    if which=='quality':
                        command=[PY,TOOLS/'evaluate_openbookqa.py','--dataset',a.qa,'--base-url',f'http://127.0.0.1:{a.port}',
                                 '--mode',name,'--output-prefix',prefix,'--journal',folder/'quality.questions.jsonl']
                    else:
                        workload=work/(f'{which}_mixed.jsonl' if iscal else f'{split}_{which}.jsonl')
                        command=[PY,TOOLS/'benchmark_stream.py','--base-url',f'http://127.0.0.1:{a.port}',
                            '--workload',workload,'--mode',name,'--repeat',str(rep),'--concurrency','16',
                            '--output-tokens','256','--warmup','16','--output-prefix',prefix]
                        if iscal:command+=['--trace-marker',server.env['CWH_CAPTURE_MARKER'],'--marker-value',which]
                    child(command,deadline,attempt/'client.log')
                if iscal:
                    child([PY,TOOLS/'check_artifacts.py','capture',cap],deadline,attempt/'capture_check.log')
                    # Ensure completed calibration files are persisted before committing.
                    for f in cap.iterdir():
                        if f.is_file():
                            fd=os.open(f,os.O_RDONLY)
                            try:os.fsync(fd)
                            finally:os.close(fd)
                publish(folder,prefix,'quality' if which=='quality' else 'performance',fingerprint,attempt,time.time()-started)
                atomic_json(attempt/'state.json',dict(status='COMPLETE',fingerprint=fingerprint,session=stamp))
            except BaseException as exc:
                atomic_json(attempt/'state.json',dict(status='INTERRUPTED' if isinstance(exc,Paused) else 'FAILED',
                            fingerprint=fingerprint,session=stamp,reason=str(exc)))
                raise

        if a.stage in ('calibrate','focus','all'):
            for split in ('train','validation'):
                unit('calibration',0,'native_probe','probe','torch',None,split,legacy)
            cap=merge_capture(root)
            enough(deadline,420)
            child([PY,TOOLS/'quality_pairs.py','--capture',cap,'--model',a.model,'--out',root/'calibration/pair_costs.json',
                   '--deadline',deadline],deadline,session/'quality_pairs.log')
            if not (policies/'selected.json').exists():
                child([PY,TOOLS/'optimize_policy.py','--capture',cap,'--pairs',root/'calibration/pair_costs.json',
                       '--out',policies/'selected.json','--dispatch','dynamic','--deadline',deadline],deadline,session/'optimize.log')
            if not all((policies/(s+'.json')).exists() for s in ('frequency','quality')):
                child([PY,TOOLS/'ablation_policies.py','--capture',cap,'--pairs',root/'calibration/pair_costs.json',
                       '--out-dir',policies],deadline,session/'ablation_policies.log')
            atomic_json(root/'calibration/COMPLETE.json',dict(studyhash=studyhash,active_layers=
                        sum(x['enabled'] for x in json.loads((policies/'selected.json').read_text())['layers'])))
        if a.stage!='calibrate' and not (root/'calibration/COMPLETE.json').exists():
            raise RuntimeError('Finish --stage calibrate first, using this SAME tag')
        def measured(stage,core_only=False):
            modes=list(DIAGNOSTIC if stage=='diagnostic' else FORMAL)
            for rep in ([a.repeat] if a.repeat is not None else range(3)):
                shift=rep*len(modes)//3;ordered=modes[shift:]+modes[:shift]
                if rep%2:ordered=list(reversed(ordered))
                if core_only:ordered=[x for x in ordered if x[0] in CORE]
                for name,mode,backend,fraction in ordered:
                    policy=(policies/'selected.json' if name.startswith('selective') else
                            policies/(name.split('_')[0]+'.json') if name in ('frequency_cwh','quality_cwh') else legacy)
                    for which in (('short','long') if rep%2==0 else ('long','short'))+('quality',):
                        unit(stage,rep,name,mode,backend,fraction,which,policy,
                             split='diagnostic' if stage=='diagnostic' else 'test')
                    (root/stage/f'rep_{rep}'/name/'COMPLETE').write_text('Completed and validated three units\n')
        if a.stage in ('core','focus','all'):measured('formal',core_only=True)
        if a.stage in ('diagnostic','all'):measured('diagnostic')
        if a.stage in ('formal','all'):measured('formal')
        if a.stage in ('verify','profile','all'):
            for phase in (('verify','profile') if a.stage=='all' else (a.stage,)):
                done=root/(phase+'_COMPLETE.json')
                if not done.exists():
                    enough(deadline,3600)
                    out=root/phase/('attempt_'+stamp)
                    child([PY,TOOLS/'run_suite.py','--phase',phase,'--out',out,'--workloads',work,
                           '--model',a.model,'--qa',a.qa,'--policy',policies/'selected.json'],deadline,session/(phase+'.log'))
                    atomic_json(done,dict(out=str(out),studyhash=studyhash))
        atomic_json(session/'state.json',dict(status='STAGE_COMPLETE',stage=a.stage,completed_epoch=time.time()))
        print('STAGE_COMPLETE',a.stage,'RESULTS',root,flush=True)
    except Paused as e:
        atomic_json(session/'state.json',dict(status='PAUSED',reason=str(e),completed_epoch=time.time()))
        print('PAUSED_SAFELY:',e,'\nReallocate GPUs, then repeat the SAME stage/tag command.',flush=True)
    except BaseException as e:
        atomic_json(session/'state.json',dict(status='FAILED',reason=str(e),completed_epoch=time.time()))
        raise
    finally:
        lock.close()


if __name__=='__main__':main()
