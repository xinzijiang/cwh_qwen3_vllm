#!/usr/bin/env python3
"""Isolated server per arm per repeat; fixed arrays, fail-fast, no live patching."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time

PACK=Path(__file__).resolve().parents[1]
TOOLS=PACK/'tools'
PY=sys.executable

DIAGNOSTIC=[
    ('native',None,None,None),
    ('legacy_v1','v1',None,None),
    ('cw_torch','cw','torch',0),
    ('legacy_torch','legacy','torch',None),
    ('legacy_shadow','legacy_shadow','torch',None),
    ('cw_fused','cw','triton',0),
    ('legacy_fused','legacy','triton',None),
    ('p25_fused','ratio','triton',.25),
    ('p50_fused','ratio','triton',.5),
    ('p100_fused','ratio','triton',1),
]
FORMAL=[('native',None,None,None), ('eplb_default','eplb',None,None),
        ('fixed_legacy_fused','legacy','triton',None),
        ('frequency_cwh','ratio','triton',None),
        ('quality_cwh','ratio','triton',None),
        ('selective_cw','cw','triton',0),
        ('selective_full','ratio','triton',None),
        ('selective_full_torch','ratio','torch',None)]


def run(args, **kw):
    print('RUN', ' '.join(map(str,args)),flush=True)
    subprocess.run([str(x) for x in args],check=True,**kw)


class Server:
    def __init__(self,a,name,mode,backend,fraction,policy=None,capture=None,profile=False):
        self.a=a;self.target=a.out/name;self.target.mkdir(parents=True,exist_ok=True)
        self.env=os.environ.copy()
        for key in ('CWH_FRACTION','CWH_CAPTURE_DIR','CWH_CAPTURE_MARKER','VLLM_TORCH_PROFILER_DIR'):
            self.env.pop(key,None)
        self.env.update(CWH_POLICY=str(policy or a.legacy_policy), CWH_MODE=mode or 'cw',
                        CWH_BACKEND=backend or 'torch',CWH_AUDIT_DIR=str(self.target/'weight_audits'),
                        CWH_REPLICATE='1',CWH_TRACE_ROUTING='0')
        if fraction is not None:self.env['CWH_FRACTION']=str(fraction)
        if capture:
            self.env.update(CWH_CAPTURE_DIR=str(capture),CWH_CAPTURE_MARKER=str(self.target/'marker'))
        if profile:self.env['VLLM_TORCH_PROFILER_DIR']=str(self.target/'profiler')
        self.cmd=[PY,'-m','vllm.entrypoints.openai.api_server','--model',str(a.model),
                  '--served-model-name','qwen3-moe','--host','127.0.0.1','--port',str(a.port),
                  '--tensor-parallel-size','4','--enable-expert-parallel',
                  '--all2all-backend','allgather_reducescatter','--attention-backend','FLASH_ATTN',
                  '--dtype','bfloat16','--no-enable-prefix-caching','--no-enable-chunked-prefill',
                  '--stream-interval','1','--safetensors-load-strategy','eager','--enforce-eager',
                  '--max-model-len','4096','--max-num-seqs',str(a.concurrency),
                  '--gpu-memory-utilization','0.85','--seed','20261103',
                  '--block-size','16','--num-gpu-blocks-override','16384']
        if mode in ('eplb','eplb0'):
            self.cmd+=['--enable-eplb','--eplb-config',json.dumps(dict(num_redundant_experts=(0 if mode=='eplb0' else 4),
                        window_size=1000,step_interval=3000,use_async=False))]
        elif mode is not None:
            arch=('CWHIdentityQwen3MoeForCausalLM' if mode=='identity' else
                  'CWHProbeQwen3MoeForCausalLM' if mode=='probe' else
                  'CWHV1Qwen3MoeForCausalLM' if mode=='v1' else 'CWHV2Qwen3MoeForCausalLM')
            self.cmd+=['--hf-overrides',json.dumps(dict(architectures=[arch]))]
        self.p=None;self.log=None

    def __enter__(self):
        s=socket.socket()
        try:s.bind(('127.0.0.1',self.a.port))
        finally:s.close()
        self.log=(self.target/'server.log').open('w')
        (self.target/'server_config.json').write_text(json.dumps(dict(command=self.cmd,
            environment={k:v for k,v in self.env.items() if k.startswith(('CWH_','VLLM_'))}),indent=2))
        self.p=subprocess.Popen(self.cmd,env=self.env,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            run([PY,TOOLS/'wait_ready.py','--url',f'http://127.0.0.1:{self.a.port}',
                 '--pid',self.p.pid,'--seconds','1800'])
            if self.env['CWH_MODE'] not in ('cw',) or '--hf-overrides' in self.cmd:
                # v1 reference only has structural counts, v2/probe has tensor hashes.
                if self.env['CWH_MODE'] not in ('v1','eplb') and '--hf-overrides' in self.cmd:
                    run([PY,TOOLS/'check_artifacts.py','weights',self.target/'weight_audits'])
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self,*unused):
        if self.p:
            try:os.killpg(self.p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:self.p.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid,signal.SIGKILL);self.p.wait()
            # Terminate workers even if parent exited first.
            try:os.killpg(self.p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        if self.log:self.log.close()
        time.sleep(5)


def bench(a,server,work,name,marker=None):
    cmd=[PY,TOOLS/'benchmark_stream.py','--base-url',f'http://127.0.0.1:{a.port}',
         '--workload',work,'--mode',server.target.name,'--repeat',str(a.repeat),
         '--concurrency',str(a.concurrency),'--output-tokens','256',
         '--warmup','16','--output-prefix',server.target/name]
    if marker:cmd+=['--trace-marker',server.env['CWH_CAPTURE_MARKER'],'--marker-value',marker]
    run(cmd)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['prepare','calibrate','diagnostic','formal','profile','verify'],required=True)
    p.add_argument('--model',type=Path,default=Path('/work/qwen3_real/models/Qwen3-30B-A3B'))
    p.add_argument('--source',type=Path,default=Path('/work/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json'))
    p.add_argument('--qa',type=Path,default=Path('/work/openbookqa/validation.jsonl'))
    p.add_argument('--workloads',type=Path,default=PACK/'workloads')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--legacy-policy',type=Path,default=PACK/'legacy/cwh_policy_48layers.json')
    p.add_argument('--policy',type=Path,default=PACK/'policies/selected.json')
    p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--concurrency',type=int,default=16)
    p.add_argument('--port',type=int,default=8765)
    a=p.parse_args()
    if a.out.exists():raise RuntimeError('Use a NEW output directory; never mix stale servers/results')
    a.out.mkdir(parents=True)
    if a.phase=='prepare':
        if a.workloads.exists():raise RuntimeError('Workloads exist: reuse frozen files; do not silently regenerate')
        run([PY,TOOLS/'prepare_splits.py','--source',a.source,'--model',a.model,'--out',a.workloads])
        return
    if a.phase in ('formal','verify','profile'):
        if a.phase=='formal':
            for s in ('frequency','quality'):
                shutil.copyfile(a.policy.with_name(s+'.json'),a.out/(s+'_policy.json'))
        snapshot=a.out/'selected_policy.json'
        shutil.copyfile(a.policy,snapshot)
        a.policy=snapshot
    for name,metadata in json.loads((a.workloads/'manifest.json').read_text()).items():
        if hashlib.sha256((a.workloads/name).read_bytes()).hexdigest()!=metadata['sha256']:
            raise RuntimeError(f'Frozen workload modified: {name}')
    import torch
    import vllm
    if vllm.__version__!='0.13.0' or torch.cuda.device_count()!=4:
        raise RuntimeError('Need original vLLM 0.13.0 container and four GPUs')
    gpus=[dict(name=torch.cuda.get_device_name(i),bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(4)]
    if any('A100' not in x['name'] or x['bytes']<75*1024**3 for x in gpus):
        raise RuntimeError('Use four A100-80GB GPUs for all primary comparisons')
    env=dict(torch=torch.__version__,vllm=vllm.__version__,gpus=gpus,
             manifest=json.loads((a.workloads/'manifest.json').read_text()),
             code_hashes={str(f.relative_to(PACK)):hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in PACK.rglob('*.py')},
             note='Eager / chunked prefill OFF / prefix caching OFF for every arm')
    (a.out/'environment.json').write_text(json.dumps(env,indent=2))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True)
    for t,c in zip([362,425,356,422],'ABCD'):
        if tok.decode([t]).strip()!=c:raise RuntimeError('QA token IDs do not match tokenizer')
    # Required correctness gate, not a pilot throughput experiment.
    with (a.out/'cuda_correctness.log').open('w') as f:
        run([PY,TOOLS/'gpu_check.py'],stdout=f,stderr=subprocess.STDOUT)
    if a.phase=='calibrate':
        capture=a.out/'capture'
        with Server(a,'native_probe','probe','torch',None,capture=capture) as server:
            for split in ('train','validation'):
                bench(a,server,a.workloads/f'{split}_mixed.jsonl',split,marker=split)
        run([PY,TOOLS/'check_artifacts.py','capture',capture])
        run([PY,TOOLS/'quality_pairs.py','--capture',capture,'--model',a.model,'--out',a.out/'pair_costs.json'])
        run([PY,TOOLS/'optimize_policy.py','--capture',capture,'--pairs',a.out/'pair_costs.json','--out',a.policy])
        run([PY,TOOLS/'ablation_policies.py','--capture',capture,'--pairs',a.out/'pair_costs.json','--out-dir',a.policy.parent])
        return
    if a.phase=='profile':
        # Instrumented mechanism study; never mixed with speed benchmark results.
        from profile_run import profile_work
        for name,mode,backend,fraction in [('native',None,None,None),
                ('selective_cw','cw','triton',0),('selective_dynamic','dynamic','triton',None)]:
            with Server(a,name,mode,backend,fraction,policy=a.policy,profile=True) as server:
                profile_work(a,server)
        return
    if a.phase=='verify':
        for name,mode,frac in [('selective_cw','cw',0),('selective_dynamic','dynamic',None)]:
            capture=a.out/name/'capture'
            with Server(a,name,mode,'triton',frac,policy=a.policy,capture=capture) as server:
                bench(a,server,a.workloads/'validation_mixed.jsonl','trace',marker='trace')
            run([PY,TOOLS/'check_artifacts.py','capture',capture])
            run([PY,TOOLS/'summarize_capture.py','--capture',capture,'--policy',a.policy,
                 '--out',a.out/name/'routing.json'])
        return
    modes=DIAGNOSTIC if a.phase=='diagnostic' else FORMAL
    # Rotate mode order across three independent allocations; reversal for block1.
    shift=(a.repeat*len(modes))//3
    modes=modes[shift:]+modes[:shift]
    if a.repeat%2:modes=list(reversed(modes))
    split='diagnostic' if a.phase=='diagnostic' else 'test'
    for name,mode,backend,fraction in modes:
        pol=a.policy if name.startswith('selective') else a.legacy_policy
        if name in ('frequency_cwh','quality_cwh'):
            pol=a.out/(name.split('_')[0]+'_policy.json')
        with Server(a,name,mode,backend,fraction,policy=pol) as server:
            order=('short','long') if a.repeat%2==0 else ('long','short')
            for w in order:
                bench(a,server,a.workloads/f'{split}_{w}.jsonl',w)
            run([PY,TOOLS/'evaluate_openbookqa.py','--dataset',a.qa,
                 '--base-url',f'http://127.0.0.1:{a.port}','--mode',name,
                 '--output-prefix',server.target/'quality'])
        (server.target/'COMPLETE').write_text('All performance/quality/weight checks succeeded\n')


if __name__=='__main__':main()
