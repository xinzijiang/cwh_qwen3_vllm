"""Separate instrumented run: profiler never contributes formal throughput."""
import json,time,sys,uuid,os
from pathlib import Path
from types import SimpleNamespace
from runtime import ManagedServer,enough,fresh_port,child
from durable import atomic_json,sha

def run_profiles(root,a,deadline,policies,work):
    from session import ARMS
    tools=Path(__file__).parent
    for name in ('native','identity','cwh_cw','cwh_static','cwh_old_dynamic','cwh_cost'):
        folder=root/'profile'/name;folder.mkdir(parents=True,exist_ok=True)
        done=folder/'COMPLETE.json'
        if done.exists():continue
        enough(deadline,900)
        attempt=folder/('attempt_'+uuid.uuid4().hex[:8]);attempt.mkdir()
        mode,backend,fraction,pname=ARMS[name]
        params=SimpleNamespace(out=attempt,model=a.model,port=fresh_port(),legacy_policy=policies/'selected.json',concurrency=16,deadline=deadline)
        server=ManagedServer(params,name,mode,backend,fraction,policy=policies/(pname+'.json'),profile=True)
        with server:
            small=attempt/'profile8.jsonl'
            rows=[json.loads(line) for line in (work/'diagnostic_long.jsonl').open() if line.strip()][:8]
            small.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            import urllib.request
            url=f'http://127.0.0.1:{params.port}'
            child([sys.executable,tools/'benchmark_stream.py','--base-url',url,'--workload',small,'--mode','PROFILE_WARMUP_NOT_SPEED','--output-prefix',attempt/'warmup','--concurrency','16','--output-tokens','32'],deadline,attempt/'warmup.log')
            def post(path):
                with urllib.request.urlopen(urllib.request.Request(url+'/'+path,data=b''),timeout=60) as r:
                    if r.status!=200:raise RuntimeError(path)
            post('start_profile')
            try:
                child([sys.executable,tools/'benchmark_stream.py','--base-url',url,'--workload',small,'--mode','PROFILE_NOT_SPEED','--output-prefix',attempt/'instrumented','--concurrency','16','--output-tokens','32','--warmup','0'],deadline,attempt/'profile.log')
            finally:post('stop_profile')
        traces=list(server.target.rglob('*.pt.trace.json*'))
        if not traces:raise RuntimeError('Profiler produced no trace; inspect '+str(server.target))
        atomic_json(done,dict(attempt=str(attempt),traces=[str(p) for p in traces],not_a_speed_result=True))
