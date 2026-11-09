"""Bounded official torch profiler window; these timings are NOT speed results."""
import json
from pathlib import Path
import subprocess
import sys
import urllib.request


def profile_work(a,server):
    tools=Path(__file__).parent
    work=a.workloads/'test_long.jsonl'
    tiny=server.target/'profile64.jsonl'
    tiny.write_text('\n'.join(work.read_text().split('\n')[:64])+'\n')
    from benchmark_stream import one_request
    rows=[json.loads(s) for s in tiny.read_text().split('\n') if s.strip()]
    for r in rows[:16]:
        result=one_request(f'http://127.0.0.1:{a.port}','qwen3-moe',r,16,1200)
        if result['error']:raise RuntimeError(result)
    def post(endpoint):
        req=urllib.request.Request(f'http://127.0.0.1:{a.port}/{endpoint}',data=b'',method='POST')
        with urllib.request.urlopen(req,timeout=600) as res:
            if res.status!=200:raise RuntimeError(endpoint)
    post('start_profile')
    try:
        subprocess.run([sys.executable,str(tools/'benchmark_stream.py'),'--base-url',f'http://127.0.0.1:{a.port}',
          '--workload',str(tiny),'--mode','INSTRUMENTED_NOT_SPEED','--repeat','0','--concurrency',str(a.concurrency),
          '--output-tokens','32','--warmup','0','--output-prefix',str(server.target/'instrumented')],check=True)
    finally:
        post('stop_profile')


if __name__=='__main__':
    raise SystemExit('Use run_suite.py --phase profile')
