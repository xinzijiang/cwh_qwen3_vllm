import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

PACK=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PACK),str(PACK/'tools')]
from cwh_v4.dynamic import optimal_quota,dynamic_reference
from optimize_policy import Replay
from durable import atomic_json,append_jsonl,read_journal,sha
import session


class DynamicTests(unittest.TestCase):
    def test_closed_form_exhaustive_against_integer_search(self):
        for a in range(13):
            for b in range(13):
                for o in range(13):
                    for h in range(a+1):
                        loads=[a,b,o,2]
                        candidates=[max(a-m,b+m,o,2) for m in range(h+1)]
                        target=min(candidates);expected=candidates.index(target)
                        # The no-improvement case deliberately moves zero.
                        got=optimal_quota(loads,0,1,h,0)
                        self.assertEqual(got,expected)

    def test_moving_bottleneck_and_guard(self):
        self.assertEqual(optimal_quota([50,30,44,20],0,1,20,0),6)
        self.assertEqual(optimal_quota([40,56,22,36],0,2,16,0),0)
        self.assertEqual(optimal_quota([40,56,22,36],1,2,16,0),16)
        self.assertEqual(optimal_quota([50,30,44,20],0,1,20,10000),0)

    def test_mixed_phase_replay_matches_full_call_dispatch(self):
        rng=random.Random(42)
        ids=[rng.sample(range(32),8) for _ in range(40)]
        for row in ids:row[0]=16
        d=[False]*23+[True]*17
        rec=dict(layer=0,call=16,ids=ids,decode=d)
        cfg=dict(layer=0,enabled=True,cold=71,warm=34,hot=16)
        out=np.array(dynamic_reference(ids,cfg));before=np.array(ids)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.jsonl';p.write_text(json.dumps(rec)+'\n')
            replay=Replay(p);g=replay.dynamic_gains(71,34,16)
            for phase,mask in [('decode',np.array(d)),('prefill',~np.array(d))]:
                a=np.bincount((before[mask]//32).ravel(),minlength=4)
                b=np.bincount((out[mask]//32).ravel(),minlength=4)
                self.assertAlmostEqual(g[phase],1-b.max()/a.max())

    def test_static_kernel_factory_smoke_without_cuda(self):
        # Catches the previous undefined `return kernel` without pretending to JIT.
        tri=types.ModuleType('triton');tri.jit=lambda f:f
        tl=types.ModuleType('triton.language');tl.constexpr=object;tri.language=tl
        with patch.dict(sys.modules,{'torch':types.ModuleType('torch'),'triton':tri,'triton.language':tl}):
            spec=importlib.util.spec_from_file_location('cwh_v4.routing_smoke',PACK/'cwh_v4/routing.py')
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            self.assertTrue(callable(mod._get_kernel()))
            self.assertIs(mod._get_kernel(),mod._get_kernel())


class ResumeTests(unittest.TestCase):
    def test_atomic_and_torn_journal_repair(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'state.json';atomic_json(p,dict(done=2));atomic_json(p,dict(done=3))
            self.assertEqual(json.loads(p.read_text()),dict(done=3))
            j=Path(t)/'requests.jsonl';append_jsonl(j,dict(i=1))
            with j.open('ab') as f:f.write(b'{"i":2')
            self.assertEqual(read_journal(j,repair=True),[dict(i=1)])
            append_jsonl(j,dict(i=2));self.assertEqual(len(read_journal(j)),2)

    def test_completed_unit_hash_guard(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);f=root/'short.json';atomic_json(f,dict(success=800))
            done=root/'done.json';atomic_json(done,dict(fingerprint='a',files={'short.json':sha(f)}))
            self.assertTrue(session.valid_done(done,'a'))
            with self.assertRaises(RuntimeError):session.valid_done(done,'b')
            atomic_json(f,dict(success=700))
            with self.assertRaises(RuntimeError):session.valid_done(done,'a')

    def test_workflow_interruption_and_skip_completed_benchmarks(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t);pack=base/'project/bundle';pack.mkdir(parents=True)
            (pack/'legacy').mkdir();(pack/'legacy/cwh_policy_48layers.json').write_text('{}')
            model=base/'model';model.mkdir()
            for name in ('config.json','model.safetensors.index.json'):(model/name).write_text('{}')
            source=base/'source.json';source.write_text('[]')
            qa=base/'qa.jsonl';qa.write_text('{}\n')
            root=pack.parent/'runs/cwh_v4_test';work=root/'workloads';work.mkdir(parents=True)
            manifest={}
            for name in ('test_short.jsonl','test_long.jsonl'):
                f=work/name;f.write_text('{}\n');manifest[name]=dict(sha256=sha(f),count=800)
            atomic_json(work/'manifest.json',manifest)
            policies=root/'policies';policies.mkdir()
            atomic_json(policies/'selected.json',dict(layers=[]))
            atomic_json(root/'calibration/COMPLETE.json',dict(files={}))
            counts=dict(bench=0,qa=0);paused=[False]
            class FakeServer:
                def __init__(self,a,name,*args,**kwargs):
                    self.target=a.out/name;self.target.mkdir();self.env={}
                def __enter__(self):return self
                def __exit__(self,*unused):pass
            def fake_child(cmd,deadline,log):
                cmd=list(map(str,cmd))
                if any('preflight.py' in x for x in cmd):
                    atomic_json(Path(cmd[cmd.index('--out')+1]),{k:'fake' for k in ('python','torch','cuda','vllm','triton','transformers')})
                    return
                if not any('benchmark_stream.py' in x or 'evaluate_openbookqa.py' in x for x in cmd):return
                prefix=Path(cmd[cmd.index('--output-prefix')+1]);isqa='--dataset' in cmd
                if isqa and not paused[0]:paused[0]=True;raise session.Paused('simulated disconnect')
                if isqa:
                    counts['qa']+=1;data=dict(error_count=0,questions=500)
                else:
                    counts['bench']+=1;data=dict(error_count=0,success_count=800,total_output_tokens=800*256)
                atomic_json(prefix.with_suffix('.json'),data)
                prefix.with_suffix('.csv').write_text('test fixture\n')
            torch=types.ModuleType('torch')
            torch.__version__='fake'
            torch.cuda=types.SimpleNamespace(device_count=lambda:4,get_device_name=lambda i:'A100-80GB',
                  get_device_properties=lambda i:types.SimpleNamespace(total_memory=80*1024**3))
            vllm=types.ModuleType('vllm');vllm.__version__='0.13.0'
            argv=['session.py','--stage','main','--tag','test','--model',str(model),'--source',str(source),'--qa',str(qa)]
            with patch.object(session,'PACK',pack),patch.object(session,'ManagedServer',FakeServer),\
                 patch.object(session,'child',fake_child),patch.object(session.signal,'signal'),\
                 patch.dict(sys.modules,{'torch':torch,'vllm':vllm}),patch.object(sys,'argv',argv):
                session.STOP=False;session.main()
                self.assertEqual(counts['bench'],2)
                session.main()
                self.assertEqual(counts['bench'],24) # not 26: two complete units skipped
                self.assertEqual(counts['qa'],12)
                self.assertEqual(len(list((root/'main').rglob('.done_*.json'))),36)


if __name__=='__main__':unittest.main()
