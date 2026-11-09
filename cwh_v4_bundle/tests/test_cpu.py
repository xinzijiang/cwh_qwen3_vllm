"""CPU tests for slot invariants, duplicate semantics, exact replay and HTTP checks."""
import contextlib
from collections import Counter
import http.server
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import unittest
import numpy as np

PACK=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PACK),str(PACK/'tools')]
from cwh_v4.policy import read_policy,remap_weights,validate_audit,reference_ids,hash24,salt_for
from optimize_policy import Replay,masks,GRID
from benchmark_stream import one_request


class PolicyTests(unittest.TestCase):
    def test_general_owners_and_disabled_weight_stream(self):
        policies={l:dict(layer=l,enabled=l%2==0,cold=3,warm=4,hot=120,replica_fraction=.5) for l in range(48)}
        # C and W on the SAME rank, H on rank3, both rejected by old implementation.
        stream=[]
        sentinels={}
        for l in range(48):
            for e in (3,4,120):
                for proj in ('gate_proj','up_proj','down_proj'):
                    name=f'model.layers.{l}.mlp.experts.{e}.{proj}.weight'
                    sentinels[name]=object();stream.append((name,sentinels[name]))
        audit=Counter();out=dict(remap_weights(stream,policies,audit));validate_audit(audit,policies)
        for l in range(48):
            name=f'model.layers.{l}.mlp.experts.3.gate_proj.weight'
            source=f'model.layers.{l}.mlp.experts.120.gate_proj.weight' if l%2==0 else name
            self.assertIs(out[name],sentinels[source])
        audit[0,'cold_skipped']-=1
        with self.assertRaises(RuntimeError):validate_audit(audit,policies)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'p.json';p.write_text(json.dumps(dict(layers=list(policies.values()))))
            self.assertEqual(read_policy(p)[0]['hot'],120)
            bad=list(policies.values());bad[0]=bad[0]|dict(hot=5)
            p.write_text(json.dumps(dict(layers=bad)))
            with self.assertRaises(ValueError):read_policy(p)

    def test_routing_conservation_and_collisions(self):
        cfg=dict(layer=7,enabled=True,cold=71,warm=34,hot=16,replica_fraction=.5)
        ids=[[71,34,16,9,3,7,6,5]]*16
        a=reference_ids(ids,cfg,'ratio',0)
        b=reference_ids(ids,cfg,'ratio',1)
        self.assertEqual(a[0][:3],[34,34,16])
        self.assertEqual(b[0][:3],[34,34,71])
        self.assertEqual(sum(map(len,a)),128)
        self.assertEqual(reference_ids(ids,cfg|dict(enabled=False)),ids)
        # Duplicate W means weights add. Never silently delete a top-k slot.
        vals={34:2,16:3,71:3,9:1,3:1,7:1,6:1,5:1}
        weights=[.12,.15,.2,.1,.1,.1,.1,.13]
        self.assertAlmostEqual(sum(w*vals[e] for w,e in zip(weights,a[0])),
                               sum(w*vals[e] for w,e in zip(weights,b[0])))

    def test_exact_replay_matches_oracle(self):
        rng=random.Random(7)
        ids=[rng.sample(range(128),8) for _ in range(32)]
        ids[0]=[71,34,16,1,2,3,4,5]
        d=[False]*16+[True]*16
        rec=dict(layer=7,call=123,ids=ids,decode=d)
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'r.jsonl';path.write_text(json.dumps(rec)+'\n')
            replay=Replay(path)
            cfg=dict(layer=7,enabled=True,cold=71,warm=34,hot=16)
            for pi,p in enumerate(GRID):
                out=np.asarray(reference_ids(ids,cfg,'ratio',p,123))
                before=np.asarray(ids)
                for phase,sel in [('prefill',~np.array(d)),('decode',np.array(d))]:
                    l0=np.bincount((before[sel]//32).ravel(),minlength=4)
                    l1=np.bincount((out[sel]//32).ravel(),minlength=4)
                    self.assertAlmostEqual(replay.gains(71,34,16,pi)[phase],1-l1.max()/l0.max())
                h=masks(len(ids),salt_for(7,123))[pi]
                self.assertEqual(h.tolist(),[hash24(i,salt_for(7,123))<int(p*(1<<24)) for i in range(len(ids))])

    def test_aggregate_cannot_bound_per_call_bottleneck(self):
        a=np.array([[64,0,0,0],[0,64,0,0],[0,0,64,0],[0,0,0,64]])
        self.assertEqual(a.sum(0).max()/a.sum(0).mean(),1)
        self.assertEqual((a.max(1)/a.mean(1)).mean(),4)

    def test_plugin_registration_metadata(self):
        matches=[e for e in importlib.metadata.entry_points(group='vllm.general_plugins') if e.name=='cwh_v4']
        self.assertEqual(len(matches),1)


class HTTPTests(unittest.TestCase):
    def test_fixed_token_stream_and_short_output_rejected(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                n=body['max_tokens'];bad=body['prompt']=='bad'
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                for i in range(n):
                    self.wfile.write(('data: '+json.dumps(dict(choices=[dict(text='a',finish_reason=None)]))+'\n\n').encode())
                obj=dict(choices=[dict(text='',finish_reason='length')],usage=dict(completion_tokens=n-1 if bad else n))
                self.wfile.write(('data: '+json.dumps(obj)+'\n\ndata: [DONE]\n\n').encode())
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_port}'
            result=one_request(url,'x',dict(prompt='good'),8,5)
            self.assertEqual(result['output_tokens'],8);self.assertFalse(result['error'])
            result=one_request(url,'x',dict(prompt='bad'),8,5)
            self.assertTrue(result['error'])
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
