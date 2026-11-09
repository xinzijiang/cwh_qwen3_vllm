import json,random,sys,tempfile,unittest
from pathlib import Path
PACK=Path(__file__).resolve().parents[1];sys.path[:0]=[str(PACK),str(PACK/'tools')]
from cwh_v4.cost import choose_move,loads,reference,bucket_for
from calibrate_cost import nnls_small
import numpy as np

class CostTests(unittest.TestCase):
    def test_grid_minimum_and_conservation(self):
        rng=random.Random(41)
        for _ in range(400):
            c,w,h=rng.sample(range(128),3)
            if c//32==h//32:continue
            cfg=dict(cold=c,warm=w,hot=h,cost_coeff_ns=[1200,240,7],cost_tile=16,online_gain_bp=0)
            ids=[rng.sample(range(128),8) for _ in range(rng.randint(1,80))]
            counts=np.bincount(np.array(ids).ravel(),minlength=128).tolist()
            out,m=reference(ids,cfg);base=counts.copy();base[w]+=base[c];base[c]=0
            costs={}
            for q in sorted({counts[h]*i//4 for i in range(5)}):
                v=base.copy();v[h]-=q;v[c]+=q;costs[q]=max(loads(v,cfg))
            expected=min(costs,key=lambda q:(costs[q],q))
            self.assertEqual(m,expected)
            self.assertEqual(len(np.array(out).ravel()),len(ids)*8)
            for before,after in zip(ids,out):
                for e,z in zip(before,after):
                    self.assertIn(z,[w] if e==c else [h,c] if e==h else [e])
    def test_whole_hot_avoids_extra_activation(self):
        cfg=dict(cold=71,warm=72,hot=3,cost_coeff_ns=[1000,0,0],cost_tile=16,online_gain_bp=0)
        counts=[0]*128
        for e in range(16):counts[e]=5
        for e in range(64,68):counts[e]=5
        self.assertEqual(choose_move(counts,cfg),5)
    def test_margin_rejects_small_gain(self):
        cfg=dict(cold=71,warm=72,hot=3,cost_coeff_ns=[1000,0,0],cost_tile=16,online_margin_ns=1001)
        counts=[0]*128
        for e in range(16):counts[e]=5
        self.assertEqual(choose_move(counts,cfg),0)
    def test_nonnegative_fit(self):
        rng=np.random.default_rng(42);x=rng.uniform(size=(100,4));truth=np.array([3,0,5,8])
        got=nnls_small(x,x@truth)
        np.testing.assert_allclose(got,truth,atol=1e-10)
    def test_unicode_and_blank_replay(self):
        from optimize_policy import Replay
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.jsonl'
            rows=[dict(layer=0,call=1,ids=[[1,2,3,4,5,6,7,8]],decode=[1],note='a\u2028b'),
                  dict(layer=0,call=2,ids=[[1,2,3,4,5,6,7,8]],decode=[0])]
            p.write_text('\n'+''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)+'\n')
            got=Replay(p);self.assertEqual(len(got.all_counts),2)
    def test_buckets(self):
        self.assertEqual([bucket_for(n) for n in (1,2,3,16,17,512)],[1,2,4,16,32,512])

if __name__=='__main__':unittest.main()
