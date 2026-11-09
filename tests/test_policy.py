import io, json,sys,unittest
from pathlib import Path
from collections import Counter

sys.path.insert(0,str(Path(__file__).parents[1]))
from cwh_vllm.policy import remap_weights, validate_audit,apply_expert_redirects,read_policy
from scripts.make_policy import build

class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.path=Path(__file__).parents[1]
        self.policy=build(self.path/'data/router_counts.csv',8)
        self.policies={d['layer']:d for d in self.policy['layers']}

    def test_selection(self):
        self.assertEqual(len(self.policies),48)
        self.assertEqual(set(self.policies),set(range(48)))
        for cfg in self.policies.values():
            self.assertIn(cfg['hot'],range(32))
            self.assertIn(cfg['warm'],range(32,64))
            self.assertIn(cfg['cold'],range(64,96))

    def test_loader_replaces_every_c_matrix(self):
        names=[]
        src=[]
        for i in range(48):
            c,h=self.policies[i]['cold'],self.policies[i]['hot']
            for suffix in ('gate_proj','up_proj','down_proj'):
                # Original C may arrive before H. Alias must still win.
                src.append((f'model.layers.{i}.mlp.experts.{c}.{suffix}.weight',('C',i,suffix)))
                src.append((f'model.layers.{i}.mlp.experts.{h}.{suffix}.weight',('H',i,suffix)))
            src.append((f'model.layers.{i}.mlp.gate.weight',('gate',i)))
        audit=Counter()
        changed=list(remap_weights(src,self.policies,audit))
        validate_audit(audit)
        self.assertEqual(len(changed),len(src))  # replace C with alias H (same 32 slots)
        for i in range(48):
            c,h=self.policies[i]['cold'],self.policies[i]['hot']
            for suffix in ('gate_proj','up_proj','down_proj'):
                target=f'model.layers.{i}.mlp.experts.{c}.{suffix}.weight'
                self.assertEqual([t for n,t in changed if n==target],[('H',i,suffix)])
                orig=f'model.layers.{i}.mlp.experts.{h}.{suffix}.weight'
                self.assertEqual([t for n,t in changed if n==orig],[('H',i,suffix)])

    def test_route_weights_and_token_selection(self):
        import torch
        ids=torch.tensor([[71,16,9,5,2,3,4,8],[16,7,18,4,10,11,12,13]])
        logits=torch.zeros(2,128)
        logits[0,29]=2  # hot=16 -> compare 29 with 53: True
        logits[1,29]=0  # compare 29 with 53: False
        logits[1,53]=1
        out=apply_expert_redirects(ids,logits,71,34,16)
        self.assertEqual(out[0,0].item(),34)
        self.assertEqual(out[0,1].item(),71)
        self.assertEqual(out[1,0].item(),16)
        self.assertTrue(torch.equal(ids,torch.tensor([[71,16,9,5,2,3,4,8],[16,7,18,4,10,11,12,13]])))

if __name__=='__main__':unittest.main()
