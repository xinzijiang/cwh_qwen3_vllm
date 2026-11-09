import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PACK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACK))
from cwh_v4.policy import read_policy


class OptimizerIntegration(unittest.TestCase):
    def test_selects_useful_exchange_and_rejects_validation_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pairs=dict(layers={})
            for layer in range(48):
                for split in ('train','validation'):
                    # Train is overloaded on GPU0. Odd-layer validation shifts
                    # the bottleneck to GPU2, so H0->replica2 must be rejected.
                    e=list(range(8)) if split=='train' or layer%2==0 else list(range(64,72))
                    record=dict(layer=layer,call=16,ids=[e]*32,decode=[False]*16+[True]*16)
                    (root/f'{split}_layer{layer:02d}.jsonl').write_text(json.dumps(record)+'\n')
                pairs['layers'][str(layer)]=[dict(cold=71,warm=64,relative_error=.1,relative_layer_rms_error=.01,
                                                   conditional_rows=32,decode_mass_fraction=.001)]
            (root/'pairs.json').write_text(json.dumps(pairs))
            env=__import__('os').environ.copy();env['PYTHONPATH']=str(PACK)
            subprocess.run([sys.executable,str(PACK/'tools/optimize_policy.py'),'--capture',str(root),
                '--pairs',str(root/'pairs.json'),'--out',str(root/'policy.json')],env=env,
                check=True,stdout=subprocess.DEVNULL)
            policy=read_policy(root/'policy.json')
            self.assertTrue(all(policy[l]['enabled'] for l in range(0,48,2)))
            self.assertFalse(any(policy[l]['enabled'] for l in range(1,48,2)))


if __name__=='__main__':unittest.main()
