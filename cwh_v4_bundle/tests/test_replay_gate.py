import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from validate_policy_gpu import admissible
class ReplayGateTests(unittest.TestCase):
    def rows(self):
        return [dict(phase=p,native_ns=[[100,95,90,80] for _ in range(3)],cost_ns=[[96,90,92,80] for _ in range(3)]) for p in ('decode','prefill_mixed')]
    def test_consistent_gain(self):self.assertTrue(admissible(self.rows()))
    def test_one_bad_timing_pass_vetoes(self):
        r=self.rows();r[0]['cost_ns'][2][0]=105;self.assertFalse(admissible(r))
    def test_missing_phase_vetoes(self):self.assertFalse(admissible(self.rows()[:1]))
    def test_third_gpu_bottleneck_vetoes(self):
        r=self.rows();r[0]['cost_ns'][0][2]=110;self.assertFalse(admissible(r))
