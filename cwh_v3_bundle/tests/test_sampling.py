import os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from cwh_v2.capture import Capture

class SamplingTests(unittest.TestCase):
    def test_prefill_cannot_alias_with_decode_stride(self):
        with tempfile.TemporaryDirectory() as t, patch.dict(os.environ,dict(CWH_CAPTURE_DIR=t,CWH_CAPTURE_MARKER=t+'/marker')):
            c=Capture(0)
            self.assertTrue(c.admit('train',0,True))
            for _ in range(15):c.admit('train',0,False)
            self.assertTrue(c.admit('train',0,True))
            self.assertTrue(c.admit('validation',0,True))
            for _ in range(126):self.assertTrue(c.admit('train',0,True))
            self.assertFalse(c.admit('train',0,True))
