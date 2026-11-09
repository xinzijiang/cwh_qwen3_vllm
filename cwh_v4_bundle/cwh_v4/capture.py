"""Separate intrusive calibration only. One file per EP rank avoids log mixing.

Full routing data are saved only by EP0, appropriate for this pinned gathered
Ag/Rs path. Cross-rank route fingerprints must match before accepting the data.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


class Capture:
    def __init__(self, rank):
        self.rank = rank
        self.root = Path(os.environ['CWH_CAPTURE_DIR'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.marker = Path(os.environ['CWH_CAPTURE_MARKER'])
        self.stride = int(os.getenv('CWH_CAPTURE_STRIDE', '16'))
        self.cap = int(os.getenv('CWH_CAPTURE_CAP', '1024'))
        self.counts = {}
        self.hidden_counts = {}
        self.real_calls = {}
        self.phase_counts = {}

    def admit(self, split, layer, is_prefill):
        key=(split,layer)
        n=self.real_calls.get(key,0)+1
        self.real_calls[key]=n
        phase_key=(split,layer,bool(is_prefill))
        limit=128 if is_prefill else self.cap
        if self.phase_counts.get(phase_key,0)>=limit:return False
        if not is_prefill and n!=1 and n%self.stride:return False
        self.phase_counts[phase_key]=self.phase_counts.get(phase_key,0)+1
        return True

    def __call__(self, layer, call, hidden, ids, weights, output):
        # Only calibration/trace architecture calls this, never timing architecture.
        if not self.marker.is_file():
            return
        split = self.marker.read_text().strip()
        if split not in ('train', 'validation', 'trace'):
            raise RuntimeError('Capture marker must contain train/validation/trace')
        key = split, layer
        from vllm.forward_context import get_forward_context
        md = get_forward_context().attn_metadata
        if isinstance(md, dict):md = next(iter(md.values()))
        if md is None or isinstance(md,list) or not hasattr(md,'max_query_len'):
            raise RuntimeError('Real request FlashAttention metadata required for stratified capture')
        if not self.admit(split,layer,int(md.max_query_len)>1):return
        self.counts[key] = self.counts.get(key, 0) + 1
        a = ids.detach().cpu().numpy()
        b = output.detach().cpu().numpy()
        w = weights.detach().float().cpu().numpy()
        digest = hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest()
        # A few cross-rank fingerprints validate this callback sees gathered IDs.
        if self.counts[key] <= 8:
            with (self.root / f'fingerprint_rank{self.rank}.jsonl').open('a') as f:
                f.write(json.dumps(dict(split=split, layer=layer, call=call,
                                       digest=digest, rows=len(a))) + '\n')
        if self.rank != 0:
            return
        from vllm.forward_context import get_forward_context
        md = get_forward_context().attn_metadata
        if isinstance(md, dict):
            md = next(iter(md.values()))
        if isinstance(md, list):
            raise RuntimeError('DBO/microbatch metadata unsupported: keep supplied launch flags')
        if md is None or not hasattr(md, 'query_start_loc'):
            raise RuntimeError('Real request FlashAttention metadata required for capture')
        q = md.query_start_loc.detach().cpu().numpy()
        actual = int(md.num_actual_tokens)
        if int(q[-1]) != actual or not actual <= len(a) <= actual + 3:
            raise RuntimeError(f'Gathered token alignment invalid: actual={actual}, route={len(a)}')
        # With chunked prefill OFF and prompts >=96 tokens, qlen==1 is Decode.
        # Do not infer phase from batch size. Strip any padded route rows.
        phase = np.repeat(np.diff(q) == 1, np.diff(q)).astype(np.int8)
        record = dict(split=split, layer=layer, call=call, ids=a[:actual].tolist(),
                      weights=w[:actual].tolist(), after=b[:actual].tolist(),
                      decode=phase.tolist(), route_rows=len(a), actual_rows=actual,
                      padding_ids=a[actual:].tolist(),
                      phase_definition='query_len==1; chunked prefill disabled; prompts>=96')
        with (self.root / f'{split}_layer{layer:02d}.jsonl').open('a') as f:
            f.write(json.dumps(record, separators=(',', ':')) + '\n')
        # Train-only representative activations for actual SwiGLU C/W error.
        n = self.hidden_counts.get(key, 0)
        if split == 'train' and n < 256 and actual:
            choose = np.linspace(0, actual - 1, min(actual, 8), dtype=int)
            x = hidden[choose.tolist()].detach().float().cpu().numpy()
            with (self.root / f'hidden_layer{layer:02d}.npyseq').open('ab') as f:
                for v in (x, a[choose], w[choose]):
                    np.save(f, v, allow_pickle=False)
            self.hidden_counts[key] = n + 1
