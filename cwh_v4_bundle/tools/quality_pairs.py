#!/usr/bin/env python3
"""Actual conditional SwiGLU output error for train-only cold/warm candidates.

Run AFTER calibration server exits. Uses one GPU and original BF16 safetensors;
does not load the entire model. Error is a selection proxy, not a quality guarantee.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import time
from durable import atomic_json


def read_hidden(path):
    arrays = [[], [], []]
    with path.open('rb') as f:
        while f.tell() < path.stat().st_size:
            for a in arrays:
                a.append(np.load(f, allow_pickle=False))
    return tuple(np.concatenate(a) for a in arrays)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--min-rows', type=int, default=16)
    p.add_argument('--deadline',type=float,default=0)
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    index = json.loads((a.model / 'model.safetensors.index.json').read_text())['weight_map']
    def load(l, e):
        out = []
        for kind in ('gate_proj', 'up_proj', 'down_proj'):
            name = f'model.layers.{l}.mlp.experts.{e}.{kind}.weight'
            if name not in index:
                raise RuntimeError(f'Unrecognized BF16 checkpoint naming: {name}')
            with safe_open(str(a.model / index[name]), framework='pt', device='cpu') as f:
                out.append(f.get_tensor(name).to('cuda', dtype=torch.bfloat16))
        return out
    def expert(x, mats):
        gate, up, down = mats
        return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down).float()
    result = dict(definition='actual router-weighted C->W reconstruction RMS / native MoE RMS, train only', layers={})
    if a.out.exists():result=json.loads(a.out.read_text())
    for l in range(48):
        if str(l) in result['layers']:
            print('RESUME_SKIP_QUALITY_LAYER',l,flush=True);continue
        if a.deadline and time.time()+300>a.deadline:
            raise SystemExit(75)
        records = [json.loads(s) for s in (a.capture / f'train_layer{l:02d}.jsonl').read_text().split('\n') if s.strip()]
        counts = np.zeros(128)
        mass = np.zeros(128)
        for r in records:
            ids = np.array(r['ids']); ws = np.array(r['weights'])
            mask = np.array(r['decode'], dtype=bool)
            counts += np.bincount(ids[mask].ravel(), minlength=128)
            mass += np.bincount(ids[mask].ravel(), weights=ws[mask].ravel(), minlength=128)
        # Candidate search is bounded and documented, not a global optimum claim.
        cold = [int(g*32+i) for g in range(4) for i in np.argsort(counts[g*32:(g+1)*32])[:2]]
        # Evaluate every W, then retain the two lowest-error choices per GPU/C.
        # This avoids assuming frequent experts are good semantic substitutes.
        warm = list(range(128))
        x, ids, ws = read_hidden(a.capture / f'hidden_layer{l:02d}.npyseq')
        rows = []
        with torch.inference_mode():
            # Reconstruct native weighted top-k output on the SAME representative
            # hidden inputs. Expert-to-expert cosine alone ignores gate weights
            # and the scale of the full MoE mixture.
            all_x=torch.tensor(x,device='cuda',dtype=torch.bfloat16)
            native=torch.zeros_like(all_x,dtype=torch.float32)
            outputs=torch.empty((128,len(all_x),all_x.shape[1]),device='cuda',dtype=torch.float32)
            for e in range(128):
                outputs[e]=expert(all_x,load(l,e))
                idx=np.flatnonzero((ids==e).any(1))
                if not len(idx):continue
                y=outputs[e,idx.tolist()]
                alpha=torch.tensor((ws[idx]*(ids[idx]==e)).sum(1),device='cuda').float()
                native[idx.tolist()]+=alpha[:,None]*y
            layer_den=native.square().sum().clamp_min(1e-20)
            for c in cold:
                if counts[c] > .75 * counts.mean():
                    continue
                chosen = np.any(ids == c, axis=1)
                if chosen.sum() < a.min_rows:
                    continue
                cw = torch.tensor((ws[chosen] * (ids[chosen] == c)).sum(1), device='cuda').float()
                yc = outputs[c,np.flatnonzero(chosen).tolist()]
                den = ((cw[:, None] * yc).square().sum()).clamp_min(1e-20)
                for w in warm:
                    if w == c:
                        continue
                    yw = outputs[w,np.flatnonzero(chosen).tolist()]
                    numerator=(cw[:, None] * (yc - yw)).square().sum()
                    err = (numerator / den).sqrt().item()
                    rows.append(dict(cold=c, warm=w, relative_error=err,
                                     relative_layer_rms_error=(numerator/layer_den).sqrt().item(),
                                     conditional_rows=int(chosen.sum()),
                                     decode_mass_fraction=float(mass[c] / max(mass.sum(), 1e-20))))
        retained=[]
        for c in cold:
            for g in range(4):
                selected=sorted((r for r in rows if r['cold']==c and r['warm']//32==g),
                                key=lambda r:r['relative_layer_rms_error'])[:2]
                retained.extend(selected)
        result['layers'][str(l)] = retained
        result['warm_search']='all 128 pretrained experts; retain 2 lowest-error W per GPU per C'
        # Release the per-layer output cache before allocating the next layer.
        del outputs,native,all_x
        a.out.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(a.out,result)
        print('QUALITY_PAIRS', l, len(rows), flush=True)


if __name__ == '__main__':
    main()
