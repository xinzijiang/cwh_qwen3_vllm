"""Policy validation and streamed BF16 checkpoint aliasing. No torch import."""
import json
import re
from pathlib import Path

EXPERT_NAME = re.compile(
    r'(?P<prefix>(?:^|\.)layers\.(?P<layer>\d+)\.mlp\.experts\.)(?P<expert>\d+)'
    r'(?P<suffix>\.(?:gate_proj|up_proj|down_proj)\.weight)$')


def read_policy(path):
    obj = json.loads(Path(path).read_text())
    result = {}
    for entry in obj['layers']:
        cfg = dict(entry)
        l = int(cfg['layer'])
        if l in result or not 0 <= l < 48:
            raise ValueError('Need each layer 0..47 exactly once')
        cfg['enabled'] = bool(cfg.get('enabled', True))
        if cfg['enabled']:
            c, w, h = (int(cfg[k]) for k in ('cold', 'warm', 'hot'))
            if not all(0 <= x < 128 for x in (c, w, h)) or len({c, w, h}) != 3:
                raise ValueError(f'Invalid distinct C/W/H at layer {l}')
            if c // 32 == h // 32:
                raise ValueError('H replica must reside on a different GPU')
            # W may be on ANY GPU, including the C owner. This avoids forcing
            # C traffic onto an unrelated third GPU.
            cfg.update(cold=c, warm=w, hot=h)
            p = float(cfg.get('replica_fraction', .5))
            if not 0 <= p <= 1:
                raise ValueError('replica_fraction must be in [0,1]')
            cfg['replica_fraction'] = p
        result[l] = cfg
    if set(result) != set(range(48)):
        raise ValueError('Exactly 48 layer entries required')
    return result


def remap_weights(weights, policies, audit):
    for name, tensor in weights:
        m = EXPERT_NAME.search(name)
        if m is None:
            yield name, tensor
            continue
        l, e = int(m['layer']), int(m['expert'])
        cfg = policies[l]
        if not cfg['enabled']:
            yield name, tensor
            continue
        if e == cfg['cold']:
            audit[l, 'cold_skipped'] += 1
            continue
        yield name, tensor
        if e == cfg['hot']:
            audit[l, 'hot_aliased_into_c'] += 1
            yield name[:m.start('expert')] + str(cfg['cold']) + name[m.end('expert'):], tensor


def validate_audit(audit, policies):
    for l, cfg in policies.items():
        for kind in ('cold_skipped', 'hot_aliased_into_c'):
            expected = 3 if cfg['enabled'] else 0
            if audit[l, kind] != expected:
                raise RuntimeError(f'Incomplete BF16 slot replacement layer={l} {kind}: '
                                   f'{audit[l, kind]} != {expected}')


def hash24(row, salt):
    """Identical uint32 mixer in Python, torch and Triton; not a semantic hash."""
    x = (int(row) + int(salt)) & 0xffffffff
    x = ((x ^ (x >> 16)) * 0x7feb352d) & 0xffffffff
    x = ((x ^ (x >> 15)) * 0x846ca68b) & 0xffffffff
    return (x ^ (x >> 16)) & 0xffffff


def salt_for(layer, call):
    return (int(layer) * 1000003 + int(call) * 9176 + 20261102) & 0xffffffff


def reference_ids(ids, cfg, mode='ratio', fraction=None, call=1):
    """List-based oracle for correctness tests and offline exact routing replay."""
    if not cfg.get('enabled', True):
        return [list(row) for row in ids]
    c, w, h = (cfg[k] for k in ('cold', 'warm', 'hot'))
    p = cfg.get('replica_fraction', .5) if fraction is None else fraction
    out = []
    for t, row in enumerate(ids):
        move = mode == 'ratio' and hash24(t, salt_for(cfg['layer'], call)) < int(p * (1 << 24))
        out.append([w if e == c else c if e == h and move else e for e in row])
    return out
