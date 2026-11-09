"""One extra kernel after upstream fused_topk; no CPU sync in timing runs.

This fuses only post-topk C->W and H->C remapping, NOT topk, dispatch or GEMM.
Shadow is eager torch only so mask computation cannot be dead-code eliminated.
"""
import torch
import triton
import triton.language as tl
from .policy import salt_for


def torch_redirect(ids, logits, cfg, mode, fraction, call):
    c, w, h = (cfg[k] for k in ('cold', 'warm', 'hot'))
    base = torch.where(ids == c, w, ids)
    if mode == 'cw' or (mode in ('ratio', 'shadow') and fraction == 0):
        return base
    if mode in ('legacy', 'legacy_shadow'):
        mask = (logits[:, (h + 13) % 128] > logits[:, (h + 37) % 128]).unsqueeze(1)
    else:
        # Mask low 32 bits after each integer multiplication (modular arithmetic).
        x = (torch.arange(ids.shape[0], device=ids.device, dtype=torch.int64)
             + salt_for(cfg['layer'], call)) & 0xffffffff
        x = ((x ^ (x >> 16)) * 0x7feb352d) & 0xffffffff
        x = ((x ^ (x >> 15)) * 0x846ca68b) & 0xffffffff
        mask = (((x ^ (x >> 16)) & 0xffffff) < int(fraction * (1 << 24))).unsqueeze(1)
    mask = (ids == h) & mask
    # Eager torch executes the same final where kernel for shadow and real.
    return torch.where(mask, base if 'shadow' in mode else c, base)


_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    @triton.jit
    def cwh_post_topk_redirect(IDS, LOGITS, OUT, N, K: tl.constexpr,
               RS: tl.constexpr, CS: tl.constexpr, SALT,
               C: tl.constexpr, W: tl.constexpr, H: tl.constexpr,
               MODE: tl.constexpr, THRESHOLD: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        valid = i < N
        row = i // K
        e = tl.load(IDS + i, valid, other=-1)
        result = tl.where(e == C, W, e)
        if MODE == 1:
            a = tl.load(LOGITS + row * RS + ((H + 13) % 128) * CS, valid, other=0)
            b = tl.load(LOGITS + row * RS + ((H + 37) % 128) * CS, valid, other=0)
            move = a > b
            result = tl.where((e == H) & move, C, result)
        elif MODE == 2:
            x = row.to(tl.uint32) + SALT.to(tl.uint32)
            x = (x ^ (x >> 16)) * 0x7feb352d
            x = (x ^ (x >> 15)) * 0x846ca68b
            move = ((x ^ (x >> 16)) & 0xffffff) < THRESHOLD
            result = tl.where((e == H) & move, C, result)
        tl.store(OUT + i, result, valid)

    _kernel = cwh_post_topk_redirect
    return _kernel


def redirect(ids, logits, cfg, mode='ratio', backend='triton', fraction=None, call=1):
    if not cfg['enabled'] or ids.numel() == 0:
        return ids
    if mode == 'dynamic':
        if backend == 'triton':
            from .dynamic_cuda import redirect_dynamic
            return redirect_dynamic(ids,cfg,int(cfg.get('online_gain_bp',100)))
        if backend == 'torch':
            from .dynamic import dynamic_reference
            # Debug oracle only: includes a CPU transfer, NEVER a speed baseline.
            return torch.tensor(dynamic_reference(ids.cpu().tolist(),cfg,int(cfg.get('online_gain_bp',100))),
                                device=ids.device,dtype=ids.dtype)
        raise ValueError(backend)
    p = cfg['replica_fraction'] if fraction is None else fraction
    if backend == 'torch':
        return torch_redirect(ids, logits, cfg, mode, p, call)
    if backend != 'triton' or 'shadow' in mode:
        raise ValueError('Use eager torch for shadow; do not let JIT remove shadow work')
    if not ids.is_cuda or not ids.is_contiguous() or ids.ndim != 2:
        raise RuntimeError('Triton requires contiguous CUDA [tokens,k] IDs')
    out = torch.empty_like(ids)
    m = 0 if mode == 'cw' or p == 0 and mode == 'ratio' else 1 if mode == 'legacy' else 2
    _get_kernel()[((ids.numel() + 255) // 256,)](
        ids, logits, out, ids.numel(), ids.shape[1], logits.stride(0), logits.stride(1),
        salt_for(cfg['layer'], call), cfg['cold'], cfg['warm'], cfg['hot'], m,
        int(p * (1 << 24)), 256, num_warps=4)
    return out
