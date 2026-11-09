"""Load-time C-slot reuse; selective layers retain their original C weights."""
import hashlib
import json
import os
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import torch
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
from .policy import read_policy, remap_weights, validate_audit


def check_environment(vllm_config):
    p = vllm_config.parallel_config
    if version('vllm') != '0.13.0':
        raise RuntimeError('Pinned to vLLM 0.13.0, not latest')
    if p.enable_eplb or not p.enable_expert_parallel or p.tensor_parallel_size != 4:
        raise RuntimeError('Custom model requires EP4/TP4 and EPLB disabled')
    if p.pipeline_parallel_size != 1 or p.data_parallel_size != 1:
        raise RuntimeError('PP/DP unsupported')
    if p.all2all_backend != 'allgather_reducescatter':
        raise RuntimeError('Only gathered Ag/Rs path verified by supplied capture checks')
    if vllm_config.model_config.quantization is not None:
        raise RuntimeError('Unquantized BF16 checkpoint required')
    if not vllm_config.model_config.enforce_eager:
        raise RuntimeError('Use enforce-eager for this controlled study; graph support is not claimed')


class CWHV2Qwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config, prefix=''):
        check_environment(vllm_config)
        self.policies = read_policy(os.environ['CWH_POLICY'])
        self.mode = os.getenv('CWH_MODE', 'ratio')
        self.backend = os.getenv('CWH_BACKEND', 'triton')
        if self.mode not in ('cw', 'ratio', 'shadow', 'legacy', 'legacy_shadow','dynamic','cost','cost_shadow'):
            raise ValueError(self.mode)
        self.fraction = os.getenv('CWH_FRACTION')
        if self.fraction is not None:
            self.fraction = float(self.fraction)
            if not 0 <= self.fraction <= 1:
                raise ValueError('CWH_FRACTION out of range')
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._setup()

    def _setup(self):
        if self.num_moe_layers != 48 or self.num_logical_experts != 128:
            raise RuntimeError('Require Qwen3-30B-A3B 48 layers/128 experts')
        if self.num_redundant_experts or self.num_local_physical_experts != 32:
            raise RuntimeError('Exactly 128 global / 32 local physical slots required')
        self._capture = None
        if os.getenv('CWH_CAPTURE_DIR'):
            from .capture import Capture
            self._capture = Capture(self.moe_layers[0].ep_rank)
        for l, experts in enumerate(self.moe_layers):
            if experts.ep_size != 4 or not experts.use_ep or experts._expert_map is None:
                raise RuntimeError('Missing EP4 map')
            actual = experts._expert_map.cpu().tolist()
            expected = [e % 32 if e // 32 == experts.ep_rank else -1 for e in range(128)]
            if actual != expected:
                raise RuntimeError('Require exact LINEAR EP4 slot ownership, not just H/W/C membership')
            # Disabled means true native layer: no C overwrite and no callback.
            if self.policies[l]['enabled'] or self._capture is not None:
                experts.custom_routing_function = self._route(l)

    def _route(self, l):
        cfg = self.policies[l]
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
        from .routing import redirect
        calls = 0
        def route(hidden_states, gating_output, topk, renormalize):
            nonlocal calls
            calls += 1
            result = fused_topk(hidden_states=hidden_states, gating_output=gating_output,
                                topk=topk, renormalize=renormalize)
            weights, ids = result[:2]
            output = redirect(ids, gating_output, cfg, self.mode, self.backend,
                              self.fraction, calls) if cfg['enabled'] else ids
            if self._capture is not None:
                self._capture(l, calls, hidden_states, ids, weights, output)
            return weights, output
        if os.getenv('VLLM_TORCH_PROFILER_DIR'):
            def profiled(*args, **kwargs):
                with torch.profiler.record_function(f'CWH.route.layer{l}'):
                    return route(*args, **kwargs)
            return profiled
        return route

    def load_weights(self, weights):
        audit = Counter()
        result = super().load_weights(remap_weights(weights, self.policies, audit))
        validate_audit(audit, self.policies)
        rank = self.moe_layers[0].ep_rank
        rows = []
        for l, experts in enumerate(self.moe_layers):
            cfg = self.policies[l]
            if not cfg['enabled']:
                continue
            for kind, expert in (('H', cfg['hot']), ('replica', cfg['cold'])):
                if expert // 32 != rank:
                    continue
                for pname in ('w13_weight', 'w2_weight'):
                    value = getattr(experts, pname)[expert % 32].detach().cpu().contiguous()
                    digest = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
                    rows.append(dict(layer=l, kind=kind, parameter=pname,
                                     shape=list(value.shape), sha256=digest))
        out = Path(os.environ['CWH_AUDIT_DIR'])
        out.mkdir(parents=True, exist_ok=True)
        (out / f'rank{rank}.json').write_text(json.dumps(dict(
            rank=rank, active_layers=sum(c['enabled'] for c in self.policies.values()),
            physical_slots_per_layer=128, tensor_hashes=rows,
            policy_sha256=hashlib.sha256(Path(os.environ['CWH_POLICY']).read_bytes()).hexdigest(),
            audit={str(k):v for k,v in audit.items()}), indent=2))
        return result


class CWHProbeQwen3MoeForCausalLM(CWHV2Qwen3MoeForCausalLM):
    """Native weights/native IDs plus real-request capture; never a timed baseline."""
    def __init__(self, *, vllm_config, prefix=''):
        check_environment(vllm_config)
        self.policies = {l: dict(layer=l, enabled=False) for l in range(48)}
        self.mode, self.backend, self.fraction = 'cw', 'torch', None
        Qwen3MoeForCausalLM.__init__(self, vllm_config=vllm_config, prefix=prefix)
        self._setup()


class CWHIdentityQwen3MoeForCausalLM(CWHProbeQwen3MoeForCausalLM):
    """Native weights and IDs with the custom fused_topk callback, no capture."""
    def _setup(self):
        super()._setup()
        if self._capture is not None: raise RuntimeError('Identity control must not capture')
        for l, experts in enumerate(self.moe_layers):
            experts.custom_routing_function = self._route(l)
