"""Research-only out-of-tree Qwen3-MoE model for vLLM 0.13.0.

REAL OPERATION on 4-way linear EP / BF16 Qwen3-30B-A3B:
  * skip original C tensors while loading;
  * load H's original checkpoint weights into C's ALREADY ALLOCATED local slot;
  * after normal top-k, redirect ALL C assignments to W (original router weight);
  * redirect some H assignments to the C slot, now physically holding H.

This is static load-time slot reuse and token dispatch (not online hot switching,
not actual cudaFree/malloc, and not a model accuracy improvement). Only compare
against native EP with identical model+inputs and no EPLB.
"""
import json
import os
import logging
from collections import Counter
from pathlib import Path

from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

from .policy import apply_expert_redirects, read_policy, remap_weights, validate_audit

log = logging.getLogger(__name__)


class CWHQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        self._cwh_policy_path = os.environ.get("CWH_POLICY", "")
        if not self._cwh_policy_path:
            raise RuntimeError("CWH_POLICY is mandatory for custom Qwen3 model")
        self._cwh_policy = read_policy(self._cwh_policy_path)
        from importlib.metadata import version
        actual = version("vllm")
        if not actual.startswith("0.13.0"):
            raise RuntimeError(f"Pinned to vLLM 0.13.0; got {actual}. Do not silently run on unverified API.")
        parallel = vllm_config.parallel_config
        if parallel.enable_eplb:
            raise RuntimeError("Custom CWH and vLLM EPLB must NOT be enabled together")
        if not parallel.enable_expert_parallel:
            raise RuntimeError("Must use --enable-expert-parallel")
        if parallel.tensor_parallel_size != 4:
            raise RuntimeError("Expected --tensor-parallel-size 4")
        if parallel.pipeline_parallel_size != 1:
            raise RuntimeError("Pipeline parallelism not supported by this prototype")
        if vllm_config.model_config.quantization is not None:
            raise RuntimeError("Use unquantized BF16 only for this prototype")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if self.num_moe_layers != 48 or self.num_logical_experts != 128:
            raise RuntimeError("This code is specific to 48-layer / 128-expert Qwen3-30B-A3B")
        if self.num_redundant_experts:
            raise RuntimeError("No extra redundant experts: this prototype reuses a C slot")
        if self.num_local_physical_experts != 32:
            raise RuntimeError("EP4 must own 32 physical expert slots per layer")
        for l, experts in enumerate(self.moe_layers):
            if not experts.use_ep or experts.ep_size != 4:
                raise RuntimeError(f"Layer {l}: actual expert-parallel group must have 4 ranks")
            if experts._expert_map is None:
                raise RuntimeError("Expert linear ownership map missing")
            # Confirm the expected physical slot layout; fail if round-robin
            # or different backend mapping would invalidate weight replacement.
            c,w,h = (int(self._cwh_policy[l][k]) for k in ("cold","warm","hot"))
            ep_rank = experts.ep_rank
            for global_id, owner in ((c,2),(w,1),(h,0)):
                got = int(experts._expert_map[global_id].item())
                if (got >= 0) != (ep_rank == owner):
                    raise RuntimeError("Unexpected expert_map; require linear EP=4")
            experts.custom_routing_function = self._routing_function(l)
        log.warning("CWH ACTIVE: static C->W / overwritten-C-slot->H replica, %d layers",len(self.moe_layers))

    def _routing_function(self, layer):
        cfg = self._cwh_policy[layer]
        cold,warm,hot = (int(cfg[k]) for k in ("cold","warm","hot"))
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
        def route(hidden_states, gating_output, topk, renormalize):
            # Invoke upstream fused_topk so Qwen3's top-k probabilities remain
            # precisely the original values. Only its selected IDs change.
            result = fused_topk(
                hidden_states=hidden_states,
                gating_output=gating_output,
                topk=topk,
                renormalize=renormalize,
            )
            topk_weights, topk_ids = result[:2]
            ids = apply_expert_redirects(topk_ids, gating_output, cold, warm, hot, layer_id=layer)
            return topk_weights, ids
        return route

    def load_weights(self, weights):
        audit=Counter()
        output = super().load_weights(remap_weights(weights, self._cwh_policy, audit))
        validate_audit(audit)
        # Each source checkpoint H MLP matrix was streamed into BOTH: its
        # original EP0 owner and cold slot (EP2). No C tensor is loaded.
        rank=os.environ.get("RANK", os.environ.get("LOCAL_RANK", str(os.getpid())))
        audit_dir=os.environ.get("CWH_AUDIT_DIR")
        summary={"rank":rank,"pid":os.getpid(),"vllm_model":"CWHQwen3MoeForCausalLM",
                 "layers":48,"physical_slots_per_ep_rank_per_layer":32,
                 "expert_weight_tensors_aliased":sum(audit[(l,"hot_aliased_into_c")] for l in range(48)),
                 "original_c_tensors_skipped":sum(audit[(l,"cold_skipped")] for l in range(48)),
                 "actual_cuda_free_called":False,"meaning":"C physical slot overwritten in-place via checkpoint loader"}
        if audit_dir:
            p=Path(audit_dir); p.mkdir(parents=True,exist_ok=True)
            (p/f"cwh_weights_rank{rank}_pid{os.getpid()}.json").write_text(json.dumps(summary,indent=2))
        log.warning("CWH LOAD AUDIT: %s",summary)
        return output
