"""Version-independent, CPU-testable policy validation and weight stream aliasing."""
import json
import os
import re
from collections import Counter
from pathlib import Path

EXPERT_NAME = re.compile(
    r"(?P<prefix>(?:^|\.)layers\.(?P<layer>\d+)\.mlp\.experts\.)(?P<expert>\d+)"
    r"(?P<suffix>\.(?:gate_proj|up_proj|down_proj)\.weight)$"
)


def read_policy(path):
    obj = json.loads(Path(path).read_text())
    entries = obj.get("layers")
    if not isinstance(entries, list) or len(entries) != 48:
        raise ValueError("Qwen3-30B-A3B needs exactly 48 entries in policy['layers']")
    policies = {}
    for item in entries:
        layer, cold, warm, hot = [int(item[k]) for k in ("layer", "cold", "warm", "hot")]
        if layer in policies or not 0 <= layer < 48:
            raise ValueError("Layer indices must be exactly 0..47 without duplicates")
        if not (0 <= hot < 32 and 32 <= warm < 64 and 64 <= cold < 96):
            raise ValueError("For linear EP4: H must be rank0, W rank1, C rank2")
        if len({hot,warm,cold}) != 3:
            raise ValueError("H/W/C must be distinct")
        policies[layer] = dict(item)
    if set(policies) != set(range(48)):
        raise ValueError("Missing Qwen3 MoE layers")
    return policies


def remap_weights(weights, policies, audit):
    """Replace EACH original C weight tensor with a *true* H tensor, during load.

    The destination C's existing physical EP slot receives H's weights, not an
    additional buffer. The original C weights are omitted. This does not
    instantiate or copy a second parameter. Upstream loader owns all GPU copies.
    Checkpoint expert naming must match Qwen3 BF16 (3 matrices / expert).
    """
    for name, tensor in weights:
        match = EXPERT_NAME.search(name)
        if match is None:
            yield name, tensor
            continue
        layer, expert = int(match["layer"]), int(match["expert"])
        if layer not in policies:
            raise RuntimeError(f"Unexpected expert layer {layer}")
        cfg = policies[layer]
        if expert == int(cfg["cold"]):
            audit[(layer,"cold_skipped")] += 1
            continue
        yield name, tensor
        if expert == int(cfg["hot"]):
            # The vLLM weight loader maps the synthetic C name into its local
            # resident slot on EP rank 2. On other ranks, the expert map skips it.
            new_name = (name[:match.start("expert")] + str(cfg["cold"]) +
                        name[match.end("expert"):])
            audit[(layer,"hot_aliased_into_c")] += 1
            yield new_name, tensor


def validate_audit(audit, num_layers=48):
    for layer in range(num_layers):
        for key in ("cold_skipped", "hot_aliased_into_c"):
            n = audit[(layer,key)]
            if n != 3:
                raise RuntimeError(
                    f"Incomplete Qwen3 BF16 weight replacement layer={layer} "
                    f"{key}={n} != 3. Abort rather than use uninitialized weights."
                )



def apply_expert_redirects(topk_ids, router_logits, cold, warm, hot, *, replicate=True, layer_id=None):
    """C->W and optionally H->C. Optional trace only for real, marked requests."""
    import json
    import os
    import torch

    original = topk_ids
    chosen_c = original == cold
    new_ids = torch.where(chosen_c, warm, original)
    replicate = bool(replicate) and os.getenv("CWH_REPLICATE", "1") != "0"
    replica_mask = torch.zeros_like(chosen_c)
    if replicate:
        e = router_logits.shape[-1]
        a, b = (hot + 13) % e, (hot + 37) % e
        shard_mask = (router_logits[:, a] > router_logits[:, b]).unsqueeze(1)
        replica_mask = (original == hot) & shard_mask
        new_ids = torch.where(replica_mask, cold, new_ids)

    # Expensive synchronization only in a separate TRACE experiment.
    # The benchmark creates this marker ONLY after server readiness/warmup.
    marker = os.getenv("CWH_TRACE_MARKER", "")
    if os.getenv("CWH_TRACE_ROUTING", "0") == "1" and marker and os.path.isfile(marker):
        layer_key = int(layer_id) if layer_id is not None else -1
        counts = getattr(apply_expert_redirects, "_postready_counts", None)
        if counts is None:
            counts = {}
            apply_expert_redirects._postready_counts = counts
        call_num = counts.get(layer_key, 0) + 1
        counts[layer_key] = call_num
        stride = max(1, int(os.getenv("CWH_TRACE_STRIDE", "128")))
        cap = max(1, int(os.getenv("CWH_TRACE_MAX_SAMPLES_LAYER", "100")))
        sample_no = 1 + (call_num - 1) // stride
        if (call_num == 1 or call_num % stride == 0) and sample_no <= cap:
            # All numbers are selected-expert *assignments*, not unique tokens.
            # GPU load = estimated number of assignments by linear expert ID
            # placement (0..31 / 32..63 / 64..95 / 96..127).
            sorted_new = torch.sort(new_ids, dim=-1).values
            duplicate_token_count = int((sorted_new[:, 1:] == sorted_new[:, :-1]).any(dim=1).sum().item())
            before_gpu = torch.bincount(
                (original.reshape(-1).long() // 32), minlength=4).cpu().tolist()
            after_gpu = torch.bincount(
                (new_ids.reshape(-1).long() // 32), minlength=4).cpu().tolist()
            stats = {
                "pid": os.getpid(), "layer": layer_key, "call": call_num,
                "cold": int(cold), "warm": int(warm), "hot": int(hot),
                "c_selected": int(chosen_c.sum().item()),
                "h_selected": int((original == hot).sum().item()),
                "h_to_replica": int(replica_mask.sum().item()),
                "h_remaining": int((new_ids == hot).sum().item()),
                "duplicate_token_count": duplicate_token_count,
                "token_rows": int(new_ids.shape[0]),
                "before_gpu_assignment_counts": before_gpu,
                "after_gpu_assignment_counts": after_gpu,
                "source": "post-server-ready trace marker",
            }
            print("CWH_POSTREADY_TRACE " + json.dumps(stats, sort_keys=True), flush=True)
    return new_ids


