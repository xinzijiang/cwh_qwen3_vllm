def register():
    """Register a NEW architecture key without overriding upstream Qwen3."""
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "CWHQwen3MoeForCausalLM", "cwh_vllm.model:CWHQwen3MoeForCausalLM"
    )
