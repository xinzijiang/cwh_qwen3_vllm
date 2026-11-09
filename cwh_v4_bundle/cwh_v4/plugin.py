def register():
    from vllm import ModelRegistry
    for name in ('CWHV2Qwen3MoeForCausalLM', 'CWHProbeQwen3MoeForCausalLM', 'CWHIdentityQwen3MoeForCausalLM'):
        ModelRegistry.register_model(name, 'cwh_v4.model:' + name)
    ModelRegistry.register_model('CWHV1Qwen3MoeForCausalLM',
                                 'cwh_v1_reference.model:CWHV1Qwen3MoeForCausalLM')
