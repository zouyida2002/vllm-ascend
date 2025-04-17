from vllm import ModelRegistry


def register_model():
    from .qwen2_vl import CustomQwen2VLForConditionalGeneration  # noqa: F401

    ModelRegistry.register_model(
        "Qwen2VLForConditionalGeneration",
        "vllm_ascend.models.qwen2_vl:CustomQwen2VLForConditionalGeneration")

    ModelRegistry.register_model(
        "Qwen2_5_VLForConditionalGeneration",
        "vllm_ascend.models.qwen2_5_vl:CustomQwen2_5_VLForConditionalGeneration")

    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",
        "vllm_ascend.models.deepseek_v2:CustomDeepseekV2ForCausalLM")

    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM",
        "vllm_ascend.models.deepseek_v2:CustomDeepseekV3ForCausalLM")
