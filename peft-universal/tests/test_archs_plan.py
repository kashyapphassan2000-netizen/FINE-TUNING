import json

import pytest
import torch
from peftkit.archs import exact_counts, kv_bytes_per_token, spec_from_config
from peftkit.plan import plan
from transformers import AutoModelForCausalLM

CFG = "configs/models"


@pytest.mark.parametrize("name,total_b,active_b", [
    ("deepseek-v3", 671.0, 37.6), ("kimi-k2-like", 1026.4, 32.9), ("llama-3.1-405b", 405.9, 405.9),
    ("qwen3-235b-a22b", 235.1, 22.2), ("mixtral-8x7b", 46.7, 12.9), ("llama-3.1-8b", 8.03, 8.03)])
def test_exact_counts_match_published_sizes(name, total_b, active_b):
    c = exact_counts(json.load(open(f"{CFG}/{name}.config.json")), lora_r=None)
    assert abs(c["total"] / 1e9 - total_b) < 0.1
    assert abs(c["active"] / 1e9 - active_b) < 0.1


def test_exact_counts_equal_real_model(tiny_model):
    cfg = json.loads((tiny_model / "config.json").read_text())
    real = AutoModelForCausalLM.from_pretrained(tiny_model)
    c = exact_counts(cfg, lora_r=8)
    assert c["total"] == sum(p.numel() for p in real.parameters())
    assert c["lora_trainable_all_linear"] == 16384          # same as PEFT's count in training
    assert c["lora_trainable_attention"] < c["lora_trainable_all_linear"]


def test_attention_kinds_and_kv():
    ds = spec_from_config(json.load(open(f"{CFG}/deepseek-v3.config.json")))
    assert ds.attention == "mla" and ds.family == "moe"
    assert kv_bytes_per_token(ds, 4096) == 61 * (512 + 64) * 2          # compressed latent per layer
    ll = spec_from_config(json.load(open(f"{CFG}/llama-3.1-8b.config.json")))
    assert ll.attention == "gqa" and kv_bytes_per_token(ll, 4096) == 2 * 32 * 8 * 128 * 2
    sw = spec_from_config({"model_type": "x", "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4,
                           "num_key_value_heads": 2, "vocab_size": 10, "sliding_window": 100,
                           "layer_types": ["sliding_attention", "full_attention"] * 2})
    assert sw.attention == "sliding-mix"
    assert kv_bytes_per_token(sw, 1000) < kv_bytes_per_token(sw, 100)   # sliding layers stop growing
    hy = spec_from_config({"model_type": "x", "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4,
                           "vocab_size": 10, "layer_types": ["linear_attention"] * 3 + ["full_attention"]})
    assert hy.attention == "hybrid-linear" and hy.full_layers == 1
    nested = spec_from_config({"model_type": "vlm", "text_config": {"model_type": "llama", "hidden_size": 8,
                               "num_hidden_layers": 2, "num_attention_heads": 2, "vocab_size": 5}})
    assert nested.layers == 2
    q = spec_from_config({"model_type": "x", "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
                          "vocab_size": 5, "quantization_config": {"quant_method": "fp8"}})
    assert q.quantization == "fp8" and "dequantize" in q.notes[0]


def test_plan_trillion_moe_truths():
    p = plan(f"{CFG}/kimi-k2-like.config.json", ["lora", "qlora"], 4096, 1, 16, 2e7, 1, 500, 2.0, "fp8", 32768, 16)
    lora, qlora = p["training"]
    assert lora["total_gb"] > 1800 and "nodes" in lora["gpus"]["H100-80"]["layout"]
    # fused experts are not 4-bit quantized and get upcast: QLoRA is NOT smaller here
    assert qlora["weights_gb"] > lora["weights_gb"]
    p2 = plan(f"{CFG}/kimi-k2-like.config.json", ["qlora"], 4096, 1, 16, None, 1, None, None, "fp8", 4096, 1,
              targets="attention", upcast=False)
    assert p2["training"][0]["trainable"] < lora["trainable"] / 100
    assert p["compute"]["gpu_hours_per_gpu_rate"] == pytest.approx(2e7 / 500 / 3600)
    assert p["compute"]["cost"] == pytest.approx(2 * 2e7 / 500 / 3600)
    assert p["serving"]["per_gpu_type"]["B200-180"]["gpus"] >= 8


def test_plan_small_dense_fits_one_gpu():
    p = plan(f"{CFG}/smollm2-135m.config.json", ["lora"], 1024, 1, 16, 1e6, 1, None, None, "bf16", 4096, 4,
             gpu_flops=312e12, mfu=0.3)
    assert p["training"][0]["gpus"]["L4-24"]["n"] == 1
    assert p["compute"]["tokens_per_sec_derived"] == pytest.approx(0.3 * 312e12 / (6 * p["counts"]["active"]))
    assert torch.tensor(0) == 0


def test_light_kbit_prep_keeps_experts_bf16():
    from peftkit.modeling import light_kbit_prep
    from transformers import Qwen3MoeConfig

    tiny = Qwen3MoeConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                          num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32, intermediate_size=128,
                          vocab_size=100, head_dim=16)
    m = light_kbit_prep(AutoModelForCausalLM.from_config(tiny, dtype=torch.bfloat16), gradient_checkpointing=False)
    dt = {n: p.dtype for n, p in m.named_parameters()}
    assert dt["model.layers.0.mlp.experts.gate_up_proj"] == torch.bfloat16
    assert dt["model.layers.0.input_layernorm.weight"] == torch.float32
