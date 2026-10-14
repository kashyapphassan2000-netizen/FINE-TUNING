import json

import torch
from peftkit.hardware import Shape, adapter_params, count_params, estimate_serve, estimate_train, recommend
from transformers import AutoModelForCausalLM


def test_param_count_matches_real_model(tiny_model):
    shape = Shape.from_config(json.loads((tiny_model / "config.json").read_text()))
    real = sum(p.numel() for p in AutoModelForCausalLM.from_pretrained(tiny_model).parameters())
    assert count_params(shape)["total"] == real


def test_published_sizes():
    s8 = Shape.from_config(json.load(open("configs/models/llama-3.1-8b.config.json")))
    s70 = Shape.from_config(json.load(open("configs/models/llama-3.1-70b.config.json")))
    assert abs(count_params(s8)["total"] / 1e9 - 8.03) < 0.01
    assert abs(count_params(s70)["total"] / 1e9 - 70.55) < 0.05


def test_adapter_params_match_peft(tiny_model):
    from peft import AdaLoraConfig, LoraConfig, get_peft_model

    shape = Shape.from_config(json.loads((tiny_model / "config.json").read_text()))
    cases = [("lora", 8, LoraConfig(r=8, target_modules="all-linear")),
             ("dora", 8, LoraConfig(r=8, target_modules="all-linear", use_dora=True)),
             ("adalora", 12, AdaLoraConfig(init_r=12, target_r=8, target_modules="all-linear", total_step=100,
                                           tinit=10, tfinal=20))]
    for method, r, pc in cases:
        m = get_peft_model(AutoModelForCausalLM.from_pretrained(tiny_model), pc)
        assert adapter_params(shape, r, "all-linear", method) == m.get_nb_trainable_parameters()[0], method


def test_estimates_are_ordered():
    s = Shape.from_config(json.load(open("configs/models/llama-3.1-8b.config.json")))
    lora, qlora = estimate_train(s, "lora"), estimate_train(s, "qlora", optim_8bit=True)
    assert qlora.total_gb < lora.total_gb
    assert estimate_train(s, "lora", seq_len=8192).activations_gb > lora.activations_gb
    assert estimate_serve(s)["kv_cache_per_token_mb"] == 0.125  # 2*32*8*128*2 bytes
    assert recommend(20)[0].startswith("1x") and recommend(10_000) == []
    assert torch.tensor(1)  # keep torch import used
