import pytest
from peftkit.config import load_config, parse_cli_overrides

ROOT = "configs"


def test_merge_order_and_overrides():
    cfg = load_config(f"{ROOT}/base.yaml", f"{ROOT}/methods/dora.yaml", f"{ROOT}/models/llama-3.1-8b.yaml",
                      overrides=parse_cli_overrides(["train.lr=5e-5", "method.r=32"]))
    assert cfg.method.name == "dora"
    assert cfg.model.name_or_path == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg.train.lr == 5e-5 and cfg.method.r == 32
    assert cfg.train.grad_accum == 8  # from model file


def test_qlora_forces_4bit():
    cfg = load_config(f"{ROOT}/base.yaml", f"{ROOT}/methods/qlora.yaml")
    assert cfg.method.load_in_4bit and cfg.train.optim == "paged_adamw_8bit"


def test_validation_errors():
    with pytest.raises(ValueError):
        load_config(f"{ROOT}/base.yaml", f"{ROOT}/methods/adalora.yaml",
                    overrides={"method": {"adalora": {"init_r": 4, "target_r": 8}}})
    with pytest.raises(ValueError):
        load_config(f"{ROOT}/base.yaml", f"{ROOT}/methods/qlora.yaml", overrides={"model": {"dtype": "float32"}})
    with pytest.raises(ValueError):
        parse_cli_overrides(["no_equals_sign"])
