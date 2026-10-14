import pytest
from peftkit.config import MethodCfg
from peftkit.peft_factory import adalora_schedule, build_peft_config, optimizer_cls_for, total_optimizer_steps


def test_method_mapping():
    assert build_peft_config(MethodCfg(name="lora")).use_dora is False
    assert build_peft_config(MethodCfg(name="dora")).use_dora is True
    assert build_peft_config(MethodCfg(name="loraplus")).peft_type.value == "LORA"
    ada = build_peft_config(MethodCfg(name="adalora"), total_steps=1000)
    assert (ada.tinit, ada.tfinal, ada.total_step) == (100, 200, 1000)
    with pytest.raises(ValueError):
        build_peft_config(MethodCfg(name="adalora"))


def test_schedule_guard():
    with pytest.raises(ValueError):
        adalora_schedule(5, MethodCfg().adalora.model_copy(update={"tinit_ratio": 0.6, "tfinal_ratio": 0.5}))


def test_steps_math():
    assert total_optimizer_steps(1000, 4, 4, 1, -1) == 63       # ceil(ceil(1000/4)/4)
    assert total_optimizer_steps(1000, 4, 4, 2, -1, world_size=2) == 64   # 125 batches -> 32 per epoch x 2
    assert total_optimizer_steps(10, 4, 4, 3, 50) == 50
    import torch
    assert optimizer_cls_for("adamw_torch") is torch.optim.AdamW
