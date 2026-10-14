import json

import pytest
import torch
from peftkit.config import load_config
from peftkit.generate import chat, load
from peftkit.merge import merge
from peftkit.serve_check import check
from peftkit.train import run

EXPECT_SERVABLE = {"lora": True, "loraplus": True, "dora": False, "adalora": False}


@pytest.mark.parametrize("method", ["lora", "dora", "adalora", "loraplus"])
def test_train_merge_servecheck(method, tiny_model, sft_files, tmp_path):
    cfg = load_config("configs/base.yaml", f"configs/methods/{method}.yaml", overrides={
        "model": {"name_or_path": str(tiny_model), "max_seq_length": 128},
        "method": {"r": 8},
        "data": {"train_file": str(sft_files / "train.jsonl"), "eval_file": str(sft_files / "eval.jsonl")},
        "train": {"output_dir": str(tmp_path / method), "epochs": 3, "per_device_batch_size": 8,
                  "grad_accum": 1, "lr": 5e-3, "logging_steps": 2, "eval_steps": 9, "save_steps": 1000},
    })
    meta = run(cfg)
    assert meta["global_step"] == 18
    losses = [h["loss"] for h in meta["log_history"] if "loss" in h]
    assert losses[-1] < losses[0], "training loss should go down"
    adapter = tmp_path / method / "adapter"
    assert (adapter / "adapter_model.safetensors").exists()
    assert check(str(adapter))["servable_as_adapter"] is EXPECT_SERVABLE[method]
    if method == "adalora":
        assert meta["adalora_final_active_rank"] <= 8 * 14  # budget = target_r x modules
    rep = merge(str(adapter), str(tmp_path / f"{method}-merged"), dtype="float32")
    assert rep["max_abs_logit_diff"] < 1e-4
    model, tok = load(str(tmp_path / f"{method}-merged"))
    assert isinstance(chat(model, tok, "What is the maximum pressure?", 5), str)
    saved = json.loads((tmp_path / method / "run_meta.json").read_text())
    assert saved["method"] == method and "versions" in saved


@pytest.mark.skipif(not torch.cuda.is_available(), reason="QLoRA needs an NVIDIA GPU + bitsandbytes")
def test_qlora_gpu(tiny_model, sft_files, tmp_path):
    cfg = load_config("configs/base.yaml", "configs/methods/qlora.yaml", overrides={
        "model": {"name_or_path": str(tiny_model), "max_seq_length": 128},
        "data": {"train_file": str(sft_files / "train.jsonl"), "eval_file": None},
        "train": {"output_dir": str(tmp_path / "q"), "max_steps": 5, "per_device_batch_size": 4, "grad_accum": 1},
    })
    assert run(cfg)["global_step"] == 5


def test_qlora_refuses_cpu(tiny_model, sft_files, tmp_path):
    if torch.cuda.is_available():
        pytest.skip("GPU present")
    cfg = load_config("configs/base.yaml", "configs/methods/qlora.yaml", overrides={
        "model": {"name_or_path": str(tiny_model)},
        "data": {"train_file": str(sft_files / "train.jsonl"), "eval_file": None},
        "train": {"output_dir": str(tmp_path / "q")},
    })
    with pytest.raises(RuntimeError, match="NVIDIA GPU"):
        run(cfg)
