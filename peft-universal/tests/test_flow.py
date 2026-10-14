"""The four flow tools: openness/licence, GPU stack plan, documents->JSONL builder, HPO."""
import json

import pytest
from peftkit import gpu_stack, hpo, openness


def test_openness_decides_correctly():
    ok = openness.assess({"id": "a", "license": "apache-2.0", "weight_files": ["m.safetensors"]}, "commercial")
    assert ok["decision"] == "PROCEED" and ok["open_weights"] and ok["commercial_use"]
    nc = openness.assess({"id": "b", "license": "cc-by-nc-4.0", "weight_files": ["m.safetensors"]}, "commercial")
    assert nc["decision"] == "STOP" and "commercial" in nc["blockers"][0]
    assert openness.assess({"id": "b", "license": "cc-by-nc-4.0", "weight_files": ["m"]}, "research")["decision"] == "PROCEED"
    closed = openness.assess({"id": "c", "license": None, "weight_files": []}, "commercial")
    assert closed["decision"] == "STOP" and "not open-weights" in closed["blockers"][0]
    gated = openness.assess({"id": "d", "license": "llama3.1", "gated": True, "weight_files": ["m"]}, "commercial")
    assert gated["decision"] == "PROCEED" and any("GATED" in a for a in gated["actions"])
    assert any("monthly-active-user" in a for a in gated["actions"])          # community licence duty surfaced
    assert openness.normalise("Apache-2.0") == "apache-2.0" and openness.normalise(None) == "unknown"
    assert openness.normalise("bigscience-openrail-m") == "openrail"
    assert "not legal advice" in gated["disclaimer"]
    unreadable = openness.fetch("definitely/not-a-real-model-xyz")
    assert unreadable["error"]                                                 # offline or missing: reported
    assert openness.assess(unreadable)["decision"] == "UNKNOWN"                # not a false STOP


def test_gpu_stack_plan_is_capability_aware():
    t4 = gpu_stack.plan("T4", {})
    assert t4["gpu"] == "T4" and not t4["spec"]["bf16"]
    assert any("no bf16" in w for w in t4["warnings"]) and any("FlashAttention-2" in w for w in t4["warnings"])
    l40 = gpu_stack.plan("L40S", {})
    assert l40["gpu"] == "L40S" and l40["spec"]["vram"] == 48          # not matched to "L4"
    h100 = gpu_stack.plan("H100", {})
    assert h100["spec"]["fp8"] and not h100["warnings"]
    b200 = gpu_stack.plan("B200", {})
    assert any("very new architecture" in w for w in b200["warnings"])
    assert any("unknown GPU" in w for w in gpu_stack.plan("SomethingNew", {})["warnings"])
    names = [s[0] for s in h100["steps"]]
    assert names[0] == "driver + CUDA" and "serving env (separate!)" in names and names[-1] == "verify"
    d = gpu_stack.detect()
    assert "torch" in d and d["cuda_available"] in (True, False)


def test_hpo_search_space_and_scoring():
    space = {"train.lr": [1e-4, 2e-4], "method.r": [8, 16]}
    picks = hpo.sample_space(space, 4, seed=0)
    assert len(picks) == 4 and len({json.dumps(p, sort_keys=True) for p in picks}) == 4
    assert hpo.nest({"train.lr": 1e-4, "method.r": 8}) == {"train": {"lr": 1e-4}, "method": {"r": 8}}
    assert hpo.score_of({"eval": {"eval_loss": 1.5}, "train_loss": 0.1}) == 1.5
    assert hpo.score_of({"train_loss": 0.1}) > 1e5          # train loss can never beat a real eval score


def test_hpo_runs_successive_halving(tiny_model, sft_files, tmp_path):
    extra = {"model": {"name_or_path": str(tiny_model), "max_seq_length": 64},
             "data": {"train_file": str(sft_files / "train.jsonl"), "eval_file": str(sft_files / "eval.jsonl")},
             "train": {"per_device_batch_size": 8, "grad_accum": 1, "logging_steps": 50}}
    res = hpo.search(["configs/base.yaml", "configs/methods/lora.yaml"],
                     {"train.lr": [1e-4, 5e-3], "method.r": [4, 8]}, budget=4, out_dir=str(tmp_path / "hpo"),
                     steps_per_rung=[3, 6], extra=extra)
    assert len(res["candidates"]) == 4
    survivors = [c for c in res["candidates"] if len(c["history"]) == 2]
    assert 1 <= len(survivors) <= 2                                  # half were killed after the first rung
    assert res["best"]["score"] == min(c["score"] for c in res["candidates"])
    assert (tmp_path / "hpo" / "hpo.json").exists()
    noise = hpo.seed_noise(["configs/base.yaml", "configs/methods/lora.yaml"], res["best"]["params"],
                           seeds=(0, 1), steps=3, extra=extra, out_dir=str(tmp_path / "seeds"))
    assert len(noise["scores"]) == 2 and noise["spread"] >= 0 and "noise" in noise["rule"]
