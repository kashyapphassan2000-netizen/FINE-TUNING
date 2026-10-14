import json

import pytest
import torch
from peftkit.compat import check as compat_check
from peftkit.config import load_config
from peftkit.diagnostics import doctor
from peftkit.evaluate import compare, evaluate, score_item
from peftkit.experiments import as_markdown, expand, run_grid
from peftkit.tokenize_inspect import render, table
from transformers import AutoTokenizer

MSGS = [{"role": "user", "content": "What is the maximum pressure?"},
        {"role": "assistant", "content": "6 bar."}]


def test_compat_decoder_only(tiny_model):
    r = compat_check(str(tiny_model))
    assert r["verdict"] == "SUPPORTED" and r["kind"] == "decoder-only"
    assert "q_proj" in r["target_modules_attention"] and "gate_proj" in r["target_modules_all"]
    assert r["lora_trainable_all_linear"] == 32768   # r=16 default in compat; r=8 would be 16384
    assert r["tokenizer"]["generation_markers"] is True
    assert r["config_snippet"]["method"]["name"] == "lora"


def test_compat_rejects_other_families(tmp_path):
    (tmp_path / "t5").mkdir()
    (tmp_path / "t5" / "config.json").write_text(json.dumps({
        "model_type": "t5", "architectures": ["T5ForConditionalGeneration"], "is_encoder_decoder": True,
        "d_model": 8, "num_layers": 2, "num_heads": 2, "vocab_size": 32}))
    r = compat_check(str(tmp_path / "t5"), check_tokenizer=False)
    assert r["verdict"] == "OTHER-PIPELINE" and "encoder-decoder" in r["blockers"][0]
    (tmp_path / "vl").mkdir()
    (tmp_path / "vl" / "config.json").write_text(json.dumps({
        "model_type": "llava", "architectures": ["LlavaForConditionalGeneration"], "vision_config": {"x": 1},
        "text_config": {"model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1,
                        "num_attention_heads": 2, "vocab_size": 32}}))
    assert compat_check(str(tmp_path / "vl"), check_tokenizer=False)["verdict"] == "OTHER-PIPELINE"


def test_tokenize_inspect_shows_mask(tiny_model):
    tok = AutoTokenizer.from_pretrained(tiny_model)
    t = table(tok, MSGS)
    assert 0 < t["trained_tokens"] < t["n_tokens"] and t["eos_is_trained"]
    trained = [r["token"] for r in t["rows"] if r["trained_on"]]
    assert any("bar" in x for x in trained)
    assert all(r["label"] == -100 for r in t["rows"] if not r["trained_on"])
    text = render(t, limit=5)
    assert "label -100" in text and "trained on" in text
    cut = table(tok, MSGS, max_length=5)
    assert sum(r["dropped_by_truncation"] for r in cut["rows"]) == cut["n_tokens"] - 5


def test_doctor_catches_planted_problems(tiny_model, tmp_path):
    rows = [{"messages": MSGS}] * 30
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "eval.jsonl").write_text(json.dumps({"messages": MSGS}))  # leaked: identical to train
    cfg = load_config("configs/base.yaml", "configs/methods/lora.yaml", overrides={
        "model": {"name_or_path": str(tiny_model), "max_seq_length": 8},          # forces truncation
        "method": {"target_modules": ["does_not_exist"]},
        "data": {"train_file": str(tmp_path / "train.jsonl"), "eval_file": str(tmp_path / "eval.jsonl"),
                 "assistant_only_loss": True},
        "train": {"lr": 2e-5, "epochs": 1, "per_device_batch_size": 8, "grad_accum": 1}})
    res = {r.name: r for r in doctor(cfg)}
    assert res["train/eval leakage"].status == "FAIL"
    assert res["truncation"].status == "FAIL"
    assert res["target_modules exist"].status == "FAIL"
    assert res["learning rate"].status == "WARN"
    assert res["duplicates in train"].status == "WARN"
    assert "too low for LoRA" in res["learning rate"].fix


def test_doctor_passes_a_good_config(tiny_model, tmp_path):
    # a clean split: eval questions do not appear in train (the shared fixture deliberately repeats rows)
    train = [{"messages": [{"role": "user", "content": f"Question number {i} about the pump?"},
                           {"role": "assistant", "content": f"Answer number {i}."}]} for i in range(60)]
    ev = [{"messages": [{"role": "user", "content": f"Held-out question {i}?"},
                        {"role": "assistant", "content": f"Held-out answer {i}."}]} for i in range(10)]
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train))
    (tmp_path / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in ev))
    cfg = load_config("configs/base.yaml", "configs/methods/lora.yaml", overrides={
        "model": {"name_or_path": str(tiny_model), "max_seq_length": 128},
        "data": {"train_file": str(tmp_path / "train.jsonl"), "eval_file": str(tmp_path / "eval.jsonl"),
                 "assistant_only_loss": True},
        "train": {"lr": 2e-4, "epochs": 3, "per_device_batch_size": 8, "grad_accum": 1}})
    res = {r.name: r.status for r in doctor(cfg)}
    assert "FAIL" not in res.values()
    assert res["loss mask"] == "PASS" and res["EOS is trained"] == "PASS" and res["trainable params"] == "PASS"


def test_scoring_rules():
    item = {"expected": "6 bar", "must_include": ["6 bar"], "must_match": r"^\d"}
    assert score_item("6 bar", item)["score"] == 1.0
    bad = score_item("about twenty bar", item)
    assert bad["score"] == 0.0 and bad["format_ok"] == 0.0
    partial = score_item("The answer is 6 bar.", {"expected": "6 bar"})
    assert partial["exact"] == 0.0 and partial["contains_gold"] == 1.0


def test_evaluate_and_compare_verdicts(tiny_model, tmp_path):
    suite = tmp_path / "domain.jsonl"
    suite.write_text("\n".join(json.dumps({"id": f"d{i}", "messages": [MSGS[0]], "expected": "6 bar"})
                               for i in range(2)))
    rep = evaluate(str(tiny_model), {"domain": str(suite)}, max_new_tokens=4)
    assert rep["suites"]["domain"]["n"] == 2 and 0.0 <= rep["overall"] <= 1.0
    assert rep["suites"]["domain"]["mean_answer_nll"] > 0
    base = {"suites": {"domain": {"score": 0.2, "mean_answer_nll": 3.0,
                                  "items": [{"id": "a", "score": 0.2}, {"id": "b", "score": 1.0}]},
                       "general": {"score": 0.9, "mean_answer_nll": 2.0, "items": [{"id": "g", "score": 0.9}]}}}
    good = {"suites": {"domain": {"score": 0.8, "mean_answer_nll": 1.0,
                                  "items": [{"id": "a", "score": 0.8, "prediction": "x"},
                                            {"id": "b", "score": 0.5, "prediction": "worse"}]},
                       "general": {"score": 0.9, "mean_answer_nll": 2.0, "items": [{"id": "g", "score": 0.9}]}}}
    c = compare(base, good)
    assert c["verdict"].startswith("PASS") and c["n_regressions"] == 1 and c["regressions"][0]["id"] == "b"
    forgot = json.loads(json.dumps(good))
    forgot["suites"]["general"]["score"] = 0.5
    forgot["suites"]["general"]["items"] = [{"id": "g", "score": 0.5, "prediction": "?"}]
    assert "catastrophic forgetting" in compare(base, forgot)["verdict"]
    worse = json.loads(json.dumps(base))
    assert compare(base, worse)["verdict"].startswith("FAIL")


def test_experiment_grid_runs_and_tabulates(tiny_model, sft_files, tmp_path):
    assert expand({"method.r": [4, 8], "train.lr": [1e-3]}) == [
        {"method": {"r": 4}, "train": {"lr": 1e-3}}, {"method": {"r": 8}, "train": {"lr": 1e-3}}]
    rows = run_grid(["configs/base.yaml", "configs/methods/lora.yaml"], {"method.r": [4, 8]}, str(tmp_path / "exp"),
                    extra={"model": {"name_or_path": str(tiny_model), "max_seq_length": 64},
                           "data": {"train_file": str(sft_files / "train.jsonl"),
                                    "eval_file": str(sft_files / "eval.jsonl")},
                           "train": {"max_steps": 3, "per_device_batch_size": 8, "grad_accum": 1,
                                     "logging_steps": 3, "save_steps": 1000, "eval_steps": 3}})
    assert len(rows) == 2 and rows[0]["trainable"] * 2 == rows[1]["trainable"]
    assert all(r["adapter_mb"] > 0 and r["steps"] == 3 for r in rows)
    assert len({r["run_id"] for r in rows}) == 2
    assert (tmp_path / "exp" / "experiments.csv").exists()
    md = as_markdown(rows)
    assert md.startswith("| run_id") and rows[0]["run_id"] in md
    assert torch.tensor(1) == 1


def test_kv_calculator_matches_hand_math(tiny_model):
    from peftkit.kv import report

    r = report(str(tiny_model), context=128, gpu_gb=1, gpus=1, kv_dtype="bf16", weight_gb=0.0)
    # 2 (K,V) x layers x kv_heads x head_dim x bytes
    assert r["kv_bytes_per_token"] == 2 * 2 * 2 * 16 * 2
    assert r["max_concurrent_sequences_at_context"] == r["max_cached_tokens"] // 128
    v = r["variant_bytes_per_token"]
    assert v["MHA (kv_heads = heads)"] == 2 * v["GQA (kv_heads = 2)"]        # 4 heads vs 2 kv heads
    assert v["MQA (kv_heads = 1)"] * 2 == v["GQA (kv_heads = 2)"]


def test_kv_calculator_on_mla_and_gqa_configs():
    from peftkit.kv import report

    ds = report("configs/models/deepseek-v3.config.json", context=4096, gpu_gb=80, gpus=8,
                kv_dtype="bf16", weight_gb=700.0)
    assert ds["attention"] == "mla"
    assert ds["kv_bytes_per_token"] == 61 * (512 + 64) * 2
    ll = report("configs/models/llama-3.1-8b.config.json", context=8192, gpu_gb=80, gpus=1,
                kv_dtype="fp8", weight_gb=16.0)
    assert ll["kv_bytes_per_token"] == 2 * 32 * 8 * 128 * 1        # fp8 halves bf16
    assert ll["max_concurrent_sequences_at_context"] > 0
