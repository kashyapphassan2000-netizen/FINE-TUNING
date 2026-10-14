"""Additions from the second enterprise review: batch maths, benchmark gate, release gate with benchmarks,
standard-library script syntax, and the production configs (parsed by their own tools where installed)."""
import json
import py_compile
from pathlib import Path

import pytest
import yaml
from peftkit import batchmath, bench_gate, release_gate


def test_batch_maths_keeps_the_global_batch():
    one = batchmath.plan(16, 2, 1, 1)
    assert one["gradient_accumulation_steps"] == 8 and one["exact"]
    eight = batchmath.plan(16, 2, 8, 1)
    assert eight["gradient_accumulation_steps"] == 1 and eight["global_batch_achieved"] == 16
    big = batchmath.plan(256, 2, 8, 2)                      # 2 nodes x 8 H100
    assert big["world_size"] == 16 and big["gradient_accumulation_steps"] == 8 and big["exact"]
    too_many = batchmath.plan(16, 2, 8, 2)                  # cannot reach 16 with 16 GPUs at batch 2
    assert not too_many["exact"] and "lower per_device_batch to 1" in too_many["warning"]
    assert "sqrt" in too_many["lr_note"]


def _write(path: Path, results: dict):
    d = path / "org__model"
    d.mkdir(parents=True)
    (d / "results_2026-09-28T00-00-00.json").write_text(json.dumps({"results": results}))


def test_bench_gate_reads_lm_eval_format_and_blocks_forgetting(tmp_path):
    _write(tmp_path / "base", {"mmlu": {"alias": "mmlu", "acc,none": 0.62, "acc_stderr,none": 0.004},
                               "gsm8k": {"exact_match,strict-match": 0.55, "exact_match,flexible-extract": 0.57},
                               "hellaswag": {"acc,none": 0.60, "acc_norm,none": 0.79}})
    _write(tmp_path / "tuned", {"mmlu": {"acc,none": 0.615}, "gsm8k": {"exact_match,strict-match": 0.49},
                                "hellaswag": {"acc,none": 0.61, "acc_norm,none": 0.795}})
    r = bench_gate.compare(bench_gate.latest_results(tmp_path / "base"),
                           bench_gate.latest_results(tmp_path / "tuned"), max_drop=0.02)
    by = {x["task"]: x for x in r["rows"]}
    assert by["hellaswag"]["metric"] == "acc_norm,none"          # preferred metric chosen
    assert by["gsm8k"]["metric"] == "exact_match,strict-match" and not by["gsm8k"]["ok"]
    assert by["mmlu"]["ok"]                                      # -0.005 is inside the tolerance
    assert r["verdict"] == "FAIL" and r["failed"] == ["gsm8k"]
    with pytest.raises(FileNotFoundError):
        bench_gate.latest_results(tmp_path / "missing")


def test_release_gate_benchmark_check_is_optional_and_blocking():
    good = ({"verdict": "PASS: ok"}, {"verdict": "PASS"}, {"recommendation": {"meets_slo": True, "max_concurrency_per_replica": 8}})
    assert release_gate.decide(*good)["decision"] == "RELEASE"                     # unchanged without --bench
    assert release_gate.decide(*good, bench={"verdict": "PASS", "failed": []})["decision"] == "RELEASE"
    r = release_gate.decide(*good, bench={"verdict": "FAIL", "failed": ["gsm8k"]})
    assert r["decision"] == "BLOCK" and r["failed"] == ["benchmarks"] and "replay" in r["next"]["benchmarks"]


def test_standard_library_scripts_and_configs_are_valid():
    for f in ("examples/standard/train_sft_raw.py", "examples/standard/merge_publish_raw.py"):
        py_compile.compile(f, doraise=True)
        code = Path(f).read_text()
        assert "import peftkit" not in code and "from peftkit" not in code        # standard libraries only
    for f in ("configs/deepspeed/zero2.json", "configs/deepspeed/zero3.json"):
        z = json.loads(Path(f).read_text())["zero_optimization"]
        assert z["stage"] in (2, 3) and "offload_optimizer" in z
    assert json.loads(Path("configs/deepspeed/zero3.json").read_text())["zero_optimization"][
        "stage3_gather_16bit_weights_on_model_save"] is True
    for f in ("configs/accelerate/deepspeed_zero3.yaml", "configs/accelerate/multinode_fsdp2.yaml"):
        c = yaml.safe_load(Path(f).read_text())
        assert c["num_processes"] % c["num_machines"] == 0
    conf = Path("serving/nginx/nginx.conf").read_text()
    for needle in ("proxy_next_upstream error timeout http_502 http_503 http_504 non_idempotent",
                   "proxy_buffering off", "/.well-known/acme-challenge/", "limit_req_zone", "llm_json",
                   "listen 443 ssl http2"):
        assert needle in conf


def test_deepspeed_parses_our_zero_configs_when_installed():
    ds = pytest.importorskip("deepspeed.runtime.zero.config")
    for f in ("configs/deepspeed/zero2.json", "configs/deepspeed/zero3.json"):
        z = {k: (50_000_000 if v == "auto" else v) for k, v in json.loads(Path(f).read_text())["zero_optimization"].items()}
        assert int(ds.DeepSpeedZeroConfig(**z).stage) == z["stage"]
