"""Public-API layer: self-serve keys, per-key limits and model access, metering, capacity maths, showcase."""
import hashlib
import importlib
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving"))
from peftkit import capacity, keys, showcase  # noqa: E402

up = FastAPI()


@up.post("/v1/chat/completions")
async def fake(body: dict):
    return {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}}


@pytest.fixture()
def gw(monkeypatch, tmp_path):
    kf = tmp_path / "keys.json"
    monkeypatch.setenv("UPSTREAMS", "http://live:8000")
    monkeypatch.setenv("GATEWAY_KEY_HASHES", "")
    monkeypatch.setenv("GATEWAY_ADMIN_KEY_HASH", hashlib.sha256(b"admin").hexdigest())
    monkeypatch.setenv("GATEWAY_KEYS_FILE", str(kf))
    monkeypatch.setenv("RATE_LIMIT_RPM", "1000")
    mod = importlib.import_module("gateway.app")
    mod.S.reload()
    mod.S.client_factory = lambda: httpx.AsyncClient(transport=httpx.ASGITransport(app=up))
    return TestClient(mod.app), mod, kf


def test_self_serve_keys_limits_acl_and_metering(gw):
    client, mod, kf = gw
    admin = {"Authorization": "Bearer admin"}
    assert client.post("/admin/keys", json={"name": "x"}).status_code == 401          # admin only
    alice = client.post("/admin/keys", headers=admin, json={"name": "alice", "rpm": 2, "models": ["mymodel"]}).json()
    bob = client.post("/admin/keys", headers=admin, json={"name": "bob"}).json()
    assert alice["api_key"].startswith("sk-") and alice["key_id"] in json.dumps(json.loads(kf.read_text()))
    assert alice["api_key"] not in kf.read_text()                                       # only hashes on disk
    ha = {"Authorization": f"Bearer {alice['api_key']}"}
    body = {"model": "mymodel", "messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/v1/chat/completions", json=body, headers=ha).status_code == 200
    # forbidden model: 403, and it still spends rate-limit budget (abuse attempts are not free)
    assert client.post("/v1/chat/completions", json={**body, "model": "other"}, headers=ha).status_code == 403
    assert client.post("/v1/chat/completions", json=body, headers=ha).status_code == 429  # per-key rpm = 2 used up
    hb = {"Authorization": f"Bearer {bob['api_key']}"}
    assert client.post("/v1/chat/completions", json={**body, "model": "other"}, headers=hb).status_code == 200
    use = client.get("/admin/usage", headers=admin).json()
    assert use[alice["key_id"]]["requests"] == 1 and use[alice["key_id"]]["completion_tokens"] == 3
    assert use[alice["key_id"]]["name"] == "alice"
    assert client.delete(f"/admin/keys/{bob['key_id']}", headers=admin).json() == {"revoked": bob["key_id"]}
    assert client.post("/v1/chat/completions", json=body, headers=hb).status_code == 401
    mod.S.reload()                                                                      # keys survive a restart
    mod.S.client_factory = lambda: httpx.AsyncClient(transport=httpx.ASGITransport(app=up))
    assert client.post("/v1/chat/completions", json=body, headers=ha).status_code == 200  # fresh bucket after restart
    assert client.post("/v1/chat/completions", json=body, headers=hb).status_code == 401  # revocation persisted


def test_keys_cli_offline(tmp_path):
    f = tmp_path / "k.json"
    k = keys.create(str(f), "carol", rpm=10, models=["m"])
    stored = json.loads(f.read_text())
    assert hashlib.sha256(k["api_key"].encode()).hexdigest() in stored
    assert keys.revoke(str(f), k["key_id"]) and json.loads(f.read_text()) == {}


def test_capacity_math():
    t = capacity.little_law(users=2000, active_share=0.1, req_per_active_min=2, latency_s=4)
    assert t["arrival_rate_rps"] == pytest.approx(6.667, abs=1e-3) and t["avg_concurrency"] == pytest.approx(26.67, abs=0.01)
    r = capacity.replicas(26.67, peak_factor=3, cap_per_replica=16, headroom=0.3)
    assert r["replicas"] == 7                                   # ceil(80.01 * 1.3 / 16) = ceil(6.5) = 7
    ld = capacity.load_time(15.0, net_gbps=1.0, disk_gb_s=2.0, init_s=60)
    assert ld["download_s"] == 120 and ld["cold_start_s"] == 120 + 8 + 60
    prices = [{"provider": "P", "gpu": "H100", "vram_gb": 80, "gpus_per_node": 8, "price_per_gpu_hour": 2.0},
              {"provider": "P", "gpu": "small", "vram_gb": 8, "gpus_per_node": 8, "price_per_gpu_hour": 0.5}]
    lt = {"recommendation": {"max_concurrency_per_replica": 16}}
    res = capacity.compute("configs/models/llama-3.1-8b.config.json", 2000, 0.1, 2, 600, 250, 4.0,
                           loadtest=lt, prices=prices)
    h100 = res["providers"][0]
    assert h100["capacity_source"] == "load test (SLO)" and h100["replicas"] == 7 and h100["fits_on_tp"] == 1
    assert h100["serve_cost_month"] == pytest.approx(7 * 1 * 2.0 * 730)
    assert res["providers"][1]["fits_on_tp"] in (2, 4)          # 15 GB of weights needs tensor parallel on 8 GB cards
    assert res["warning"] is None
    no_lt = capacity.compute("configs/models/llama-3.1-8b.config.json", 2000, 0.1, 2, 600, 250, 4.0, prices=prices[:1])
    assert "under-provision" in no_lt["warning"]


def test_showcase_is_built_from_reports(tmp_path):
    cmp_ = {"rows": [{"suite": "domain", "before": 0.2, "after": 0.8, "delta": 0.6}], "verdict": "PASS: ok",
            "n_regressions": 1, "regressions": [{"suite": "domain", "id": "d3", "before": 1.0, "after": 0.0, "prediction": "wrong"}]}
    lt = {"results": [{"concurrency": 4, "p50_s": 0.2, "p95_s": 0.3, "req_per_s": 20, "out_tok_per_s": 180, "errors": 0}],
          "recommendation": {"meets_slo": True, "slo_p95_s": 2, "max_concurrency_per_replica": 4, "suggested_scale_up_threshold": 1}}
    doc = showcase.build("Demo <b>", compare=cmp_, loadtest=lt, run={"base_model": "b", "method": "lora",
                          "trainable_params": 10, "total_params": 100, "versions": {}})
    assert "Demo &lt;b&gt;" in doc and "PASS: ok" in doc and "wrong" in doc and "Honest limits" in doc
    assert "<script" not in doc
