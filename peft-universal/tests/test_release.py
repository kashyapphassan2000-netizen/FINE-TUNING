"""The post-training chain: serve -> evaluate the endpoint -> parity -> load test -> release gate.
Uses the REAL local server with the tiny model (not a mock)."""
import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
from peftkit import eval_endpoint, loadtest, release_gate, serve_local


@pytest.fixture(scope="module")
def server(tiny_model):
    import uvicorn

    serve_local.configure(str(tiny_model), name="tiny", api_key="k")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = uvicorn.Server(uvicorn.Config(serve_local.app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(timeout=5)


def test_serve_local_speaks_openai(server):
    assert httpx.get(f"{server}/health").json() == {"ok": True}
    assert httpx.post(f"{server}/v1/chat/completions", json={"model": "tiny", "messages": []}).status_code == 401
    r = httpx.post(f"{server}/v1/chat/completions", headers={"Authorization": "Bearer k"}, timeout=60,
                   json={"model": "tiny", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    body = r.json()
    assert r.status_code == 200 and body["object"] == "chat.completion"
    assert body["usage"]["completion_tokens"] <= 4 and isinstance(body["choices"][0]["message"]["content"], str)
    wrong = httpx.post(f"{server}/v1/chat/completions", headers={"Authorization": "Bearer k"},
                       json={"model": "other", "messages": [{"role": "user", "content": "hi"}]})
    assert wrong.status_code == 404
    m = httpx.get(f"{server}/metrics").text
    assert "vllm:num_requests_waiting" in m and "peftkit_requests_served_total" in m
    assert httpx.get(f"{server}/v1/models", headers={"Authorization": "Bearer k"}).json()["data"][0]["id"] == "tiny"


def test_eval_endpoint_and_parity(server, tmp_path):
    suite = tmp_path / "s.jsonl"
    suite.write_text("\n".join(json.dumps({"id": f"x{i}", "messages": [{"role": "user", "content": "hi"}],
                                           "expected": "6 bar"}) for i in range(3)))
    rep = asyncio.run(eval_endpoint.run_suites(f"{server}/v1", "tiny", "k", {"domain": str(suite)}, max_tokens=4))
    d = rep["suites"]["domain"]
    assert d["n"] == 3 and d["errors"] == 0 and d["mean_latency_s"] > 0
    same = eval_endpoint.parity(rep, rep)
    assert same["verdict"] == "PASS" and same["rows"][0]["gap"] == 0.0
    offline = json.loads(json.dumps(rep))
    offline["suites"]["domain"]["score"] = d["score"] + 0.5          # served much worse than offline
    assert eval_endpoint.parity(offline, rep)["verdict"] == "FAIL"
    broken = asyncio.run(eval_endpoint.run_suites(f"{server}/v1", "tiny", "WRONG-KEY", {"domain": str(suite)}))
    assert broken["suites"]["domain"]["errors"] == 3                  # auth failures are counted, not hidden
    assert eval_endpoint.parity(rep, broken)["verdict"] == "FAIL"


def test_loadtest_measures_and_recommends(server):
    prompts = [[{"role": "user", "content": "hi"}]]
    res = asyncio.run(loadtest.run(f"{server}/v1", "tiny", "k", prompts, [1, 2], 6, 4,
                                   metrics_url=f"{server}/metrics"))
    assert [r["concurrency"] for r in res] == [1, 2]
    assert all(r["errors"] == 0 and r["p95_s"] >= r["p50_s"] > 0 and r["req_per_s"] > 0 for r in res)
    rec = loadtest.recommend(res, slo_p95=60.0)
    assert rec["meets_slo"] and rec["max_concurrency_per_replica"] == 2
    assert loadtest.recommend(res, slo_p95=1e-6)["meets_slo"] is False
    assert loadtest.pct([1, 2, 3, 4], 0.5) == 2.5


def test_release_gate_needs_all_three():
    good = ({"verdict": "PASS: domain improved"}, {"verdict": "PASS"},
            {"recommendation": {"meets_slo": True, "max_concurrency_per_replica": 8}})
    assert release_gate.decide(*good)["decision"] == "RELEASE"
    for i, broken in enumerate([{"verdict": "WARNING: ... catastrophic forgetting"}, {"verdict": "FAIL"},
                                {"recommendation": {"meets_slo": False}}]):
        args = list(good)
        args[i] = broken
        r = release_gate.decide(*args)
        assert r["decision"] == "BLOCK" and r["failed"] == [["quality", "parity", "capacity"][i]]
