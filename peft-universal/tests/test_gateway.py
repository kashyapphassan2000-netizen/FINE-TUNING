import hashlib
import importlib
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serving"))

fake_vllm = FastAPI()
seen = {"auth": None, "hits": 0}


@fake_vllm.post("/v1/chat/completions")
async def fake_chat(req: Request):
    seen["auth"] = req.headers.get("authorization")
    seen["hits"] += 1
    body = await req.json()
    if body.get("stream"):
        async def gen():
            for tok in ("Hel", "lo"):
                yield f'data: {{"choices":[{{"delta":{{"content":"{tok}"}}}}]}}\n\n'
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "model": body["model"]}


@fake_vllm.get("/v1/models")
async def fake_models():
    return {"data": [{"id": "m"}]}


class Router(httpx.AsyncBaseTransport):
    def __init__(self):
        self.asgi = httpx.ASGITransport(app=fake_vllm)

    async def handle_async_request(self, request):
        if request.url.host == "dead":
            raise httpx.ConnectError("down", request=request)
        return await self.asgi.handle_async_request(request)


@pytest.fixture()
def gw(monkeypatch):
    monkeypatch.setenv("UPSTREAMS", "http://dead:8000,http://live:8000")
    monkeypatch.setenv("UPSTREAM_API_KEY", "vllm-secret")
    monkeypatch.setenv("GATEWAY_KEY_HASHES", hashlib.sha256(b"client-key").hexdigest())
    monkeypatch.setenv("GATEWAY_ADMIN_KEY_HASH", hashlib.sha256(b"admin").hexdigest())
    monkeypatch.setenv("RATE_LIMIT_RPM", "3")
    mod = importlib.import_module("gateway.app")
    mod.S.reload()
    mod.S.client_factory = lambda: httpx.AsyncClient(transport=Router())
    return TestClient(mod.app), mod


H = {"Authorization": "Bearer client-key"}
BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def test_auth_failover_and_rate_limit(gw):
    client, mod = gw
    assert client.post("/v1/chat/completions", json=BODY).status_code == 401
    r = client.post("/v1/chat/completions", json=BODY, headers=H)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "ok"
    assert seen["auth"] == "Bearer vllm-secret"          # client key is never forwarded
    assert mod.S.counters["upstream_failover"] >= 1       # dead node was skipped
    client.post("/v1/chat/completions", json=BODY, headers=H)
    client.post("/v1/chat/completions", json=BODY, headers=H)
    assert client.post("/v1/chat/completions", json=BODY, headers=H).status_code == 429
    m = client.get("/metrics").text
    assert 'event="rate_limited"' in m and "gateway_upstreams_healthy 1" in m


def test_streaming_models_admin(gw):
    client, mod = gw
    with client.stream("POST", "/v1/chat/completions", json={**BODY, "stream": True}, headers=H) as r:
        text = "".join(r.iter_text())
    assert '"Hel"' in text and "[DONE]" in text
    assert client.get("/v1/models", headers=H).json()["data"][0]["id"] == "m"
    assert client.post("/admin/upstreams", json={"upstreams": ["http://x"]}, headers=H).status_code == 401
    r = client.post("/admin/upstreams", json={"upstreams": ["http://live:8000/"]},
                    headers={"Authorization": "Bearer admin"})
    assert r.json() == {"upstreams": ["http://live:8000"]}
    assert client.get("/healthz").json()["healthy"] == 1


def test_adapter_acl(gw, monkeypatch):
    client, mod = gw
    import json as _json

    mod.S.acl = {hashlib.sha256(b"client-key").hexdigest(): ["finance"]}
    ok = client.post("/v1/chat/completions", json={**BODY, "model": "finance"}, headers=H)
    bad = client.post("/v1/chat/completions", json={**BODY, "model": "legal"}, headers=H)
    assert ok.status_code == 200 and bad.status_code == 403
    assert _json.loads(bad.text)["detail"].startswith("key not allowed")
