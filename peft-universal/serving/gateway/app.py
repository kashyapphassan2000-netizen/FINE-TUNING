"""API gateway in front of one or more vLLM servers.

- API keys: clients send `Authorization: Bearer <key>`; only SHA-256 hashes are stored (GATEWAY_KEY_HASHES)
- per-key rate limit (token bucket, in-memory -> use Redis when running >1 gateway replica)
- round-robin over healthy upstreams, failover on connection errors, streaming passthrough
- /healthz and Prometheus /metrics
Env: UPSTREAMS="http://10.0.0.5:8000,http://10.0.0.6:8000"  UPSTREAM_API_KEY  GATEWAY_KEY_HASHES  RATE_LIMIT_RPM
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import secrets
import time
from collections import defaultdict
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

UNHEALTHY_SECONDS = 30


class State:
    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        self.upstreams = [u.strip().rstrip("/") for u in os.environ.get("UPSTREAMS", "").split(",") if u.strip()]
        self.key_hashes = {h.strip() for h in os.environ.get("GATEWAY_KEY_HASHES", "").split(",") if h.strip()}
        self.admin_hash = os.environ.get("GATEWAY_ADMIN_KEY_HASH", "").strip()
        self.upstream_key = os.environ.get("UPSTREAM_API_KEY", "")
        # optional per-key adapter permissions: {"<key sha256>": ["finance", "base"], ...}
        self.acl = json.loads(os.environ.get("ADAPTER_ACL", "") or "{}")
        # file-backed key store for self-serve keys: {sha256: {"name", "rpm", "models", "created"}}
        self.keys_file = os.environ.get("GATEWAY_KEYS_FILE", "")
        self.key_meta: dict = {}
        if self.keys_file and Path(self.keys_file).exists():
            self.key_meta = json.loads(Path(self.keys_file).read_text() or "{}")
            self.key_hashes |= set(self.key_meta)
            for h, meta in self.key_meta.items():
                if meta.get("models"):
                    self.acl[h] = meta["models"]
        self.usage: dict = defaultdict(lambda: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0})
        self.rpm = float(os.environ.get("RATE_LIMIT_RPM", "60"))
        self.buckets: dict[str, tuple[float, float]] = {}
        self.down_until: dict[str, float] = {}
        self.rr = itertools.count()
        self.counters: dict[str, int] = defaultdict(int)
        self.client_factory = lambda: httpx.AsyncClient(timeout=httpx.Timeout(600, connect=5))


S = State()
app = FastAPI(title="LLM gateway")
if os.environ.get("ALLOWED_ORIGINS"):
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(CORSMiddleware, allow_origins=os.environ["ALLOWED_ORIGINS"].split(","),
                       allow_methods=["GET", "POST"], allow_headers=["authorization", "content-type"])


def sha(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def auth(request: Request) -> str:
    header = request.headers.get("authorization", "")
    key = header[7:] if header.lower().startswith("bearer ") else ""
    h = sha(key) if key else ""
    if not key or h not in S.key_hashes:
        S.counters["auth_fail"] += 1
        raise HTTPException(401, "invalid api key")
    return h


def save_keys() -> None:
    if S.keys_file:
        Path(S.keys_file).write_text(json.dumps(S.key_meta, indent=2))


def allow(key_hash: str) -> bool:
    now = time.monotonic()
    rpm = float(S.key_meta.get(key_hash, {}).get("rpm") or S.rpm)       # per-key limit if the key has one
    tokens, last = S.buckets.get(key_hash, (rpm, now))
    tokens = min(rpm, tokens + (now - last) * rpm / 60.0)
    if tokens < 1:
        S.buckets[key_hash] = (tokens, now)
        return False
    S.buckets[key_hash] = (tokens - 1, now)
    return True


def healthy_order() -> list[str]:
    now = time.monotonic()
    alive = [u for u in S.upstreams if S.down_until.get(u, 0) <= now]
    if not alive:
        return []
    start = next(S.rr) % len(alive)
    return alive[start:] + alive[:start]


async def forward(request: Request, path: str):
    key_hash = auth(request)
    if not allow(key_hash):
        S.counters["rate_limited"] += 1
        raise HTTPException(429, "rate limit exceeded")
    body = await request.json()
    allowed = S.acl.get(key_hash)
    if allowed is not None and body.get("model") not in allowed:
        S.counters["adapter_forbidden"] += 1
        raise HTTPException(403, f"key not allowed to use model '{body.get('model')}'")
    headers = {"Authorization": f"Bearer {S.upstream_key}"} if S.upstream_key else {}
    order = healthy_order()
    if not order:
        raise HTTPException(503, "no healthy upstream")
    for upstream in order:
        client = S.client_factory()
        try:
            if body.get("stream"):
                req = client.build_request("POST", f"{upstream}{path}", json=body, headers=headers)
                resp = await client.send(req, stream=True)

                async def gen(resp=resp, client=client):
                    try:
                        async for chunk in resp.aiter_raw():
                            yield chunk
                    finally:
                        await resp.aclose()
                        await client.aclose()

                S.counters["requests_ok"] += 1
                return StreamingResponse(gen(), status_code=resp.status_code,
                                         media_type=resp.headers.get("content-type", "text/event-stream"))
            resp = await client.post(f"{upstream}{path}", json=body, headers=headers)
            await client.aclose()
            S.counters["requests_ok" if resp.status_code < 500 else "requests_upstream_error"] += 1
            data = resp.json()
            u = S.usage[key_hash]                                   # metering: the basis of any billing
            u["requests"] += 1
            u["prompt_tokens"] += int((data.get("usage") or {}).get("prompt_tokens", 0))
            u["completion_tokens"] += int((data.get("usage") or {}).get("completion_tokens", 0))
            return JSONResponse(data, status_code=resp.status_code)
        except httpx.TransportError:
            await client.aclose()
            S.down_until[upstream] = time.monotonic() + UNHEALTHY_SECONDS
            S.counters["upstream_failover"] += 1
    raise HTTPException(502, "all upstreams failed")


@app.post("/v1/chat/completions")
async def chat(request: Request):
    return await forward(request, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request):
    return await forward(request, "/v1/completions")


@app.get("/v1/models")
async def models(request: Request):
    auth(request)
    for upstream in healthy_order():
        async with S.client_factory() as client:
            try:
                headers = {"Authorization": f"Bearer {S.upstream_key}"} if S.upstream_key else {}
                r = await client.get(f"{upstream}/v1/models", headers=headers)
                return JSONResponse(r.json(), status_code=r.status_code)
            except httpx.TransportError:
                S.down_until[upstream] = time.monotonic() + UNHEALTHY_SECONDS
    raise HTTPException(503, "no healthy upstream")


@app.post("/admin/upstreams")
async def set_upstreams(request: Request):
    """Used by the autoscaler to add/remove vLLM nodes. Protected by a separate admin key."""
    header = request.headers.get("authorization", "")
    if not S.admin_hash or sha(header[7:]) != S.admin_hash:
        raise HTTPException(401, "admin key required")
    S.upstreams = [u.rstrip("/") for u in (await request.json())["upstreams"]]
    return {"upstreams": S.upstreams}


def _admin(request: Request) -> None:
    header = request.headers.get("authorization", "")
    if not S.admin_hash or sha(header[7:]) != S.admin_hash:
        raise HTTPException(401, "admin key required")


@app.post("/admin/keys")
async def create_key(request: Request):
    """Issue an API key. The plaintext is returned ONCE; only its hash is stored."""
    _admin(request)
    body = await request.json()
    key = "sk-" + secrets.token_urlsafe(32)
    h = sha(key)
    S.key_meta[h] = {"name": body.get("name", "unnamed"), "rpm": body.get("rpm"), "models": body.get("models"),
                     "created": int(time.time())}
    S.key_hashes.add(h)
    if body.get("models"):
        S.acl[h] = body["models"]
    save_keys()
    return {"api_key": key, "key_id": h[:12], "note": "store it now: it cannot be shown again"}


@app.delete("/admin/keys/{key_id}")
async def revoke_key(key_id: str, request: Request):
    _admin(request)
    hits = [h for h in S.key_hashes if h.startswith(key_id)]
    if len(hits) != 1:
        raise HTTPException(404, "unknown or ambiguous key_id")
    h = hits[0]
    S.key_hashes.discard(h)
    S.key_meta.pop(h, None)
    S.acl.pop(h, None)
    save_keys()
    return {"revoked": key_id}


@app.get("/admin/usage")
async def usage(request: Request):
    _admin(request)
    return {h[:12]: {**S.usage[h], "name": S.key_meta.get(h, {}).get("name")} for h in S.key_hashes}


@app.get("/healthz")
async def healthz():
    return {"ok": True, "upstreams": len(S.upstreams), "healthy": len(healthy_order())}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    lines = [f'gateway_events_total{{event="{k}"}} {v}' for k, v in sorted(S.counters.items())]
    lines.append(f"gateway_upstreams_healthy {len(healthy_order())}")
    return "\n".join(lines) + "\n"
