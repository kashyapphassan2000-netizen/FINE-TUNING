"""A small OpenAI-compatible server backed by transformers, for LAPTOP rehearsal (CPU or one GPU).

It is NOT a vLLM replacement: one request is generated at a time, there is no paged KV cache and no
continuous batching. It exists so that the complete chain - train -> merge -> serve -> evaluate the
ENDPOINT -> load test -> gateway -> autoscaling decision - can be executed end to end before you pay
for a GPU. It exposes the same routes the rest of the toolkit uses:

  POST /v1/chat/completions   GET /v1/models   GET /health   GET /metrics (vLLM-style queue gauges)

python -m peftkit.serve_local --model outputs/run/merged --port 8000 --api-key local-secret
"""
from __future__ import annotations

import argparse
import asyncio
import time
import uuid

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from .generate import load


class State:
    model = None
    tok = None
    name = "model"
    api_key = ""
    lock: asyncio.Lock | None = None
    running = 0
    waiting = 0
    served = 0
    tokens_out = 0


S = State()
app = FastAPI(title="peftkit local OpenAI-compatible server")


def configure(model_path: str, name: str = "model", api_key: str = "") -> None:
    S.model, S.tok = load(model_path)
    S.name, S.api_key = name, api_key
    S.lock = asyncio.Lock()


def _auth(request: Request) -> None:
    if not S.api_key:
        return
    if request.headers.get("authorization", "") != f"Bearer {S.api_key}":
        raise HTTPException(401, "invalid api key")


def _generate(messages: list[dict], max_tokens: int, temperature: float) -> tuple[str, int, int]:
    tok, model = S.tok, S.model
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    kw = dict(max_new_tokens=max_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=float(temperature))
    else:
        kw.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**enc, **kw)
    new = out[0, enc["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True), int(enc["input_ids"].shape[1]), int(new.shape[0])


@app.post("/v1/chat/completions")
async def chat(request: Request):
    _auth(request)
    body = await request.json()
    if body.get("model") not in (None, S.name):
        raise HTTPException(404, f"model '{body.get('model')}' not served here (serving '{S.name}')")
    S.waiting += 1
    t0 = time.perf_counter()
    async with S.lock:                       # one generation at a time: honest about what this server is
        S.waiting -= 1
        S.running += 1
        try:
            text, p_tok, c_tok = await asyncio.to_thread(
                _generate, body["messages"], int(body.get("max_tokens", 64)), float(body.get("temperature", 0.0)))
        finally:
            S.running -= 1
    S.served += 1
    S.tokens_out += c_tok
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "model": S.name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok},
            "latency_s": round(time.perf_counter() - t0, 4)}


@app.get("/v1/models")
async def models(request: Request):
    _auth(request)
    return {"object": "list", "data": [{"id": S.name, "object": "model"}]}


@app.get("/health")
async def health():
    return {"ok": S.model is not None}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    # same metric names vLLM exposes, so the autoscaler and dashboards work unchanged
    return (f'vllm:num_requests_waiting{{model_name="{S.name}"}} {S.waiting}\n'
            f'vllm:num_requests_running{{model_name="{S.name}"}} {S.running}\n'
            f'peftkit_requests_served_total {S.served}\n'
            f'peftkit_generation_tokens_total {S.tokens_out}\n')


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="Local OpenAI-compatible server (rehearsal only)")
    ap.add_argument("--model", required=True, help="merged model dir, adapter dir, or Hub id")
    ap.add_argument("--name", default="model")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key", default="")
    a = ap.parse_args(argv)
    configure(a.model, a.name, a.api_key)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
