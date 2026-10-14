"""Shared async client for any OpenAI-compatible endpoint (Euron, vLLM, OpenAI, Together, ...).
Env: OPENAI_BASE_URL, OPENAI_API_KEY. Retries 429/5xx with exponential backoff, parses JSON replies."""
from __future__ import annotations

import asyncio
import json
import os
import random
import re

import httpx

RETRYABLE = {429, 500, 502, 503, 504}


def extract_json(text: str):
    """Models often wrap JSON in ``` fences or add prose; take the first JSON object/array."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        if not m:
            raise
        return json.loads(m.group(1))


class LLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 concurrency: int = 8, timeout: float = 120, retries: int = 4):
        self.model = model or os.environ.get("SYNTH_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ["OPENAI_BASE_URL"]).rstrip("/")
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.sem = asyncio.Semaphore(concurrency)
        self.timeout, self.retries = timeout, retries
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "failures": 0}

    async def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int | None = None,
                   client: httpx.AsyncClient | None = None) -> str | None:
        body = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens:
            body["max_tokens"] = max_tokens
        own = client is None
        client = client or httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout,
                                             headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            for attempt in range(self.retries):
                try:
                    async with self.sem:
                        r = await client.post("/chat/completions", json=body)
                    if r.status_code in RETRYABLE:
                        raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                    r.raise_for_status()
                    data = r.json()
                    self.usage["calls"] += 1
                    for k in ("prompt_tokens", "completion_tokens"):
                        self.usage[k] += (data.get("usage") or {}).get(k, 0)
                    return data["choices"][0]["message"]["content"]
                except (httpx.HTTPError, KeyError, ValueError):
                    if attempt == self.retries - 1:
                        self.usage["failures"] += 1
                        return None
                    await asyncio.sleep(2**attempt + random.random())
        finally:
            if own:
                await client.aclose()
        return None

    async def chat_json(self, messages: list[dict], temperature: float = 0.7, **kw):
        for _ in range(2):  # one re-ask on malformed JSON
            text = await self.chat(messages, temperature, **kw)
            if text is None:
                return None
            try:
                return extract_json(text)
            except (json.JSONDecodeError, ValueError):
                continue
        return None


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
