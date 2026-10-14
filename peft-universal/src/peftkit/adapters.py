"""Multi-adapter (plug-and-play) management: one base model, many domain adapters.

registry YAML -> validate (same base + revision, servable format, unique names, rank fits vLLM)
             -> vLLM flags for static loading
             -> runtime load/unload on a running vLLM (VLLM_ALLOW_RUNTIME_LORA_UPDATING=True)
             -> request routing (explicit name, or keyword router) -> local PEFT multi-adapter use

vLLM facts used (v0.29.0 source): POST /v1/load_lora_adapter {lora_name, lora_path, load_inplace},
POST /v1/unload_lora_adapter {lora_name}; --max-loras = adapters active in one batch; --max-cpu-loras = cache.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .serve_check import VLLM_MAX_LORA_RANKS

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")


@dataclass
class Adapter:
    name: str
    source: str                      # hub id or local path
    domain: str = ""
    method: str = "lora"
    rank: int = 16
    base_revision: str | None = None
    keywords: list = field(default_factory=list)
    status: str = "active"           # active | shadow | retired
    eval: dict = field(default_factory=dict)


@dataclass
class Registry:
    base: str
    base_revision: str | None
    adapters: list

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        d = yaml.safe_load(Path(path).read_text())
        return cls(base=d["base"]["name"], base_revision=d["base"].get("revision"),
                   adapters=[Adapter(**a) for a in d.get("adapters", [])])

    def active(self) -> list:
        """Routable adapters."""
        return [a for a in self.adapters if a.status == "active"]

    def loaded(self) -> list:
        """Adapters the server should hold: active + shadow (shadow = evaluated, never routed)."""
        return [a for a in self.adapters if a.status in ("active", "shadow")]


def validate(reg: Registry) -> list[str]:
    errors, seen = [], set()
    for a in reg.adapters:
        if not NAME_RE.match(a.name):
            errors.append(f"{a.name}: name must match {NAME_RE.pattern}")
        if a.name in seen:
            errors.append(f"{a.name}: duplicate name")
        seen.add(a.name)
        if a.method in ("dora", "adalora"):
            errors.append(f"{a.name}: {a.method} adapters cannot be hot-swapped in vLLM 0.29 (merge and deploy separately)")
        if a.base_revision and reg.base_revision and a.base_revision != reg.base_revision:
            errors.append(f"{a.name}: trained on base revision {a.base_revision}, registry serves {reg.base_revision}")
        if a.rank > VLLM_MAX_LORA_RANKS[-1]:
            errors.append(f"{a.name}: rank {a.rank} > 512")
        local = Path(a.source) / "adapter_config.json"
        if local.exists():
            cfg = json.loads(local.read_text())
            base = cfg.get("base_model_name_or_path")
            if base and base != reg.base:
                errors.append(f"{a.name}: adapter_config base '{base}' != registry base '{reg.base}'")
            if cfg.get("r") and cfg["r"] != a.rank:
                errors.append(f"{a.name}: registry rank {a.rank} != adapter r {cfg['r']}")
            if cfg.get("use_dora") or cfg.get("peft_type") != "LORA" or cfg.get("modules_to_save"):
                errors.append(f"{a.name}: adapter format not servable as a vLLM LoRA (run peftkit.serve_check)")
    return errors


def vllm_args(reg: Registry, max_loras: int | None = None) -> list[str]:
    act = reg.loaded()
    need = max((a.rank for a in act), default=16)
    max_rank = next(x for x in VLLM_MAX_LORA_RANKS if x >= need)
    args = ["serve", reg.base]
    if reg.base_revision:
        args += ["--revision", reg.base_revision]
    args += ["--enable-lora", "--max-lora-rank", str(max_rank), "--max-loras", str(max_loras or max(1, len(act))),
             "--max-cpu-loras", str(max(len(act), max_loras or 1)), "--lora-modules"]
    args += [f"{a.name}={a.source}" for a in act]
    return args


async def runtime_load(url: str, api_key: str, name: str, path: str, client=None) -> dict:
    import httpx

    own = client is None
    client = client or httpx.AsyncClient(timeout=120)
    try:
        r = await client.post(f"{url.rstrip('/')}/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path},
                              headers={"Authorization": f"Bearer {api_key}"})
        return {"status": r.status_code, "body": r.text[:300]}
    finally:
        if own:
            await client.aclose()


async def runtime_unload(url: str, api_key: str, name: str, client=None) -> dict:
    import httpx

    own = client is None
    client = client or httpx.AsyncClient(timeout=60)
    try:
        r = await client.post(f"{url.rstrip('/')}/v1/unload_lora_adapter", json={"lora_name": name},
                              headers={"Authorization": f"Bearer {api_key}"})
        return {"status": r.status_code, "body": r.text[:300]}
    finally:
        if own:
            await client.aclose()


def route(reg: Registry, text: str, requested: str | None = None, default: str | None = None) -> str:
    """Explicit adapter name wins. Otherwise score keywords; ties/no hits go to `default` (or the base).
    Keyword routing is a baseline. For production, train a small classifier or ask a cheap LLM."""
    names = {a.name for a in reg.active()}
    if requested:
        if requested in names or requested == "base":
            return requested
        raise KeyError(f"unknown or inactive adapter '{requested}'")
    low = text.lower()
    scores = {a.name: sum(low.count(k.lower()) for k in a.keywords) for a in reg.active()}
    best = max(scores.values(), default=0)
    winners = [n for n, s in scores.items() if s == best and s > 0]
    return winners[0] if len(winners) == 1 else (default or "base")


def load_many(base_model, entries: dict):
    """Local PEFT: attach several named adapters to ONE base model instance.
    entries = {"finance": "path_or_hub", "legal": "..."}; returns a PeftModel. Switch with set_adapter(name),
    run the base with `with model.disable_adapter():`, or pass adapter_names=[...] per row in generate()."""
    from peft import PeftModel

    items = list(entries.items())
    model = PeftModel.from_pretrained(base_model, items[0][1], adapter_name=items[0][0])
    for name, src in items[1:]:
        model.load_adapter(src, adapter_name=name)
    return model


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Adapter registry tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("registry")
    g = sub.add_parser("vllm-args")
    g.add_argument("registry")
    g.add_argument("--max-loras", type=int, default=None)
    r = sub.add_parser("route")
    r.add_argument("registry")
    r.add_argument("text")
    for name in ("load", "unload"):
        s = sub.add_parser(name)
        s.add_argument("--url", required=True)
        s.add_argument("--name", required=True)
        if name == "load":
            s.add_argument("--path", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        errs = validate(Registry.load(a.registry))
        print("OK" if not errs else "\n".join(errs))
        raise SystemExit(1 if errs else 0)
    if a.cmd == "vllm-args":
        print("vllm " + " ".join(vllm_args(Registry.load(a.registry), a.max_loras)))
    elif a.cmd == "route":
        print(route(Registry.load(a.registry), a.text))
    else:
        import asyncio
        import os

        key = os.environ["VLLM_API_KEY"]
        coro = runtime_load(a.url, key, a.name, a.path) if a.cmd == "load" else runtime_unload(a.url, key, a.name)
        print(json.dumps(asyncio.run(coro)))


if __name__ == "__main__":
    main()
