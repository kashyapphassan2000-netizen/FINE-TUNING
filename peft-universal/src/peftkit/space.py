"""Put your fine-tuned model on a Hugging Face SPACE: a public (or private) web page anyone can chat with.

Two backends, because a free Space is a small CPU box (2 vCPU, 16 GB RAM at the time of writing):
  local    : the Space loads the model itself with transformers. Realistic only for small models (<= ~1-2B on
             free CPU, slow), or on paid GPU hardware / ZeroGPU (PRO or org plans).
  endpoint : the Space is only the UI; it calls YOUR OpenAI-compatible server (vLLM on the rented GPU, behind
             the gateway). The API key is stored as a Space SECRET, never in the code. Any model size.

  python -m peftkit.space --space you/pump-assistant-demo --model you/pump-assistant --backend local
  python -m peftkit.space --space you/pump-assistant-demo --model mymodel --backend endpoint \
      --endpoint-url https://api.example.com/v1 --endpoint-key-env DEMO_API_KEY
  python -m peftkit.space --space you/pump-assistant-demo ... --dry-run     # writes the folder, uploads nothing
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[2] / "serving" / "space"
GRADIO_VERSION = "6.30.0"   # latest on PyPI when this was written (October 2026); keep app.py tested against it

README = """---
title: {title}
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: {gradio}
app_file: app.py
pinned: false
license: {license}
short_description: {short}
---

# {title}

Chat demo for **{model}**, fine-tuned with peft-universal. Backend: `{backend}`.

{limits}
"""


def requirements(backend: str, gpu: bool) -> str:
    """Only what the chosen backend imports. endpoint: no torch at all. local on CPU: the CPU torch wheel
    (PyPI's Linux torch is a ~2.5 GB CUDA build that a CPU Space does not need)."""
    if backend == "endpoint":
        return "openai>=1.0\n"
    lines = ([] if gpu else ["--extra-index-url https://download.pytorch.org/whl/cpu"]) + \
            ["torch", "transformers==5.17.0", "accelerate==1.15.0"]
    return "\n".join(lines) + "\n"


def prepare(out: Path, model: str, backend: str, title: str, license: str, system: str | None,
            endpoint_url: str | None, gpu: bool = False) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(APP_DIR / "app.py", out / "app.py")
    (out / "requirements.txt").write_text(requirements(backend, gpu), encoding="utf-8")
    limits = ("Answers come from a fine-tuned model and can be wrong. Do not enter personal or confidential data."
              if backend == "endpoint" else
              "Runs on the Space's own hardware: on the free CPU tier, expect several seconds per answer for small "
              "models and out-of-memory for large ones. Answers can be wrong.")
    (out / "README.md").write_text(README.format(title=title, gradio=GRADIO_VERSION, license=license,
                                                 short=f"Chat with {model}"[:60], model=model, backend=backend,
                                                 limits=limits), encoding="utf-8")
    return out


def deploy(space: str, model: str, backend: str, title: str, private: bool, license: str = "other",
           system: str | None = None, endpoint_url: str | None = None, endpoint_key: str | None = None,
           hardware: str | None = None, hf_token_for_private_model: bool = False, dry_run: bool = False) -> dict:
    folder = prepare(Path(tempfile.mkdtemp()) / "space", model, backend, title, license, system, endpoint_url,
                     gpu=bool(hardware and not hardware.startswith("cpu")))
    variables = {"MODEL_ID": model, "BACKEND": backend}
    if system:
        variables["SYSTEM_PROMPT"] = system
    if endpoint_url:
        variables["ENDPOINT_URL"] = endpoint_url
    secrets = {}
    if endpoint_key:
        secrets["ENDPOINT_API_KEY"] = endpoint_key
    if hf_token_for_private_model and os.environ.get("HF_TOKEN"):
        secrets["HF_TOKEN"] = os.environ["HF_TOKEN"]          # lets the Space download YOUR private model
    result = {"space": space, "folder": str(folder), "variables": variables, "secrets": sorted(secrets),
              "url": f"https://huggingface.co/spaces/{space}"}
    if dry_run:
        return result
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="gradio", private=private, exist_ok=True)
    for k, v in variables.items():
        api.add_space_variable(space, k, v)
    for k, v in secrets.items():
        api.add_space_secret(space, k, v)
    api.upload_folder(repo_id=space, repo_type="space", folder_path=str(folder), commit_message="deploy demo")
    if hardware:
        api.request_space_hardware(space, hardware)          # paid tiers: billed to your account
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Deploy a chat demo of your model to a Hugging Face Space")
    ap.add_argument("--space", required=True, help="user/space-name")
    ap.add_argument("--model", required=True, help="Hub model id (local backend) or served model name (endpoint)")
    ap.add_argument("--backend", default="local", choices=["local", "endpoint"])
    ap.add_argument("--title", default=None)
    ap.add_argument("--system", default=None, help="system prompt the demo sends (use the one you trained with)")
    ap.add_argument("--endpoint-url", default=None, help="https://.../v1 of your vLLM/gateway (endpoint backend)")
    ap.add_argument("--endpoint-key-env", default=None, help="NAME of the env var holding the endpoint API key")
    ap.add_argument("--private", action="store_true", help="only you (and your org) can open the Space")
    ap.add_argument("--private-model", action="store_true", help="copy HF_TOKEN into a Space secret to load it")
    ap.add_argument("--hardware", default=None, help="e.g. t4-small, a10g-small (PAID; default free cpu-basic)")
    ap.add_argument("--license", default="other")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.backend == "endpoint" and not a.endpoint_url:
        ap.error("--backend endpoint needs --endpoint-url")
    key = os.environ.get(a.endpoint_key_env) if a.endpoint_key_env else None
    if a.endpoint_key_env and not key:
        ap.error(f"environment variable {a.endpoint_key_env} is empty")
    r = deploy(a.space, a.model, a.backend, a.title or a.space.split("/")[-1].replace("-", " ").title(),
               a.private, a.license, a.system, a.endpoint_url, key, a.hardware, a.private_model, a.dry_run)
    import json

    print(json.dumps(r, indent=2))
    if not a.dry_run:
        print(f"open {r['url']} - the first build takes a few minutes (Logs tab shows progress)")


if __name__ == "__main__":
    main()
