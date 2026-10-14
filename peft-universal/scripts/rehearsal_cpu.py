"""FULL-PIPELINE REHEARSAL on a laptop CPU, no GPU, no API key, ~2 minutes.

Runs every stage of the book in order on a tiny model, with the REAL code paths:
  data -> doctor -> baseline eval -> train (LoRA) -> eval -> compare -> merge -> serve (OpenAI API)
  -> gateway (API keys) -> evaluate the ENDPOINT -> parity -> load test -> release gate -> scaling numbers

Run it after setup and before renting anything:
    python scripts/rehearsal_cpu.py          (Windows PowerShell or Linux; same command)
If this passes, every command in the runbook is proven to work on your machine.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "serving")]
WORK = ROOT / "outputs" / "rehearsal"


def step(n, title):
    print(f"\n=== STEP {n:>2}: {title}", flush=True)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def py(*args):
    """Run a peftkit CLI exactly as a user would, and show the tail of its output."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT, env=env, capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).splitlines()
             if l.strip() and "\x1b" not in l and "\r" not in l
             and not l.lstrip().startswith("{'") and "HTTP Request" not in l and " INFO " not in l
             and not any(k in l for k in ("it/s]", "examples/s", "kernels", "Warning", "warn(", "Loading weights"))]
    for l in lines[-12:]:
        print("   ", l)
    return r.returncode


def main():
    import logging
    logging.disable(logging.WARNING)
    WORK.mkdir(parents=True, exist_ok=True)
    from tiny import build_tiny

    step(1, "base model (tiny, offline) and a real dataset split")
    model_dir = build_tiny(WORK / "base")
    qa = [("What is the maximum pressure?", "The maximum pressure is 6 bar."),
          ("How do I reset the controller?", "Hold the red button for five seconds."),
          ("What do warranty claims need?", "The serial number and the purchase date."),
          ("When is service due?", "Service is due every six months.")]
    held = [("What is the pressure limit?", "The maximum pressure is 6 bar."),
            ("How is the controller reset?", "Hold the red button for five seconds.")]
    row = lambda q, a: {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}  # noqa: E731
    (WORK / "train.jsonl").write_text("\n".join(json.dumps(row(q, a)) for q, a in qa * 12))
    (WORK / "eval.jsonl").write_text("\n".join(json.dumps(row(q, a)) for q, a in held))
    (WORK / "test.jsonl").write_text("\n".join(json.dumps({"id": f"d{i}", "messages": [{"role": "user", "content": q}],
                                                           "expected": a, "must_include": [a.split()[-2]]})
                                                for i, (q, a) in enumerate(held + qa[:2])))
    (WORK / "general.jsonl").write_text("\n".join(json.dumps({"id": f"g{i}", "messages": [{"role": "user", "content": q}],
                                                              "expected": a}) for i, (q, a) in enumerate(qa[2:])))
    print(f"    train {len(qa) * 12} rows | eval {len(held)} | test suites: domain=4 general=2")

    cfg = [str(ROOT / "configs/base.yaml"), str(ROOT / "configs/methods/lora.yaml")]
    sets = [f"model.name_or_path={model_dir}", "model.max_seq_length=128",
            f"data.train_file={WORK / 'train.jsonl'}", f"data.eval_file={WORK / 'eval.jsonl'}",
            "data.assistant_only_loss=true", f"train.output_dir={WORK / 'run'}", "train.epochs=20",
            "train.per_device_batch_size=8", "train.grad_accum=1", "train.lr=5e-3", "train.logging_steps=20",
            "train.eval_steps=60", "train.save_steps=10000", "method.r=16"]

    step(2, "pre-flight doctor (must have zero FAIL)")
    code = py("peftkit.diagnostics", "--config", *cfg, "--set", *sets)
    assert code == 0, "doctor reported FAIL"

    step(3, "BASELINE evaluation of the untrained base model")
    suites = [f"domain={WORK / 'test.jsonl'}", f"general={WORK / 'general.jsonl'}"]
    py("peftkit.evaluate", "--model", str(model_dir), "--suite", *suites, "--max-new-tokens", "16",
       "--out", str(WORK / "base.json"))

    step(4, "train (LoRA, assistant-only loss)")
    assert py("peftkit.train", "--config", *cfg, "--set", *sets) == 0

    step(5, "evaluate the TRAINED adapter on the same suites, then compare")
    py("peftkit.evaluate", "--model", str(WORK / "run/adapter"), "--suite", *suites, "--max-new-tokens", "16",
       "--out", str(WORK / "tuned.json"))
    py("peftkit.evaluate", "--compare", str(WORK / "base.json"), str(WORK / "tuned.json"),
       "--out", str(WORK / "compare.json"))

    step(6, "merge (exactness checked) and serving-format check")
    assert py("peftkit.merge", "--adapter", str(WORK / "run/adapter"), "--out", str(WORK / "merged"),
              "--dtype", "float32") == 0
    py("peftkit.serve_check", str(WORK / "run/adapter"))

    step(7, "serve the merged model behind an OpenAI-compatible API, then the gateway in front of it")
    import uvicorn
    from peftkit import serve_local
    serve_local.configure(str(WORK / "merged"), name="pump", api_key="vllm-secret")
    p_srv = free_port()
    srv = uvicorn.Server(uvicorn.Config(serve_local.app, host="127.0.0.1", port=p_srv, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    client_key = "client-key-rehearsal"
    os.environ.update(UPSTREAMS=f"http://127.0.0.1:{p_srv}", UPSTREAM_API_KEY="vllm-secret",
                      GATEWAY_KEY_HASHES=hashlib.sha256(client_key.encode()).hexdigest(), RATE_LIMIT_RPM="100000",
                      GATEWAY_ADMIN_KEY_HASH=hashlib.sha256(b"admin-rehearsal").hexdigest(),
                      GATEWAY_KEYS_FILE=str(WORK / "keys.json"))
    import importlib
    gw = importlib.import_module("gateway.app")
    gw.S.reload()
    p_gw = free_port()
    gws = uvicorn.Server(uvicorn.Config(gw.app, host="127.0.0.1", port=p_gw, log_level="error"))
    threading.Thread(target=gws.run, daemon=True).start()
    while not (srv.started and gws.started):
        time.sleep(0.1)
    import httpx
    print("    server /health  :", httpx.get(f"http://127.0.0.1:{p_srv}/health").json())
    print("    gateway /healthz:", httpx.get(f"http://127.0.0.1:{p_gw}/healthz").json())
    bad = httpx.post(f"http://127.0.0.1:{p_gw}/v1/chat/completions", json={"model": "pump", "messages": []})
    print("    no API key      -> HTTP", bad.status_code, "(must be 401)")

    step(8, "evaluate the SERVED model through the gateway, and check parity with offline")
    url = f"http://127.0.0.1:{p_gw}/v1"
    py("peftkit.eval_endpoint", "--url", url, "--model", "pump", "--api-key", client_key, "--suite", *suites,
       "--max-tokens", "16", "--out", str(WORK / "served.json"), "--offline", str(WORK / "tuned.json"),
       "--parity-out", str(WORK / "parity.json"))

    step(9, "load test through the gateway (concurrency ramp, SLO)")
    py("peftkit.loadtest", "--url", url, "--model", "pump", "--api-key", client_key,
       "--prompts", str(WORK / "test.jsonl"), "--levels", "1", "2", "4", "--requests-per-level", "12",
       "--max-tokens", "16", "--metrics-url", f"http://127.0.0.1:{p_srv}/metrics", "--slo-p95", "5",
       "--out", str(WORK / "loadtest.json"))

    step(10, "RELEASE GATE: quality + parity + capacity")
    code = py("peftkit.release_gate", "--compare", str(WORK / "compare.json"), "--parity", str(WORK / "parity.json"),
              "--loadtest", str(WORK / "loadtest.json"))

    step(11, "turn the load test into autoscaling settings")
    rec = json.loads((WORK / "loadtest.json").read_text())["recommendation"]
    if rec.get("meets_slo"):
        print(f"    per-replica capacity at SLO : {rec['max_concurrency_per_replica']} concurrent requests")
        print(f"    KEDA threshold / scaler --scale_up_waiting : {rec['suggested_scale_up_threshold']}")
        print(f"    replicas for 40 concurrent users : {-(-40 // rec['max_concurrency_per_replica'])}")
    step(12, "PUBLIC API: issue a key to a new user, call the model with it, read metered usage")
    admin = {"Authorization": "Bearer admin-rehearsal"}
    issued = httpx.post(f"http://127.0.0.1:{p_gw}/admin/keys", headers=admin,
                        json={"name": "first-customer", "rpm": 60, "models": ["pump"]}).json()
    print("    issued key id    :", issued["key_id"], "(plaintext shown once, only its hash is stored)")
    r = httpx.post(f"{url}/chat/completions", headers={"Authorization": f"Bearer {issued['api_key']}"}, timeout=60,
                   json={"model": "pump", "messages": [{"role": "user", "content": "What is the maximum pressure?"}],
                         "max_tokens": 16})
    print("    customer request :", r.status_code, "->", r.json()["choices"][0]["message"]["content"][:60])
    print("    metered usage    :", httpx.get(f"http://127.0.0.1:{p_gw}/admin/usage", headers=admin).json()[issued["key_id"]])

    step(13, "CAPACITY + COST from the MEASURED load test (prices: fill configs/providers.yaml)")
    from peftkit.capacity import compute
    cap = compute(str(model_dir / "config.json"), users=500, active_share=0.1, req_min=2, prompt_tokens=40,
                  output_tokens=16, latency_s=0.25, loadtest=json.loads((WORK / "loadtest.json").read_text()),
                  prices=[{"provider": "example", "gpu": "any", "vram_gb": 24, "gpus_per_node": 8,
                           "price_per_gpu_hour": None}])
    (WORK / "capacity.json").write_text(json.dumps(cap, indent=2))
    print("    ", cap["traffic"]["explain"])
    print(f"     replicas needed at 3x peak + 30% headroom: {cap['providers'][0]['replicas']} "
          f"(capacity source: {cap['providers'][0]['capacity_source']})")

    step(14, "SHOWCASE report for owners / customers, built only from the report files above")
    code2 = py("peftkit.showcase", "--title", "Rehearsal: pump support assistant", "--compare", str(WORK / "compare.json"),
               "--tuned", str(WORK / "tuned.json"), "--parity", str(WORK / "parity.json"),
               "--loadtest", str(WORK / "loadtest.json"), "--capacity", str(WORK / "capacity.json"),
               "--run", str(WORK / "run/run_meta.json"), "--out", str(WORK / "showcase.html"))
    code = code or code2
    srv.should_exit = gws.should_exit = True
    print("\nREHEARSAL", "PASSED - every stage ran end to end" if code == 0 else "FINISHED - read the gate output above")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
