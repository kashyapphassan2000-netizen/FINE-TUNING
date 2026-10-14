"""From 'this model' to 'this many GPUs, from this provider, for this many users, at this cost'.

The maths, in order (every line is printed so you can check it):
  1. TRAIN memory       -> GPUs to fine-tune (peftkit.plan formulas)
  2. SERVE memory       -> GPUs per replica = weights + KV budget, rounded to a tensor-parallel size
  3. LOAD / COLD START  -> download time + load time = how early autoscaling must react
  4. USERS -> TRAFFIC   -> Little's law: concurrent requests L = arrival rate (req/s) x latency W (s)
  5. REPLICAS           -> ceil(L_peak x (1 + headroom) / capacity_per_replica)
  6. COST               -> GPU-hours x YOUR price; cost per 1M tokens; per user per month
capacity_per_replica should come from peftkit.loadtest (latency SLO). The KV-budget number is only an
upper bound: memory can hold more sequences than the GPU can serve within your latency target.

python -m peftkit.capacity --model meta-llama/Llama-3.1-8B-Instruct --users 2000 --active-share 0.1 \
   --requests-per-active-user-per-min 2 --prompt-tokens 600 --output-tokens 250 --latency-s 4 \
   --loadtest reports/loadtest.json --prices configs/providers.yaml
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from .plan import BYTES, GIB, plan

HOURS_PER_MONTH = 730


def little_law(users: int, active_share: float, req_per_active_min: float, latency_s: float) -> dict:
    active = users * active_share
    lam = active * req_per_active_min / 60.0              # requests per second
    L = lam * latency_s                                   # average concurrent requests in the system
    return {"active_users": active, "arrival_rate_rps": round(lam, 3), "avg_concurrency": round(L, 2),
            "explain": f"L = lambda x W = {lam:.3f} req/s x {latency_s} s = {L:.2f} concurrent requests"}


def load_time(weights_gb: float, net_gbps: float = 1.0, disk_gb_s: float = 2.0, init_s: float = 60.0) -> dict:
    download = weights_gb * 8 / max(net_gbps, 1e-6)       # GB -> Gb, divided by Gbit/s
    load = weights_gb / max(disk_gb_s, 1e-6)
    return {"download_s": round(download), "disk_to_gpu_s": round(load), "engine_init_s": init_s,
            "cold_start_s": round(download + load + init_s),
            "warm_start_s": round(load + init_s),
            "explain": "cold = download(weights x 8 / network Gbit/s) + load(weights / disk GB/s) + engine init; "
                       "bake weights into the image/volume to turn cold starts into warm starts"}


def replicas(avg_conc: float, peak_factor: float, cap_per_replica: int, headroom: float, min_replicas: int = 1) -> dict:
    peak = avg_conc * peak_factor
    n = max(min_replicas, math.ceil(peak * (1 + headroom) / max(1, cap_per_replica)))
    return {"peak_concurrency": round(peak, 2), "replicas": n,
            "explain": f"ceil({peak:.2f} x (1 + {headroom}) / {cap_per_replica}) = {n}"}


def compute(model: str, users: int, active_share: float, req_min: float, prompt_tokens: int, output_tokens: int,
            latency_s: float, peak_factor: float = 3.0, headroom: float = 0.3, serve_dtype: str = "bf16",
            loadtest: dict | None = None, prices: list | None = None, train_seq: int = 4096,
            dataset_tokens: float | None = None, epochs: float = 1.0, train_tps: float | None = None,
            net_gbps: float = 1.0) -> dict:
    ctx = prompt_tokens + output_tokens
    p = plan(model, ["lora", "qlora"], train_seq, 1, 16, dataset_tokens, epochs, train_tps, None,
             serve_dtype, ctx, 1)
    weights_gb = p["serving"]["weights_gb"]
    kv_mb = p["serving"]["kv_mb_per_token"]
    traffic = little_law(users, active_share, req_min, latency_s)
    measured = (loadtest or {}).get("recommendation", {}).get("max_concurrency_per_replica")
    rows = []
    for g in prices or []:
        vram = g["vram_gb"]
        tr = {t["method"]: t for t in p["training"]}
        train_gpus = {m: max(1, math.ceil(tr[m]["total_gb"] / (vram * 0.9))) for m in tr}
        need = weights_gb / 0.9 + 1.5
        tp = 1
        while tp * vram < need and tp < g.get("gpus_per_node", 8):
            tp *= 2
        fits = tp * vram >= need
        kv_budget_gb = max(0.0, tp * vram * 0.9 - weights_gb - 1.5)
        kv_cap = int(kv_budget_gb * 1024 / max(kv_mb * ctx, 1e-9))         # sequences that FIT in memory
        cap = measured or kv_cap
        rep = replicas(traffic["avg_concurrency"], peak_factor, max(1, cap), headroom) if fits and cap else None
        price = g.get("price_per_gpu_hour")
        row = {"provider": g["provider"], "gpu": g["gpu"], "vram_gb": vram, "fits_on_tp": tp if fits else None,
               "train_gpus": train_gpus, "kv_capacity_upper_bound": kv_cap,
               "capacity_used": cap, "capacity_source": "load test (SLO)" if measured else "KV memory upper bound",
               "replicas": rep["replicas"] if rep else None, "gpus_total": rep["replicas"] * tp if rep else None}
        if price and rep:
            gph = rep["replicas"] * tp
            monthly = gph * price * HOURS_PER_MONTH
            tokens_month = traffic["arrival_rate_rps"] * output_tokens * 3600 * HOURS_PER_MONTH
            row.update({"serve_cost_month": round(monthly, 2),
                        "cost_per_1M_output_tokens": round(monthly / max(tokens_month / 1e6, 1e-9), 4),
                        "cost_per_user_month": round(monthly / max(users, 1), 4)})
            if p.get("compute", {}).get("gpu_hours_per_gpu_rate"):
                row["train_cost_once"] = round(p["compute"]["gpu_hours_per_gpu_rate"] * price, 2)
        rows.append(row)
    return {"model": model, "weights_gb": round(weights_gb, 2), "kv_mb_per_token": kv_mb,
            "context_tokens": ctx, "traffic": traffic, "load": load_time(weights_gb, net_gbps),
            "measured_capacity_per_replica": measured, "providers": rows,
            "warning": None if measured else "no load test given: replica counts use the KV-memory upper bound and "
                                             "WILL under-provision; run peftkit.loadtest and pass --loadtest"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GPUs, replicas and cost for fine-tuning and serving")
    ap.add_argument("--model", required=True)
    ap.add_argument("--users", type=int, required=True, help="registered users")
    ap.add_argument("--active-share", type=float, default=0.1, help="share active at the busy hour")
    ap.add_argument("--requests-per-active-user-per-min", type=float, default=1.0)
    ap.add_argument("--prompt-tokens", type=int, default=500)
    ap.add_argument("--output-tokens", type=int, default=200)
    ap.add_argument("--latency-s", type=float, default=4.0, help="average end-to-end latency per request")
    ap.add_argument("--peak-factor", type=float, default=3.0)
    ap.add_argument("--headroom", type=float, default=0.3)
    ap.add_argument("--serve-dtype", default="bf16", choices=list(BYTES))
    ap.add_argument("--loadtest", default=None)
    ap.add_argument("--prices", default="configs/providers.yaml")
    ap.add_argument("--dataset-tokens", type=float, default=None)
    ap.add_argument("--train-tps", type=float, default=None, help="measured training tokens/sec per GPU")
    ap.add_argument("--net-gbps", type=float, default=1.0, help="download speed to the serving node")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    lt = json.loads(Path(a.loadtest).read_text()) if a.loadtest else None
    prices = yaml.safe_load(Path(a.prices).read_text())
    r = compute(a.model, a.users, a.active_share, a.requests_per_active_user_per_min, a.prompt_tokens,
                a.output_tokens, a.latency_s, a.peak_factor, a.headroom, a.serve_dtype, lt, prices,
                dataset_tokens=a.dataset_tokens, train_tps=a.train_tps, net_gbps=a.net_gbps)
    if a.json:
        print(json.dumps(r, indent=2))
        return
    t, ld = r["traffic"], r["load"]
    print(f"MODEL {r['model']}  weights {r['weights_gb']} GB ({a.serve_dtype})  KV {r['kv_mb_per_token']:.4f} MB/token  "
          f"context {r['context_tokens']}")
    print(f"TRAFFIC {a.users} users x {a.active_share} active x {a.requests_per_active_user_per_min}/min -> "
          f"{t['arrival_rate_rps']} req/s ;  {t['explain']}")
    print(f"LOAD    download {ld['download_s']} s + disk {ld['disk_to_gpu_s']} s + init {ld['engine_init_s']} s = "
          f"cold start {ld['cold_start_s']} s (warm {ld['warm_start_s']} s)")
    if r["warning"]:
        print("WARNING", r["warning"])
    print(f"\n{'provider':<11}{'gpu':<15}{'VRAM':>5}{'TP':>4}{'train LoRA/QLoRA':>18}{'cap/replica':>12}"
          f"{'replicas':>9}{'GPUs':>6}{'$/month':>11}{'$/1M out':>10}")
    for x in r["providers"]:
        tg = f"{x['train_gpus'].get('lora')}/{x['train_gpus'].get('qlora')}"
        print(f"{x['provider']:<11}{x['gpu']:<15}{x['vram_gb']:>5}{str(x['fits_on_tp'] or '-'):>4}{tg:>18}"
              f"{x['capacity_used']:>12}{str(x['replicas'] or '-'):>9}{str(x['gpus_total'] or '-'):>6}"
              f"{str(x.get('serve_cost_month', 'price?')):>11}{str(x.get('cost_per_1M_output_tokens', '-')):>10}")
    print("\nprice? = fill price_per_gpu_hour in configs/providers.yaml with the rate YOUR account is quoted.")


if __name__ == "__main__":
    main()
