"""Run a grid of training experiments and put the results in ONE table.

An experiment is only useful if you can compare it: same data, same eval, one variable changed.
Records: run_id, method, hyperparameters, trainable params, train/eval loss, seconds, peak VRAM,
adapter size on disk, and (optionally) an eval-suite score.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

from .config import deep_merge, load_config
from .train import run


def run_id(overrides: dict) -> str:
    return hashlib.sha1(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:8]


def dir_size_mb(path: str | Path) -> float:
    return round(sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1024**2, 2)


def expand(grid: dict) -> list[dict]:
    """{"method.name": ["lora","dora"], "method.r": [8,16]} -> list of nested override dicts."""
    keys = list(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        ov: dict = {}
        for k, v in zip(keys, combo):
            node = ov
            parts = k.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = v
        out.append(ov)
    return out


def run_grid(base_configs: list[str], grid: dict, out_dir: str, suites: dict | None = None,
             extra: dict | None = None) -> list[dict]:
    rows = []
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for ov in expand(grid):
        rid = run_id(ov)
        full = deep_merge(ov, {"train": {"output_dir": str(out_root / rid)}, "run_name": rid})
        if extra:
            full = deep_merge(full, extra)
        cfg = load_config(*base_configs, overrides=full)
        meta = run(cfg)
        row = {"run_id": rid, "method": cfg.method.name, "r": cfg.method.r, "alpha": cfg.method.alpha,
               "lr": cfg.train.lr, "epochs": cfg.train.epochs, "batch": cfg.train.per_device_batch_size,
               "grad_accum": cfg.train.grad_accum, "seq_len": cfg.model.max_seq_length,
               "trainable": meta["trainable_params"], "steps": meta["global_step"],
               "train_loss": round(meta["train_loss"], 4), "seconds": meta["train_seconds"],
               "peak_vram_gb": meta.get("peak_vram_gb"), "gpu": meta["versions"].get("gpu", "cpu"),
               "adapter_mb": dir_size_mb(Path(cfg.train.output_dir) / "adapter"),
               "eval_loss": round(meta.get("eval", {}).get("eval_loss", float("nan")), 4) if meta.get("eval") else None,
               "eval_token_acc": round(meta.get("eval", {}).get("eval_mean_token_accuracy", float("nan")), 4) if meta.get("eval") else None}
        if suites:
            from .evaluate import evaluate

            rep = evaluate(str(Path(cfg.train.output_dir) / "adapter"), suites)
            row["eval_score"] = round(rep["overall"], 4)
            for name, s in rep["suites"].items():
                row[f"score_{name}"] = round(s["score"], 4)
        rows.append(row)
        write(rows, out_root)
    return rows


def write(rows: list[dict], out_root: Path) -> None:
    keys = sorted({k for r in rows for k in r})
    with (out_root / "experiments.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out_root / "experiments.md").write_text(as_markdown(rows))


def as_markdown(rows: list[dict], columns: list[str] | None = None) -> str:
    cols = columns or ["run_id", "method", "r", "lr", "trainable", "train_loss", "eval_loss",
                       "eval_token_acc", "eval_score", "seconds", "peak_vram_gb", "adapter_mb"]
    cols = [c for c in cols if any(r.get(c) is not None for r in rows)]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run an experiment grid")
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--grid", required=True, help='JSON, e.g. \'{"method.r": [8,16,32]}\'')
    ap.add_argument("--out", default="outputs/experiments")
    ap.add_argument("--suite", nargs="*", default=[], help="name=path.jsonl for generation-based scoring")
    a = ap.parse_args(argv)
    suites = dict(s.split("=", 1) for s in a.suite) or None
    rows = run_grid(a.config, json.loads(a.grid), a.out, suites)
    print(as_markdown(rows))


if __name__ == "__main__":
    main()
