"""STEP 9 of the flow: hyperparameter search, done cheaply and honestly.

"Play with the hyperparameters until the error is at the global minimum" is not achievable - the loss
surface of a neural network has no findable global minimum, and chasing training loss to zero is
memorisation (see the over/under-fitting chapter). What IS achievable:

    search a small space -> kill bad runs EARLY (successive halving) -> keep the best by EVAL score
    -> confirm the winner against seed noise -> stop when the gain is smaller than that noise.

python -m peftkit.hpo --config configs/base.yaml configs/methods/lora.yaml configs/models/m.yaml \
    --space '{"train.lr": [5e-5, 1e-4, 2e-4, 5e-4], "method.r": [8, 16, 32]}' --budget 12 --out outputs/hpo
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from .config import deep_merge, load_config
from .train import run


def sample_space(space: dict, n: int, seed: int = 0) -> list[dict]:
    """Random search: better than grid at the same budget when only some knobs matter."""
    rng = random.Random(seed)
    out, seen = [], set()
    for _ in range(n * 20):
        pick = {}
        for k, v in space.items():
            pick[k] = rng.choice(v) if isinstance(v, list) else rng.uniform(*v)
        key = json.dumps(pick, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(pick)
        if len(out) >= n:
            break
    return out


def nest(flat: dict) -> dict:
    ov: dict = {}
    for k, v in flat.items():
        node = ov
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return ov


def score_of(meta: dict) -> float:
    """Lower is better. Prefer eval loss; fall back to train loss with a penalty so it never wins."""
    ev = (meta.get("eval") or {})
    if "eval_loss" in ev:
        return float(ev["eval_loss"])
    return float(meta.get("train_loss", float("inf"))) + 1e6


def search(base_configs: list[str], space: dict, budget: int, out_dir: str, steps_per_rung: list[int] | None = None,
           keep_fraction: float = 0.5, extra: dict | None = None, seed: int = 0) -> dict:
    """Successive halving: every candidate gets a short run; survivors get longer ones."""
    rungs = steps_per_rung or [20, 60, 180]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cands = [{"id": f"c{i}", "params": p, "history": []} for i, p in enumerate(sample_space(space, budget, seed))]
    alive = cands
    for r, steps in enumerate(rungs):
        for c in alive:
            ov = deep_merge(nest(c["params"]),
                            {"train": {"max_steps": steps, "output_dir": str(out / f"{c['id']}_r{r}"),
                                       "save_steps": 10**9, "eval_steps": max(5, steps // 2)},
                             "run_name": f"{c['id']}_r{r}"})
            if extra:
                ov = deep_merge(ov, extra)
            meta = run(load_config(*base_configs, overrides=ov))
            c["history"].append({"rung": r, "steps": steps, "score": round(score_of(meta), 5),
                                 "train_loss": round(meta["train_loss"], 5)})
            c["score"] = c["history"][-1]["score"]
        alive.sort(key=lambda c: c["score"])
        keep = max(1, int(len(alive) * keep_fraction))
        killed = [c["id"] for c in alive[keep:]]
        alive = alive[:keep]
        (out / "hpo.json").write_text(json.dumps({"rung": r, "steps": steps, "alive": [c["id"] for c in alive],
                                                  "killed": killed, "candidates": cands}, indent=2))
        if len(alive) == 1:
            break
    best = alive[0]
    return {"best": best, "candidates": sorted(cands, key=lambda c: c.get("score", float("inf"))),
            "rungs": rungs, "note": "confirm the winner with 2-3 seeds: a gap smaller than seed noise is not a gap"}


def seed_noise(base_configs: list[str], params: dict, seeds=(0, 1, 2), steps: int = 180,
               extra: dict | None = None, out_dir: str = "outputs/hpo_seeds") -> dict:
    """How big is the difference between two runs that differ ONLY by seed? Anything smaller is noise."""
    scores = []
    for s in seeds:
        ov = deep_merge(nest(params), {"train": {"max_steps": steps, "seed": s, "save_steps": 10**9,
                                                 "output_dir": f"{out_dir}/s{s}"}})
        if extra:
            ov = deep_merge(ov, extra)
        scores.append(score_of(run(load_config(*base_configs, overrides=ov))))
    mean = sum(scores) / len(scores)
    spread = max(scores) - min(scores)
    return {"scores": [round(x, 5) for x in scores], "mean": round(mean, 5), "spread": round(spread, 5),
            "rule": "treat improvements smaller than the spread as noise, not progress"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Hyperparameter search with successive halving")
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--space", required=True, help='JSON: {"train.lr": [1e-4, 2e-4], "method.r": [8, 16]}')
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--rungs", nargs="*", type=int, default=[20, 60, 180])
    ap.add_argument("--out", default="outputs/hpo")
    ap.add_argument("--seeds", action="store_true", help="after the search, measure seed noise for the winner")
    a = ap.parse_args(argv)
    res = search(a.config, json.loads(a.space), a.budget, a.out, a.rungs)
    print(f"{'id':<6}{'score':>10}  params")
    for c in res["candidates"]:
        print(f"{c['id']:<6}{c.get('score', float('nan')):>10.5f}  {c['params']}")
    print("\nbest:", res["best"]["params"], "score", res["best"]["score"])
    if a.seeds:
        print("seed noise:", json.dumps(seed_noise(a.config, res["best"]["params"]), indent=2))
    print(res["note"])


if __name__ == "__main__":
    main()
