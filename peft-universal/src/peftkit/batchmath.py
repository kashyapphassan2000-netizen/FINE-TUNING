"""Scaling from 1 GPU to many: keep the GLOBAL batch fixed, change the accumulation.

  global batch = per_device_batch x gradient_accumulation_steps x world_size
  world_size   = GPUs per node x nodes

If you move a working 1-GPU recipe (batch 2, accumulation 8 -> global 16) onto 8 GPUs without
changing accumulation, the global batch becomes 128: a different optimisation problem, and your
tuned learning rate is no longer tuned. This tool gives the accumulation that keeps it equal.

python -m peftkit.batchmath --global-batch 16 --per-device 2 --gpus-per-node 8 --nodes 2
"""
from __future__ import annotations

import argparse
import math


def plan(global_batch: int, per_device: int, gpus_per_node: int, nodes: int = 1) -> dict:
    world = gpus_per_node * nodes
    exact = global_batch / (per_device * world)
    accum = max(1, round(exact))
    achieved = per_device * accum * world
    out = {"world_size": world, "per_device_batch": per_device, "gradient_accumulation_steps": accum,
           "global_batch_target": global_batch, "global_batch_achieved": achieved,
           "exact": achieved == global_batch}
    if exact < 1:
        smaller = max(1, global_batch // world)
        out["warning"] = (f"target global batch {global_batch} is smaller than per_device x world "
                          f"({per_device} x {world}); lower per_device_batch to {smaller} or raise the "
                          f"global batch (and re-tune the learning rate)")
    elif not out["exact"]:
        out["warning"] = f"not exactly divisible: achieved {achieved} instead of {global_batch}"
    ratio = achieved / global_batch
    if abs(ratio - 1) > 1e-9:
        out["lr_note"] = (f"global batch changed by x{ratio:.2f}: re-tune the learning rate; a common "
                          f"starting point is to scale it by sqrt({ratio:.2f}) = x{math.sqrt(ratio):.2f} "
                          "and confirm on eval")
    out["steps_per_epoch"] = "dataset_rows / global_batch (fewer optimizer steps as the global batch grows)"
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="gradient accumulation for multi-GPU / multi-node")
    ap.add_argument("--global-batch", type=int, required=True)
    ap.add_argument("--per-device", type=int, required=True)
    ap.add_argument("--gpus-per-node", type=int, default=8)
    ap.add_argument("--nodes", type=int, default=1)
    a = ap.parse_args(argv)
    import json

    print(json.dumps(plan(a.global_batch, a.per_device, a.gpus_per_node, a.nodes), indent=2))


if __name__ == "__main__":
    main()
