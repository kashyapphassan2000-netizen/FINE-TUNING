"""A generic training loop for ANY model and ANY loss (when Trainer/TRL does not fit).

You supply two functions:
  loss_fn(model, batch) -> scalar loss tensor      (the training objective, STEP 3)
  eval_fn(model) -> dict of metrics                 (optional, STEP 17)
The loop adds: AMP (bf16/fp16), gradient accumulation, clipping, a scheduler, NaN guard,
checkpointing, and a gradient sanity check on the first step.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from .trace import gradient_report


def train(model, loader, loss_fn, *, lr=2e-4, weight_decay=0.0, max_steps=100, accum=1, clip=1.0,
          warmup=0.03, amp_dtype: str | None = None, eval_fn=None, eval_every=0, ckpt_dir: str | None = None,
          ckpt_every=0, log_every=10, device=None) -> dict:
    device = device or next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters: nothing was injected/unfrozen (see targets + lora.inject)")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    warm = max(1, int(warmup * max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, max_steps - warm))))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp_dtype or "", None)
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16 and device.type == "cuda"))
    history, first_grad_check, step, t0 = [], None, 0, time.time()
    model.train()
    it = iter(loader)
    while step < max_steps:
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                loss = loss_fn(model, batch) / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: lower lr, check labels/masks, use bf16")
            scaler.scale(loss).backward()
            total += float(loss)
        if first_grad_check is None:
            first_grad_check = gradient_report(model)
            if not first_grad_check["ok"]:
                raise RuntimeError(f"gradient check failed on step 0: {first_grad_check}")
        scaler.unscale_(opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, clip)) if clip else None
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        rec = {"step": step, "loss": round(total, 6), "grad_norm": round(gnorm, 4) if gnorm else None,
               "lr": sched.get_last_lr()[0]}
        if eval_fn and eval_every and step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                rec.update(eval_fn(model))
            model.train()
        if step % log_every == 0 or step == 1 or step == max_steps:
            history.append(rec)
        if ckpt_dir and ckpt_every and step % ckpt_every == 0:
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"step": step, "state": {n: p for n, p in model.state_dict().items() if "lora_" in n}},
                       Path(ckpt_dir) / f"step{step}.pt")
    model.eval()
    return {"history": history, "first_step_gradients": first_grad_check, "seconds": round(time.time() - t0, 2)}
