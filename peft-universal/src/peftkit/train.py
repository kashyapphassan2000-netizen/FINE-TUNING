"""One entry point for LoRA / QLoRA / DoRA / AdaLoRA / LoRA+.

python -m peftkit.train --config configs/base.yaml configs/methods/qlora.yaml configs/models/my-model.yaml \
    --set train.max_steps=20
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import get_peft_model
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from . import __version__
from .callbacks import AdaLoraCallback, count_active_rank
from .config import RunCfg, load_config, parse_cli_overrides
from .modeling import load_model, load_tokenizer
from .peft_factory import build_peft_config, optimizer_cls_for, total_optimizer_steps

log = logging.getLogger("peftkit")


def versions() -> dict:
    import datasets, peft, transformers, trl  # noqa: E401

    v = {"peftkit": __version__, "python": platform.python_version(), "torch": torch.__version__,
         "transformers": transformers.__version__, "peft": peft.__version__, "trl": trl.__version__,
         "datasets": datasets.__version__, "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        v["gpu"] = torch.cuda.get_device_name(0)
        v["gpu_count"] = torch.cuda.device_count()
    return v


def to_prompt_completion(row: dict) -> dict:
    msgs = row["messages"]
    if msgs[-1]["role"] != "assistant":
        raise ValueError("every example must end with an assistant turn")
    return {"prompt": msgs[:-1], "completion": [msgs[-1]]}


def build_sft_args(cfg: RunCfg, has_eval: bool) -> SFTConfig:
    t = cfg.train
    cuda = torch.cuda.is_available()
    bf16 = cuda and cfg.model.dtype == "bfloat16" and torch.cuda.is_bf16_supported()
    fp16 = cuda and cfg.model.dtype == "float16"
    return SFTConfig(
        output_dir=t.output_dir,
        run_name=cfg.run_name,
        num_train_epochs=t.epochs,
        max_steps=t.max_steps,
        per_device_train_batch_size=t.per_device_batch_size,
        per_device_eval_batch_size=t.per_device_batch_size,
        gradient_accumulation_steps=t.grad_accum,
        learning_rate=t.lr,
        lr_scheduler_type=t.scheduler,
        warmup_steps=t.warmup_steps,
        weight_decay=t.weight_decay,
        max_grad_norm=t.max_grad_norm,
        optim=t.optim,
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=t.logging_steps,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=t.eval_steps if has_eval else None,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        report_to=[] if t.report_to == "none" else [t.report_to],
        seed=t.seed,
        bf16=bf16,
        fp16=fp16,
        use_cpu=not cuda,
        max_length=cfg.model.max_seq_length,
        packing=cfg.data.packing,
        assistant_only_loss=cfg.data.assistant_only_loss,
        completion_only_loss=True if cfg.data.completion_only_loss else None,
        dataloader_num_workers=2 if cuda else 0,
    )


def run(cfg: RunCfg, resume: str | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(cfg.train.seed)
    out = Path(cfg.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = {"train": cfg.data.train_file}
    if cfg.data.eval_file and Path(cfg.data.eval_file).exists() and Path(cfg.data.eval_file).stat().st_size > 0:
        files["eval"] = cfg.data.eval_file
    ds = load_dataset("json", data_files=files)
    ds = ds.select_columns(["messages"])  # drop meta columns so the collator only sees chat data
    if cfg.data.completion_only_loss:
        # conversational prompt-completion: TRL masks the prompt itself, so no {% generation %} markers are needed.
        # Everything before the last assistant turn is context (not trained); the last turn + its EOS is trained.
        ds = ds.map(to_prompt_completion, remove_columns=["messages"])

    tok = load_tokenizer(cfg)
    model = load_model(cfg)

    world = int(os.environ.get("WORLD_SIZE", 1))
    steps = total_optimizer_steps(len(ds["train"]), cfg.train.per_device_batch_size, cfg.train.grad_accum,
                                  cfg.train.epochs, cfg.train.max_steps, world)
    if cfg.method.name == "adalora" and cfg.data.packing and cfg.train.max_steps <= 0:
        raise ValueError("AdaLoRA + packing: set train.max_steps (packed dataset length is unknown up front)")
    if cfg.method.name == "full":
        # full fine-tuning: no adapter, every weight trained (16 B/param with AdamW: shard it with ZeRO-3/FSDP)
        for prm in model.parameters():
            prm.requires_grad_(True)
        trainable = total = sum(prm.numel() for prm in model.parameters())
    else:
        peft_cfg = build_peft_config(cfg.method, total_steps=steps)
        model = get_peft_model(model, peft_cfg)
        trainable, total = model.get_nb_trainable_parameters()
    log.info("method=%s trainable=%s total=%s (%.4f%%) planned_steps=%s", cfg.method.name, f"{trainable:,}",
             f"{total:,}", 100 * trainable / total, steps)

    callbacks, optimizers = [], (None, None)
    ada_cb = None
    if cfg.method.name == "adalora":
        ada_cb = AdaLoraCallback()
        callbacks.append(ada_cb)
    if cfg.method.name == "loraplus":
        from peft.optimizers import create_loraplus_optimizer

        opt = create_loraplus_optimizer(
            model=model, optimizer_cls=optimizer_cls_for(cfg.train.optim), lr=cfg.train.lr,
            loraplus_lr_ratio=cfg.method.loraplus_lr_ratio, weight_decay=cfg.train.weight_decay,
        )
        optimizers = (opt, None)  # scheduler is still built by the Trainer from args

    trainer = SFTTrainer(
        model=model,
        args=build_sft_args(cfg, "eval" in ds),
        train_dataset=ds["train"],
        eval_dataset=ds.get("eval"),
        processing_class=tok,
        callbacks=callbacks,
        optimizers=optimizers,
    )
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - t0

    # PEFT: adapter_model.safetensors + adapter_config.json in out/adapter. Full FT: a complete model in out/model
    adapter_dir = out / ("model" if cfg.method.name == "full" else "adapter")
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))

    meta = {
        "run_name": cfg.run_name,
        "saved_to": str(adapter_dir),
        "method": cfg.method.name,
        "base_model": cfg.model.name_or_path,
        "trainable_params": trainable,
        "total_params": total,
        "planned_steps": steps,
        "global_step": trainer.state.global_step,
        "train_loss": result.training_loss,
        "train_seconds": round(elapsed, 1),
        "log_history": trainer.state.log_history,
        "versions": versions(),
        "config": cfg.model_dump(),
    }
    if torch.cuda.is_available():
        meta["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    if ada_cb is not None:
        meta["adalora_rank_history"] = ada_cb.rank_history
        meta["adalora_final_active_rank"] = count_active_rank(model)
    if "eval" in ds:
        if cfg.method.name == "adalora":
            # AdaLoRA adds an orthogonality penalty to outputs.loss (train AND eval).
            # Report the pure task loss so methods are comparable.
            pc = model.peft_config[model.base_model.trainable_adapter_name]
            saved, pc.orth_reg_weight = pc.orth_reg_weight, 0.0
            meta["eval"] = trainer.evaluate()
            pc.orth_reg_weight = saved
        else:
            meta["eval"] = trainer.evaluate()
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    if cfg.hub.push and cfg.hub.repo_id:
        model.push_to_hub(cfg.hub.repo_id, private=cfg.hub.private)
        tok.push_to_hub(cfg.hub.repo_id, private=cfg.hub.private)
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Universal PEFT trainer")
    ap.add_argument("--config", nargs="+", required=True, help="YAML files merged left to right")
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.lr=1e-4 method.r=32")
    ap.add_argument("--resume", default=None, help="checkpoint dir, or 'true' for latest")
    a = ap.parse_args(argv)
    cfg = load_config(*a.config, overrides=parse_cli_overrides(a.set))
    resume = True if a.resume == "true" else a.resume
    meta = run(cfg, resume)
    print(json.dumps({k: meta[k] for k in ("method", "trainable_params", "global_step", "train_loss")}))


if __name__ == "__main__":
    main()
