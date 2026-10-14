"""THE UNIVERSAL PIPELINE: one object that derives and runs the whole thing for ANY model.

    p = UniversalPipeline(model, sample_batch, model_id=..., task=...)
    print(p.plan())          # derived: modality, contracts, loss, targets, processor, task, metrics
    p.adapt()                # adapter strategy chosen per layer type, with graceful degradation
    p.smoke_test()           # gradients reach the adapters + one batch overfits
    p.fit(loader)            # modality-independent training loop
    p.evaluate(eval_fn)      # your metric, from the task registry
    p.export("out/")         # adapter + merged weights, merge proven exact
    p.serving_plan()         # engine by architecture

Everything except the DATA, the METRIC and (for new modalities) the PROCESSOR is derived.
Those three are the parts no framework can invent for you, and the object says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from . import lora as L
from .collate import collate, make_loader
from .derive import derive
from .loop import train
from .tasks import TASKS
from .trace import gradient_report


@dataclass
class UniversalPipeline:
    model: nn.Module
    sample_batch: dict | tuple
    model_id: str | None = None
    task: str | None = None
    pairs: bool = False
    loss_fn: object = None
    state: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- derive
    def plan(self) -> dict:
        d = derive(self.model, self.sample_batch, self.model_id, self.task, self.pairs)
        self.state["derived"] = d
        self.loss_fn = self.loss_fn or d["loss_fn"]
        if self.task is None and d.get("7_task"):
            self.task = d["7_task"]["chosen_task"]
        return {k: v for k, v in d.items() if k not in ("loss_fn", "processor")}

    @property
    def processor(self):
        return self.state.get("derived", {}).get("processor")

    def loader(self, records: list[dict], label_fn=None, batch_size: int = 8, pad_id: int = 0):
        return make_loader(records, self.processor, label_fn, batch_size, pad_id=pad_id)

    # ---------------------------------------------------------------- adapt (with degradation)
    def adapt(self, r: int = 8, alpha: float = 16, train_heads: bool = True, min_trainable_pct: float = 0.01) -> dict:
        """Adapter strategy per layer type, degrading gracefully instead of failing:
        Linear/Conv -> LoRA; packed parameters (LSTM, fused MoE experts, MHA in_proj) -> selective unfreeze;
        nothing adaptable -> unfreeze the smallest sensible blocks and SAY so."""
        d = self.state.get("derived") or derive(self.model, self.sample_batch, self.model_id, self.task, self.pairs)
        tgt = d["5_trainable"]
        from .targets import discover

        full = discover(self.model, self.sample_batch)
        heads = full["train_fully_or_freeze"] if train_heads else []
        info = L.inject(self.model, full["lora_targets"] + full["conv_candidates"], r=r, alpha=alpha,
                        train_also=heads)
        info["strategy"] = "LoRA on Linear/Conv" + (" + heads trained" if heads else "")
        called = L.verify_called(self.model, self.sample_batch)
        info["not_called_via_forward"] = called["not_called"]
        if info["trainable_pct"] < min_trainable_pct or not info["wrapped"]:
            packed = [n for n, m in self.model.named_modules()
                      if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN, nn.MultiheadAttention))]
            for n in packed:
                for p in self.model.get_submodule(n).parameters():
                    p.requires_grad_(True)
            info["degraded_to"] = ("selective unfreezing of packed-parameter modules "
                                   f"({len(packed)} modules): LoRA could not reach them")
            info["trainable"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            info["trainable_pct"] = round(100 * info["trainable"] / max(1, info["total"]), 3)
        self.state["adapt"] = info
        return info

    # ---------------------------------------------------------------- verify
    def smoke_test(self, batch=None, steps: int = 30, lr: float = 5e-3) -> dict:
        """The universal proof that a pipeline is wired: gradients flow AND one batch overfits."""
        b = batch if batch is not None else self.sample_batch
        if self.loss_fn is None:
            raise RuntimeError("no loss: derivation said CUSTOM ENGINEERING REQUIRED. Pass loss_fn=...")
        self.model.zero_grad(set_to_none=True)
        self.loss_fn(self.model, b).backward()
        grads = gradient_report(self.model)
        self.model.zero_grad(set_to_none=True)
        before = float(self.loss_fn(self.model, b))
        train(self.model, [b], self.loss_fn, lr=lr, max_steps=steps, log_every=steps, warmup=0.1)
        after = float(self.loss_fn(self.model, b))
        res = {"gradients_ok": grads["ok"], "unexplained_zero_grads": len(grads["unexplained_zero_grad"]),
               "zero_at_init_because_lora_B_is_zero": grads["zero_because_lora_b_is_zero_at_init"],
               "loss_before": round(before, 4), "loss_after": round(after, 4),
               "overfits_one_batch": after < before * 0.9,
               "verdict": "READY" if grads["ok"] and after < before * 0.9 else "NOT READY"}
        self.state["smoke"] = res
        return res

    # ---------------------------------------------------------------- run
    def fit(self, loader, **kw) -> dict:
        if self.loss_fn is None:
            raise RuntimeError("no loss_fn: supply one (see plan()['4_loss'])")
        res = train(self.model, loader, self.loss_fn, **kw)
        self.state["fit"] = res
        return res

    def evaluate(self, eval_fn) -> dict:
        self.model.eval()
        with torch.no_grad():
            out = eval_fn(self.model)
        self.state["eval"] = out
        return out

    def metrics_to_report(self) -> list:
        return TASKS[self.task].metrics if self.task in TASKS else ["define the metric before training"]

    def export(self, path: str, merge: bool = True) -> dict:
        from pathlib import Path

        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(L.adapter_state_dict(self.model), Path(path) / "adapter.pt")
        out = {"adapter": str(Path(path) / "adapter.pt")}
        if merge:
            b = self.sample_batch
            self.model.eval()
            with torch.no_grad():
                ref = self.model(**b) if isinstance(b, dict) else self.model(*b)
                ref = ref.logits if hasattr(ref, "logits") else (ref if torch.is_tensor(ref) else None)
                out["merged_layers"] = L.merge_all(self.model)
                if ref is not None:
                    got = self.model(**b) if isinstance(b, dict) else self.model(*b)
                    got = got.logits if hasattr(got, "logits") else got
                    out["merge_max_abs_diff"] = float((ref - got).abs().max())
            torch.save(self.model.state_dict(), Path(path) / "merged.pt")
        self.state["export"] = out
        return out

    def serving_plan(self) -> str:
        from .adapter import UniversalModelAdapter

        return UniversalModelAdapter(self.model, self.sample_batch).serve()

    def report(self) -> dict:
        return {"task": self.task, "metrics_to_report": self.metrics_to_report(), **self.state.get("smoke", {}),
                "adapt": {k: v for k, v in self.state.get("adapt", {}).items() if k != "skipped"},
                "export": self.state.get("export"), "missing": self.state.get("derived", {}).get("missing", [])}
