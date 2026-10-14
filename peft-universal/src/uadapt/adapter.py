"""STEP 13: UniversalModelAdapter - one object that walks the whole methodology for any nn.Module.

Automated: inspection, classification evidence, forward trace, loss hints, target discovery, strategy
ranking, LoRA injection, gradient validation, a generic training loop, export (adapter or merged).
NOT automated (needs you): the training objective for a new task, the dataset, the evaluation suite,
the serving stack, and every judgement call the report flags as a risk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from . import inspect_model as im
from . import lora as L
from . import loop, strategy, targets, trace


@dataclass
class UniversalModelAdapter:
    model: nn.Module
    sample_batch: dict | tuple
    config: object = None
    notes: dict = field(default_factory=dict)

    def inspect(self) -> dict:
        self.notes["params"] = im.param_counts(self.model)
        self.notes["signature"] = im.forward_signature(self.model)
        return {"params": self.notes["params"], "signature": self.notes["signature"],
                "inventory": im.module_inventory(self.model)}

    def classify(self) -> dict:
        self.notes["class"] = im.classify(self.model, self.config)
        return self.notes["class"]

    def analyze_forward(self, limit: int = 30) -> str:
        return trace.trace_forward(self.model, self.sample_batch).table(limit)

    def identify_inputs(self) -> dict:
        return im.forward_signature(self.model)["parameters"]

    def identify_outputs(self) -> dict:
        with torch.no_grad():
            out = self.model(**self.sample_batch) if isinstance(self.sample_batch, dict) else self.model(*self.sample_batch)
        if hasattr(out, "keys"):
            return {k: tuple(v.shape) if torch.is_tensor(v) else type(v).__name__ for k, v in out.items()}
        if torch.is_tensor(out):
            return {"output": tuple(out.shape)}
        return {"output_type": type(out).__name__}

    def identify_loss(self) -> dict:
        outs = self.identify_outputs()
        return {"model_returns_loss": "loss" in outs, "loss_functions_in_source": im.losses_in_source(self.model),
                "note": "if the model returns no loss, you must write loss_fn(model, batch) yourself (STEP 3)"}

    def discover_trainable_modules(self) -> dict:
        self.notes["targets"] = targets.discover(self.model, self.sample_batch, self.config)
        return {k: v for k, v in self.notes["targets"].items() if k != "trace"}

    def recommend_peft_strategy(self, vram_gb: float, task: str, n_tasks: int = 1, data_rows: int = 1000,
                                new_output_space: bool = False) -> dict:
        fam = self.notes.get("class", self.classify())["labels"]
        t = self.notes.get("targets") or targets.discover(self.model, self.sample_batch, self.config)
        return strategy.recommend(im.param_counts(self.model)["total"], vram_gb, fam, task, n_tasks, data_rows,
                                  new_output_space, lora_targets=len(t["lora_targets"]))

    def build_training_pipeline(self, r: int = 8, alpha: float = 16, train_also: list[str] | None = None) -> dict:
        t = self.notes.get("targets") or targets.discover(self.model, self.sample_batch, self.config)
        return L.inject(self.model, t["lora_targets"], r=r, alpha=alpha, train_also=train_also)

    def validate_training(self, loss_fn, batch=None) -> dict:
        """One backward pass on a TRAINING batch (with labels), then the silent-failure report."""
        self.model.zero_grad(set_to_none=True)
        loss = loss_fn(self.model, batch if batch is not None else self.sample_batch)
        if loss is None:
            raise RuntimeError("loss_fn returned None: the model computed no loss (labels missing?)")
        loss.backward()
        rep = trace.gradient_report(self.model)
        self.model.zero_grad(set_to_none=True)
        return rep

    def train(self, loader, loss_fn, **kw) -> dict:
        return loop.train(self.model, loader, loss_fn, **kw)

    def evaluate(self, eval_fn) -> dict:
        self.model.eval()
        with torch.no_grad():
            return eval_fn(self.model)

    def export(self, path: str, merge: bool = True, also: list[str] | None = None) -> dict:
        from pathlib import Path

        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(L.adapter_state_dict(self.model, also), Path(path) / "adapter.pt")
        merged = L.merge_all(self.model) if merge else 0
        if merge:
            torch.save(self.model.state_dict(), Path(path) / "merged.pt")
        return {"adapter": str(Path(path) / "adapter.pt"), "merged_layers": merged}

    def serve(self) -> str:
        fam = self.notes.get("class", self.classify())["labels"]
        prefix = "CUSTOM ENGINEERING REQUIRED (no inference engine knows this class): " if "custom" in fam else ""
        return prefix + self._serve_for(fam)

    @staticmethod
    def _serve_for(fam) -> str:
        if "decoder-only" in fam and "text" in fam:
            return "vLLM (chapter 52) after merging, or multi-LoRA adapters; OpenAI-compatible API"
        if "encoder-decoder" in fam or "encoder-only" in fam:
            return "transformers/ONNX Runtime behind FastAPI; batch requests; vLLM support for this family is not assumed"
        if "vision" in fam or "audio" in fam or "diffusion" in fam:
            return "framework runtime (torch/ONNX/TensorRT) behind an API with a modality-specific pre/post-processor"
        return "torch inference behind FastAPI (serve_local pattern), then optimise (compile/ONNX/TensorRT)"
