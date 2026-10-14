"""Trainer callbacks. HF Trainer does NOT call AdaLoRA's rank allocator; this does."""
from __future__ import annotations

from transformers import TrainerCallback


def _unwrap(model):
    while hasattr(model, "module"):  # DDP / accelerate wrappers
        model = model.module
    return model


class AdaLoraCallback(TrainerCallback):
    """Runs after optimizer.step() and before zero_grad(), which is what AdaLoRA requires
    (it scores importance from the gradients that are still on the parameters)."""

    def __init__(self) -> None:
        self.rank_history: list[tuple[int, int]] = []

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        peft_model = _unwrap(model)
        peft_model.base_model.update_and_allocate(state.global_step)
        if state.global_step % max(1, args.logging_steps) == 0:
            self.rank_history.append((state.global_step, count_active_rank(peft_model)))
        return control


def count_active_rank(peft_model) -> int:
    """Sum of non-zero singular values across all AdaLoRA layers (the live rank budget)."""
    total = 0
    for name, p in peft_model.named_parameters():
        if "lora_E" in name:
            total += int((p.detach() != 0).sum().item())
    return total
