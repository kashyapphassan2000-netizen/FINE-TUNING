"""STEP 4: decide WHAT to train, from constraints - not from habit."""
from __future__ import annotations


def recommend(total_params: int, vram_gb: float, family: list[str], task: str, n_tasks: int = 1,
              data_rows: int = 1000, new_output_space: bool = False, quantizable: bool = True,
              lora_targets: int = 0) -> dict:
    """Returns ranked options with the reason for each. Memory model: bf16 weights 2 B/param,
    full fine-tune with Adam ~16 B/param, frozen base + small adapter ~2 B/param (+ activations)."""
    full_gb = total_params * 16 / 1e9
    frozen_gb = total_params * 2 / 1e9
    q4_gb = total_params * 0.55 / 1e9
    opts = []

    def add(name, why, score):
        opts.append({"strategy": name, "why": why, "score": score})

    if new_output_space:
        add("head-only or head + LoRA", "the output space is new (labels/classes): the head must be trained from scratch", 9)
    if full_gb < 0.6 * vram_gb and data_rows >= 50_000 and n_tasks == 1:
        add("full fine-tuning", f"fits (~{full_gb:.1f} GB with Adam) and there is enough data to use the capacity", 8)
    if lora_targets == 0:
        add("selective fine-tuning (unfreeze chosen blocks)", "no Linear targets found: adapt by unfreezing layers, or write custom adapters", 7)
    if frozen_gb < 0.7 * vram_gb and lora_targets:
        add("LoRA", f"frozen bf16 base ~{frozen_gb:.1f} GB fits; one small adapter per task", 8 + (n_tasks > 1))
    elif q4_gb < 0.7 * vram_gb and quantizable and lora_targets:
        add("QLoRA", f"bf16 base ~{frozen_gb:.1f} GB does not fit, 4-bit ~{q4_gb:.1f} GB does (Linear layers only)", 8)
    if "moe" in family:
        add("LoRA on attention only", "MoE: adapting every expert multiplies trainable params; routers stay frozen", 8)
    if "vision" in family or "audio" in family:
        add("projector / head + LoRA on the language side",
            "modality encoders are usually pretrained well; adapt the bridge and the decoder first", 7)
    if data_rows < 500:
        add("prompt/prefix tuning or IA3", "very little data: the smallest parameter budgets overfit least", 6)
    if n_tasks > 3:
        add("multiple LoRA adapters on one base", "many tasks share one frozen base (hot-swap at serving)", 7)
    if any(k in family for k in ("state-space", "recurrent", "custom", "diffusion")) and lora_targets:
        add("custom LoRA injection (uadapt.lora)", "framework support may be missing; Linear/Conv layers can still be wrapped", 7)
    opts.sort(key=lambda o: -o["score"])
    return {"memory_gb": {"full_ft_adam": round(full_gb, 2), "frozen_bf16": round(frozen_gb, 2), "nf4": round(q4_gb, 2)},
            "ranked": opts, "note": "scores rank the options for THESE constraints; always confirm with a 20-step run"}
