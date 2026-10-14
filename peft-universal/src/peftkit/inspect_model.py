"""Print linear layer names of ANY causal LM without downloading weights (meta device).
Use the output to choose `method.target_modules` for new architectures."""
from __future__ import annotations

import argparse

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from .modeling import list_linear_modules


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args(argv)
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=a.trust_remote_code)
    total = sum(p.numel() for p in model.parameters())
    print(f"architecture: {cfg.architectures}  params: {total / 1e9:.3f}B")
    for leaf, n in sorted(list_linear_modules(model).items()):
        print(f"  {leaf:<24} x{n}")
    print('\nall-linear targets every name above. Attention-only: ["q_proj","k_proj","v_proj","o_proj"] (names vary).')


if __name__ == "__main__":
    main()
