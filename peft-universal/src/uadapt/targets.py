"""STEP 5: discover adaptation targets from COMPUTATION, not from names.

Rules (each is structural, so it works on BERT 'query', T5 'q', Mamba 'in_proj' or your 'foo_7'):
  parallel group      : >=2 linears that receive the SAME input tensor
     >=3 in the group -> attention projections (q/k/v)           -> good LoRA targets
     2, both expand   -> gated MLP input (gate/up)                -> good LoRA targets
  fused projection    : one linear with out == 3*in (or larger)   -> fused qkv -> good target
  expand / contract   : out >= 1.5*in  / in >= 1.5*out            -> MLP up / down -> good targets
  head                : last linear executed, or out == vocab/num_labels -> TRAIN FULLY or freeze, not LoRA
  router              : out == num_experts                          -> never adapt (routing stability)
  embeddings / norms  : Embedding, *Norm                             -> do not LoRA
  small projections   : out or in < 1/8 of model width (e.g. the SSM dt projection)  -> skip by default
  convs               : Conv stems are adaptable with conv-LoRA (uadapt.lora), deeper convs as capacity needs
"""
from __future__ import annotations

from collections import defaultdict

from torch import nn

from .trace import trace_forward


def discover(model: nn.Module, batch, config=None) -> dict:
    cfg = config if config is not None else getattr(model, "config", None)
    d = cfg.to_dict() if hasattr(cfg, "to_dict") else (cfg or {})
    head_dims = {v for k, v in d.items() if k in ("vocab_size", "num_labels") and isinstance(v, int)}
    if d.get("id2label"):
        head_dims.add(len(d["id2label"]))
    n_experts = next((d[k] for k in ("num_experts", "num_local_experts", "n_routed_experts") if d.get(k)), None)
    width = next((d[k] for k in ("hidden_size", "d_model", "dim", "n_embd") if isinstance(d.get(k), int)), None)

    tr = trace_forward(model, batch)
    mods = dict(model.named_modules())
    linears = [c for c in tr.calls if c.kind in ("Linear", "Conv1D")]
    if width is None and linears:
        width = max(set(c.in_shape[-1] for c in linears if c.in_shape), key=lambda w: sum(
            1 for c in linears if c.in_shape and c.in_shape[-1] == w))
    groups = defaultdict(list)
    for c in linears:
        groups[c.input_id].append(c)

    role, reason = {}, {}

    def set_role(name, r, why):
        if name not in role:
            role[name], reason[name] = r, why

    last_linear = linears[-1].name if linears else None
    for c in linears:
        m = mods[c.name]
        i, o = getattr(m, "in_features", None), getattr(m, "out_features", None)
        if n_experts and o == n_experts:
            set_role(c.name, "router", f"out_features == num_experts ({n_experts})")
        elif c.name == last_linear or (o in head_dims and o not in (width,)):
            set_role(c.name, "head", "last executed linear" if c.name == last_linear else f"out_features == {o} (vocab/labels)")
    for gid, members in groups.items():
        if len(members) >= 3:
            for c in members:
                set_role(c.name, "attention_proj", f"one of {len(members)} linears sharing one input tensor")
        elif len(members) == 2:
            outs = [getattr(mods[c.name], "out_features", 0) for c in members]
            ins = [getattr(mods[c.name], "in_features", 1) for c in members]
            if all(o >= 1.5 * i for o, i in zip(outs, ins)):
                for c in members:
                    set_role(c.name, "mlp_gate_up", "2 expanding linears sharing one input (gated MLP)")
            else:
                for c in members:
                    set_role(c.name, "parallel_proj", "2 linears sharing one input (e.g. k/v)")
    # output projection: the first linear after an attention group whose input width matches the group output
    order = [c.name for c in linears]
    for idx, c in enumerate(linears):
        if role.get(c.name) == "attention_proj":
            for nxt in linears[idx + 1:]:
                if role.get(nxt.name) not in ("attention_proj",):
                    if nxt.name not in role:
                        set_role(nxt.name, "attention_out", "first linear after an attention group")
                    break
    for c in linears:
        if c.name in role:
            continue
        m = mods[c.name]
        i, o = getattr(m, "in_features", None), getattr(m, "out_features", None)
        if i is None or o is None:
            continue
        if width and (min(i, o) < max(1, width // 8)):
            set_role(c.name, "small_proj", f"narrow ({i}->{o}) relative to width {width}: sensitive, skip")
        elif o >= 2.5 * i:
            set_role(c.name, "fused_or_expand", f"{i}->{o}: fused qkv or MLP expansion")
        elif o >= 1.5 * i:
            set_role(c.name, "mlp_up", f"expands {i}->{o}")
        elif i >= 1.5 * o:
            set_role(c.name, "mlp_down", f"contracts {i}->{o}")
        else:
            set_role(c.name, "square_proj", f"{i}->{o} projection")
    for c in tr.calls:
        if c.kind in ("Conv1d", "Conv2d"):
            set_role(c.name, "conv", "convolution (conv-LoRA possible)")
        elif c.kind == "Embedding":
            set_role(c.name, "embedding", "token/position embedding")
        elif c.kind in ("LSTM", "GRU", "RNN"):
            set_role(c.name, "recurrent", "recurrent cell: adapt input/output linears around it, or full-tune")
    for n, m in model.named_modules():
        if "Norm" in type(m).__name__ and not list(m.children()):
            set_role(n, "norm", "normalization: tiny, train fully if anything")

    good = {"attention_proj", "attention_out", "mlp_gate_up", "mlp_up", "mlp_down", "fused_or_expand",
            "parallel_proj", "square_proj"}
    lora_targets = sorted(n for n, r in role.items() if r in good and isinstance(mods.get(n), nn.Linear))
    return {"roles": role, "reasons": reason, "lora_targets": lora_targets,
            "do_not_adapt": sorted(n for n, r in role.items() if r in ("router", "embedding", "norm", "small_proj")),
            "train_fully_or_freeze": sorted(n for n, r in role.items() if r == "head"),
            "conv_candidates": sorted(n for n, r in role.items() if r == "conv"),
            "leaf_names": sorted({n.split(".")[-1] for n in lora_targets}),
            "width": width, "n_linear": len(linears), "trace": tr}
