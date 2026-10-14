"""UNIVERSAL COLLATOR: a list of encoded records -> one batch, for any modality and any task.

Rules, applied per key, with no modality-specific code:
  variable-length tensors  -> pad (labels with -100, ids with pad_id, floats with 0.0) + build attention_mask
  equal-shape tensors      -> stack
  scalars / python values  -> tensor if numeric, list otherwise
  nested dicts (pairs)     -> collate recursively (embedding: anchor/positive; reward: chosen/rejected)
"""
from __future__ import annotations

import torch

LABEL_PAD = -100


def collate(rows: list[dict], pad_id: int = 0, label_pad: int = LABEL_PAD, build_mask: bool = True) -> dict:
    if not rows:
        return {}
    out = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        if isinstance(vals[0], dict):                                     # paired inputs
            out[key] = collate(vals, pad_id, label_pad, build_mask)
            continue
        if torch.is_tensor(vals[0]):
            vals = [v.squeeze(0) if v.dim() > 1 and v.shape[0] == 1 and len({x.shape for x in vals}) > 1 else v
                    for v in vals]
            shapes = {tuple(v.shape) for v in vals}
            if len(shapes) == 1:
                out[key] = torch.stack(vals)
            else:
                pad = label_pad if "label" in key else (0.0 if vals[0].is_floating_point() else pad_id)
                out[key] = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=pad)
                if build_mask and key in ("input_ids", "input_values", "input_features"):
                    out.setdefault("attention_mask", torch.nn.utils.rnn.pad_sequence(
                        [torch.ones(v.shape[0], dtype=torch.long) for v in vals], batch_first=True))
        elif isinstance(vals[0], (int, float, bool)):
            out[key] = torch.tensor(vals)
        else:
            out[key] = vals
    return out


def make_loader(records: list[dict], processor, label_fn=None, batch_size: int = 8, shuffle: bool = True,
                pad_id: int = 0):
    """records (human form) -> encoded -> batches. label_fn(record, encoded) -> dict added to the encoding."""
    from torch.utils.data import DataLoader, Dataset

    class Encoded(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, i):
            rec = records[i]
            enc = dict(processor(rec)) if processor is not None else {k: v for k, v in rec.items()}
            if label_fn is not None:
                enc.update(label_fn(rec, enc))
            return enc

    return DataLoader(Encoded(), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda rows: collate(rows, pad_id=pad_id))


def label_builders(task: str, pad_id: int = 0):
    """The label rule per task, as code. Returns label_fn(record, encoded) -> dict."""
    def causal(rec, enc):
        return {"labels": enc["input_ids"].clone()}            # the model shifts internally; mask the prompt upstream

    def cls(rec, enc):
        return {"labels": int(rec["label"])}

    def regression(rec, enc):
        return {"labels": float(rec["target"])}

    def tokens(rec, enc):
        return {"labels": torch.tensor(rec["tags"])}

    def seq2seq(rec, enc):
        return {"labels": enc.pop("target_ids")} if "target_ids" in enc else {}

    return {"causal_lm": causal, "sequence_classification": cls, "regression": regression,
            "token_classification": tokens, "seq2seq": seq2seq}.get(task)
