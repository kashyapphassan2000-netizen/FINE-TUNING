"""UNIVERSAL PROCESSOR LAYER: raw record -> model inputs, for ANY modality.

The processor is whatever turns a human record (text, image path, audio array, anything) into the
tensors forward() accepts. This layer finds it, says what it produces, and - when nothing exists
(custom models, new modalities) - defines the contract YOU implement, so the rest of the engine is
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

INPUT_KEYS = {
    "input_ids": "text token ids", "attention_mask": "1 = real token", "pixel_values": "image tensor",
    "input_values": "raw audio samples", "input_features": "audio spectrogram", "decoder_input_ids": "seq2seq decoder",
    "token_type_ids": "segment ids", "labels": "targets",
}


@dataclass
class Processor:
    kind: str                   # processor | tokenizer | image | feature_extractor | custom | none
    obj: object = None
    encode: object = None       # callable(record) -> dict of tensors (what the engine actually calls)
    produces: tuple = ()
    note: str = ""

    def __call__(self, record):
        if self.encode is None:
            raise NotImplementedError(
                "no processor for this model/modality. Provide one:\n"
                "    Processor(kind='custom', encode=lambda rec: {'x': torch.tensor(rec['x'])},\n"
                "              produces=('x',))\n"
                "That is the ONLY piece the engine cannot infer for a new modality.")
        return self.encode(record)


def resolve(model_id: str | None = None, model=None, trust_remote_code: bool = False) -> Processor:
    """Find the model's own processor. Order matters: a multimodal AutoProcessor wraps the others."""
    if model_id:
        from transformers import (AutoFeatureExtractor, AutoImageProcessor, AutoProcessor, AutoTokenizer)

        for kind, cls in (("processor", AutoProcessor), ("tokenizer", AutoTokenizer),
                          ("image", AutoImageProcessor), ("feature_extractor", AutoFeatureExtractor)):
            try:
                obj = cls.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            except Exception:
                continue
            return Processor(kind=kind, obj=obj, encode=lambda rec, o=obj: _hf_encode(o, rec),
                             produces=_produces(kind), note=f"{type(obj).__name__} from {model_id}")
    sig = set()
    if model is not None:
        import inspect as _i

        sig = set(_i.signature(model.forward).parameters)
    return Processor(kind="none" if sig else "custom", produces=tuple(k for k in INPUT_KEYS if k in sig),
                     note="no Hugging Face processor: implement Processor(kind='custom', encode=...) "
                          f"returning {sorted(k for k in sig if k != 'self') or ['your input names']}")


def _produces(kind: str) -> tuple:
    return {"tokenizer": ("input_ids", "attention_mask"), "image": ("pixel_values",),
            "feature_extractor": ("input_values|input_features",),
            "processor": ("input_ids", "attention_mask", "pixel_values|input_features")}.get(kind, ())


def _hf_encode(obj, record: dict) -> dict:
    """One record -> tensors, using whichever fields the record has. Deliberately small and readable."""
    kw = {"return_tensors": "pt"}
    if "messages" in record and hasattr(obj, "apply_chat_template"):
        out = obj.apply_chat_template(record["messages"], tokenize=True, return_dict=True, **kw)
    elif "image" in record and "text" in record and hasattr(obj, "__call__") and hasattr(obj, "image_processor"):
        out = obj(images=record["image"], text=record["text"], **kw)
    elif "image" in record:
        out = obj(images=record["image"], **kw)
    elif "audio" in record:
        rate = record.get("sampling_rate", getattr(obj, "sampling_rate", 16000))
        out = obj(record["audio"], sampling_rate=rate, **kw)
    elif "text" in record:
        out = obj(record["text"], truncation=True, **kw)
    else:
        raise KeyError(f"record has none of messages/text/image/audio: {list(record)}")
    return {k: (v[0] if torch.is_tensor(v) and v.dim() > 1 and v.shape[0] == 1 else v) for k, v in dict(out).items()}


def describe(p: Processor) -> str:
    return f"processor: {p.kind} | produces {p.produces or '(you define)'} | {p.note}"
