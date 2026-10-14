"""THE TASK REGISTRY: the missing abstraction.

Fine-tuning is FOUR independent choices, not one:

    MODALITY  x  ARCHITECTURE  x  TASK (training objective)  x  METHOD (full / LoRA / QLoRA / IA3 / ...)

A PEFT method does not tell you the data format, the collator, the loss or the metric. The TASK does.
This registry holds, for every task: the data schema, the collator rules, the loss, the metrics, the
adaptation guidance and the serving note. `resolve()` picks the candidate tasks for a model from its
architecture and modality, and `plan()` returns the full recipe.

Add your own task with `register(TrainingTask(...))` - the engine then treats it like any other.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .metrics import TASK_METRICS


@dataclass
class TrainingTask:
    name: str
    modality: str                 # text | image | audio | video | image+text | audio+text | any
    architectures: list           # decoder-only, encoder-only, encoder-decoder, vision, audio, diffusion, custom
    data_schema: dict             # what ONE training record must contain
    labels: str                   # how labels are built
    loss: str                     # the objective, in words
    loss_fn: object = None        # callable(model, batch) -> scalar, when it is generic enough to ship
    collator: str = ""            # what the collator must do
    peft: str = ""                # which modules to adapt first
    metrics: list = field(default_factory=list)
    serving: str = ""
    notes: str = ""

    def card(self) -> str:
        return (f"TASK {self.name}  [{self.modality} | {', '.join(self.architectures)}]\n"
                f"  record : {self.data_schema}\n  labels : {self.labels}\n  loss   : {self.loss}\n"
                f"  collate: {self.collator}\n  adapt  : {self.peft}\n  metrics: {', '.join(self.metrics)}\n"
                f"  serve  : {self.serving}" + (f"\n  note   : {self.notes}" if self.notes else ""))


# ---------------------------------------------------------------- generic loss functions
def hf_loss(model, batch):
    """The model computes its own loss when given labels (most Hugging Face heads)."""
    out = model(**batch)
    loss = getattr(out, "loss", None) if not isinstance(out, dict) else out.get("loss")
    if loss is None:      # base models (no head) and custom modules have no loss: that is a task decision
        raise RuntimeError(f"{type(model).__name__} returned no loss for this batch: pass labels, use a head "
                           "(e.g. *ForSequenceClassification), or supply a task-specific loss_fn")
    return loss


def regression_loss(model, batch):
    pred = model(**{k: v for k, v in batch.items() if k != "labels"})
    pred = pred.logits if hasattr(pred, "logits") else pred
    return nn.functional.mse_loss(pred.squeeze(-1).float(), batch["labels"].float())


def contrastive_loss(model, batch, temperature: float = 0.05):
    """Multiple-negatives ranking (in-batch negatives): the classic embedding objective.
    batch: {"anchor": {...}, "positive": {...}} of encoder inputs; optional "negative"."""
    def embed(inputs):
        out = model(**inputs)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        mask = inputs.get("attention_mask")
        if mask is not None and h.dim() == 3:
            h = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        elif h.dim() == 3:
            h = h.mean(1)
        return nn.functional.normalize(h, dim=-1)

    a, p = embed(batch["anchor"]), embed(batch["positive"])
    cand = [p] + ([embed(batch["negative"])] if "negative" in batch else [])
    sim = a @ torch.cat(cand, 0).T / temperature
    return nn.functional.cross_entropy(sim, torch.arange(a.shape[0], device=a.device))


def reward_model_loss(model, batch):
    """Bradley-Terry pairwise loss: score(chosen) must exceed score(rejected). NOT the same as DPO."""
    def score(inputs):
        out = model(**inputs)
        logits = out.logits if hasattr(out, "logits") else out
        return logits[..., 0] if logits.dim() > 1 else logits

    return -nn.functional.logsigmoid(score(batch["chosen"]) - score(batch["rejected"])).mean()


def diffusion_loss(model, batch, alpha_bar=None):
    """epsilon-prediction: the model predicts the noise added to a clean sample at timestep t."""
    x0 = batch["x0"]
    t = batch.get("t", torch.randint(0, 1000, (x0.shape[0],), device=x0.device))
    ab = (alpha_bar if alpha_bar is not None else torch.cos(t.float() / 1000 * torch.pi / 2) ** 2)
    ab = ab.view(-1, *[1] * (x0.dim() - 1))
    eps = batch.get("noise", torch.randn_like(x0))
    pred = model(ab.sqrt() * x0 + (1 - ab).sqrt() * eps, t, **batch.get("cond", {}))
    pred = pred.sample if hasattr(pred, "sample") else pred
    return nn.functional.mse_loss(pred.float(), eps.float())


TASKS: dict[str, TrainingTask] = {}


def register(task: TrainingTask) -> TrainingTask:
    TASKS[task.name] = task
    return task


def _m(name):
    return TASK_METRICS.get(name, [])


register(TrainingTask("causal_lm", "text", ["decoder-only"],
    {"messages": "[{role, content}, ...]"}, "token ids; prompt and system masked with -100",
    "next-token cross-entropy on answer tokens", hf_loss,
    "pad right; attention mask; assistant-only mask from the chat template",
    "attention q/k/v/o + MLP; head usually frozen", _m("causal_lm"),
    "vLLM (merged or multi-LoRA)", "this is the path of chapters 5-65"))
register(TrainingTask("masked_lm", "text", ["encoder-only"],
    {"text": "str"}, "random 15% of tokens replaced by [MASK]; labels = originals, rest -100",
    "cross-entropy on masked positions", hf_loss,
    "dynamic masking per epoch; pad to longest", "attention + MLP; MLM head trained", _m("masked_lm"),
    "not served directly: it is a pretraining step before a task head"))
register(TrainingTask("seq2seq", "text", ["encoder-decoder"],
    {"source": "str", "target": "str"}, "target ids, pad -100; decoder inputs shifted by the model",
    "cross-entropy over target tokens (teacher forcing)", hf_loss,
    "pad source and target separately", "encoder + decoder attention AND cross-attention", _m("seq2seq"),
    "generate() behind an API; check engine support before assuming vLLM"))
register(TrainingTask("sequence_classification", "text", ["encoder-only", "decoder-only"],
    {"text": "str", "label": "int"}, "class id", "cross-entropy (class weights if imbalanced)", hf_loss,
    "pad to longest; keep the label as a tensor", "encoder attention + MLP; classifier head trained fully",
    _m("sequence_classification"), "single forward pass behind FastAPI/ONNX; batch aggressively"))
register(TrainingTask("token_classification", "text", ["encoder-only"],
    {"tokens": "[str]", "tags": "[str]"}, "per-token ids; sub-word continuations -100",
    "cross-entropy per token", hf_loss, "align labels to sub-words after tokenization; pad with -100",
    "encoder + token head", _m("token_classification"), "forward pass; return spans, not raw tags"))
register(TrainingTask("regression", "any", ["encoder-only", "vision", "audio", "custom"],
    {"inputs": "...", "target": "float"}, "float target (normalise with TRAIN statistics only)",
    "MSE or Huber", regression_loss, "stack tensors; keep the normaliser with the model",
    "backbone LoRA + head trained fully", _m("regression"), "forward pass; return the de-normalised value"))
register(TrainingTask("embedding", "text", ["encoder-only", "decoder-only"],
    {"anchor": "str", "positive": "str", "negative": "str (optional)"},
    "no labels: the batch itself provides negatives",
    "multiple-negatives ranking (in-batch negatives), temperature-scaled", contrastive_loss,
    "tokenize each field separately; LARGE batches matter (more negatives)",
    "encoder attention + MLP; pooling is part of the model contract", _m("embedding"),
    "embed once, store in a vector index; serve nearest-neighbour search",
    "hard negatives beat more data; mine them with the current model"))
register(TrainingTask("reranking", "text", ["encoder-only"],
    {"query": "str", "document": "str", "label": "0/1 or graded"}, "relevance label",
    "binary cross-entropy or pairwise ranking", hf_loss, "pair encoding (query [SEP] document)",
    "encoder + head", _m("reranking"), "cross-encoder scoring on the top-K from retrieval"))
register(TrainingTask("reward_model", "text", ["decoder-only", "encoder-only"],
    {"prompt": "str", "chosen": "str", "rejected": "str"}, "no target: only the ORDER matters",
    "Bradley-Terry pairwise: -log sigmoid(s_chosen - s_rejected)", reward_model_loss,
    "tokenize both completions with the prompt; pad; keep pairs together",
    "backbone LoRA + a scalar value head trained fully", _m("reward_model"),
    "used inside RL or best-of-n sampling, not served to users directly",
    "a reward model SCORES answers; DPO trains the policy directly. Different objectives"))
register(TrainingTask("preference", "text", ["decoder-only"],
    {"prompt": "str", "chosen": "str", "rejected": "str"}, "handled by the trainer",
    "DPO / IPO / KTO / ORPO (see chapter 114)", None,
    "TRL's DPOTrainer handles padding and the reference model",
    "same targets as SFT; start from the SFT adapter", _m("preference"),
    "the tuned policy is served normally", "always SFT first"))
register(TrainingTask("multimodal_sft", "image+text", ["multimodal"],
    {"image": "path/PIL", "messages": "[{role, content}]"},
    "answer tokens only; image placeholders and prompt masked",
    "cross-entropy on answer tokens", hf_loss,
    "processor handles both modalities; keep image-placeholder counts consistent; stack pixels",
    "projector first, then LM LoRA, then the vision tower's last blocks", _m("multimodal_sft"),
    "an engine with explicit support for that VLM, else transformers behind an API"))
register(TrainingTask("asr_ctc", "audio", ["audio"],
    {"audio": "path/array", "text": "str"}, "character/token ids; NO alignment needed",
    "CTC (label length must be <= output frames)", hf_loss,
    "feature extractor at the model's sampling rate; pad audio and labels separately",
    "freeze the conv feature extractor; LoRA attention + FFN; CTC head trained", _m("asr_ctc"),
    "chunked/streaming inference behind an API"))
register(TrainingTask("asr_seq2seq", "audio", ["audio", "encoder-decoder"],
    {"audio": "path/array", "text": "str"}, "target token ids, pad -100",
    "cross-entropy (teacher forcing)", hf_loss, "log-mel features; stack; pad labels",
    "encoder + decoder + cross-attention LoRA", _m("asr_seq2seq"),
    "generation per request; watch hallucinations on silence"))
register(TrainingTask("audio_classification", "audio", ["audio"],
    {"audio": "path/array", "label": "int"}, "class id", "cross-entropy", hf_loss,
    "fixed-length crops or padding with an attention mask", "encoder LoRA + head", _m("audio_classification"),
    "forward pass; stream in windows for long audio"))
register(TrainingTask("image_classification", "image", ["vision"],
    {"image": "path/PIL", "label": "int"}, "class id", "cross-entropy (BCE for multi-label)", hf_loss,
    "image processor (size, mean, std); augment TRAIN only", "attention/conv LoRA + head trained fully",
    _m("image_classification"), "ONNX/TensorRT behind an API"))
register(TrainingTask("image_generation", "image", ["diffusion"],
    {"image": "path/PIL", "caption": "str (optional)"}, "the NOISE added at a sampled timestep",
    "epsilon- or v-prediction MSE, optionally SNR-weighted", diffusion_loss,
    "encode to latents with the VAE; resolution bucketing; tokenize captions with the text encoder",
    "LoRA on the denoiser's attention (cross-attention for style/subject); VAE and text encoder frozen",
    _m("image_generation"), "a job queue: latency = steps x step time",
    "DreamBooth, textual inversion and ControlNet are variants of this recipe (chapter 115)"))
register(TrainingTask("tts", "text->audio", ["encoder-decoder", "custom"],
    {"text": "str", "audio": "path/array", "speaker": "id (optional)"},
    "acoustic targets (mel frames or audio codec tokens)",
    "model-specific: L1/L2 on mel, or cross-entropy over codec tokens", None,
    "text normalisation, phonemisation if the model expects it, speaker embedding, mel/codec extraction",
    "acoustic model LoRA; vocoder usually frozen", _m("tts"),
    "streaming synthesis; measure real-time factor",
    "NOT executed in this book: architectures differ too much to ship one loss"))
register(TrainingTask("video_classification", "video", ["vision", "custom"],
    {"video": "path", "label": "int"}, "class id", "cross-entropy", hf_loss,
    "frame sampling (uniform/stride), fixed clip length, per-frame normalisation",
    "temporal/attention layers LoRA + head", _m("video_classification"),
    "decode + batch frames; heavy I/O, plan storage",
    "memory scales with frames x resolution; start with short clips"))
register(TrainingTask("custom", "any", ["custom"],
    {"...": "whatever your model consumes"}, "whatever your loss consumes", "yours", None,
    "yours", "uadapt discovery + custom LoRA", _m("custom"),
    "serve_local pattern, then optimise", "chapter 84 builds the whole loop"))


ARCH_TO_TASKS = {
    "decoder-only": ["causal_lm", "preference", "reward_model", "sequence_classification", "embedding"],
    "encoder-only": ["sequence_classification", "token_classification", "masked_lm", "embedding", "reranking", "regression"],
    "encoder-decoder": ["seq2seq", "asr_seq2seq", "tts"],
    "vision": ["image_classification", "regression", "video_classification"],
    "audio": ["asr_ctc", "asr_seq2seq", "audio_classification"],
    "multimodal": ["multimodal_sft"],
    "diffusion": ["image_generation"],
    "state-space": ["causal_lm"],
    "recurrent": ["regression", "sequence_classification", "custom"],
    "custom": ["custom", "regression", "sequence_classification"],
}


def resolve(labels: list[str]) -> list[TrainingTask]:
    """Which tasks are plausible for a model with these family labels (uadapt.classify)."""
    names, seen = [], set()
    for fam in labels:
        for t in ARCH_TO_TASKS.get(fam, []):
            if t not in seen:
                seen.add(t)
                names.append(t)
    if not names:
        names = ["custom"]
    return [TASKS[n] for n in names]


def plan(labels: list[str], task_name: str | None = None) -> dict:
    """The full recipe: task, data schema, collator, loss, metrics, adaptation, serving."""
    candidates = resolve(labels)
    task = TASKS[task_name] if task_name else candidates[0]
    return {"chosen_task": task.name, "candidates": [t.name for t in candidates], "card": task.card(),
            "loss_fn_available": task.loss_fn is not None, "metrics": task.metrics,
            "data_schema": task.data_schema, "collator": task.collator, "peft": task.peft,
            "serving": task.serving,
            "custom_code_needed": [] if task.loss_fn else ["loss_fn (task-specific: see the card)"]}
