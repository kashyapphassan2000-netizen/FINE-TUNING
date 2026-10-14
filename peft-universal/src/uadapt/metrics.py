"""Task-specific metrics. One registry, so evaluation is chosen by TASK, not by habit."""
from __future__ import annotations

import math
from collections import Counter


def accuracy(pred, gold) -> float:
    return sum(p == g for p, g in zip(pred, gold)) / max(1, len(gold))


def prf(pred, gold, positive=None) -> dict:
    """Macro precision/recall/F1 (per class, then averaged) plus per-class recall."""
    classes = sorted(set(gold) | set(pred)) if positive is None else [positive]
    per = {}
    for c in classes:
        tp = sum(p == c and g == c for p, g in zip(pred, gold))
        fp = sum(p == c and g != c for p, g in zip(pred, gold))
        fn = sum(p != c and g == c for p, g in zip(pred, gold))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                  "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0}
    macro = {k: round(sum(v[k] for v in per.values()) / len(per), 4) for k in ("precision", "recall", "f1")}
    return {"per_class": per, "macro": macro}


def auroc(scores, labels) -> float:
    """Rank-based AUROC for binary labels (1 = positive)."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def ece(probs, labels, bins: int = 10) -> float:
    """Expected calibration error: are the model's confidences honest?"""
    total, n = 0.0, len(labels)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo < p <= hi or (b == 0 and p == 0))]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(acc - conf)
    return round(total, 4)


def recall_at_k(ranked_ids: list[list], gold_ids: list, k: int = 5) -> float:
    return round(sum(g in r[:k] for r, g in zip(ranked_ids, gold_ids)) / max(1, len(gold_ids)), 4)


def mrr(ranked_ids: list[list], gold_ids: list) -> float:
    s = 0.0
    for r, g in zip(ranked_ids, gold_ids):
        s += 1.0 / (r.index(g) + 1) if g in r else 0.0
    return round(s / max(1, len(gold_ids)), 4)


def ndcg_at_k(ranked_ids: list[list], relevance: list[dict], k: int = 10) -> float:
    """relevance[i] = {doc_id: gain}. Graded relevance, unlike recall/MRR."""
    total = 0.0
    for r, rel in zip(ranked_ids, relevance):
        dcg = sum(rel.get(d, 0.0) / math.log2(i + 2) for i, d in enumerate(r[:k]))
        ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(rel.values(), reverse=True)[:k]))
        total += dcg / ideal if ideal else 0.0
    return round(total / max(1, len(ranked_ids)), 4)


def _edit(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wer(hyps: list[str], refs: list[str]) -> float:
    e = sum(_edit(h.split(), r.split()) for h, r in zip(hyps, refs))
    n = sum(len(r.split()) for r in refs)
    return round(e / max(1, n), 4)


def cer(hyps: list[str], refs: list[str]) -> float:
    e = sum(_edit(list(h), list(r)) for h, r in zip(hyps, refs))
    n = sum(len(r) for r in refs)
    return round(e / max(1, n), 4)


def format_compliance(preds: list[str], validator) -> float:
    return round(sum(bool(validator(p)) for p in preds) / max(1, len(preds)), 4)


def distinct_n(texts: list[str], n: int = 3) -> float:
    grams = [tuple(t.split()[i:i + n]) for t in texts for i in range(max(0, len(t.split()) - n + 1))]
    return round(len(set(grams)) / max(1, len(grams)), 4)


# which metrics a task must report: used by tasks.TASKS and by the evaluation chapters
TASK_METRICS = {
    "causal_lm": ["task score on your suite", "format compliance", "general suite", "safety suite", "judge vs baseline"],
    "masked_lm": ["masked-token accuracy", "downstream task score after fine-tuning"],
    "seq2seq": ["exact match / F1", "reference metric (BLEU/ROUGE-style)", "human review"],
    "sequence_classification": ["macro-F1", "per-class recall", "AUROC (binary)", "ECE (calibration)"],
    "token_classification": ["entity-level precision/recall/F1", "per-class recall"],
    "regression": ["MAE", "RMSE", "R^2 vs a naive baseline"],
    "embedding": ["Recall@K", "MRR", "nDCG@10"],
    "reranking": ["nDCG@10", "MRR", "latency per query"],
    "reward_model": ["pairwise accuracy on held-out preferences", "score calibration"],
    "preference": ["reward accuracy/margin", "the SFT suites re-run", "length change"],
    "multimodal_sft": ["task score", "field-level accuracy", "hallucination rate", "human review"],
    "asr_ctc": ["WER", "CER", "per-condition WER (noise, accent, device)"],
    "asr_seq2seq": ["WER", "CER", "hallucination rate on silence"],
    "audio_classification": ["accuracy", "per-class recall", "EER (speaker)"],
    "image_classification": ["accuracy", "per-class recall", "per-segment accuracy"],
    "image_generation": ["human preference", "prompt adherence (e.g. CLIP-style similarity)", "subject consistency",
                         "artifact rate", "distribution metric (FID-style)"],
    "tts": ["human MOS-style listening test", "speaker similarity", "WER of an ASR model on the output"],
    "video_classification": ["accuracy", "per-class recall", "temporal consistency"],
    "custom": ["define it before you train: what number decides success?"],
}
