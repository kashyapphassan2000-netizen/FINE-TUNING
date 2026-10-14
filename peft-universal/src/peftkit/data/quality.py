"""Stage: dedup + filtering for SFT/preference jsonl ({"messages": [...]}) rows.

exact dedup (normalized hash) -> near-dup (MinHash + LSH, no extra deps) -> heuristic quality filters
-> PII scrub/drop -> refusal/boilerplate filter -> benchmark/eval contamination (n-gram overlap)
-> poisoning scan (trigger phrases, injected URLs/instructions). Every drop is counted with a reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REFUSALS = ("as an ai language model", "i'm sorry, but i can't", "i cannot assist with", "i am unable to provide",
            "as a large language model")
PII = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\s-]?){10,13}(?!\d)"),
    "card": re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}
INJECTION = re.compile(r"(ignore (all|previous) instructions|<\s*script|https?://\S+\.(zip|exe|sh)\b|curl\s+\S+\s*\|\s*(ba)?sh)", re.I)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def row_text(row: dict) -> str:
    return "\n".join(m["content"] for m in row["messages"] if m["role"] != "system")


def shingles(text: str, n: int = 5) -> set[str]:
    words = normalize(text).split()
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def _h(s: str, seed: int) -> int:
    return struct.unpack("<Q", hashlib.blake2b(s.encode(), digest_size=8, salt=seed.to_bytes(8, "little")).digest())[0]


def minhash(sh: set[str], num_perm: int = 64) -> list[int]:
    return [min(_h(s, i) for s in sh) for i in range(num_perm)]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


class NearDupIndex:
    """LSH over MinHash signatures; candidates are confirmed with exact Jaccard."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 64, bands: int = 16):
        self.t, self.num_perm, self.bands = threshold, num_perm, bands
        self.per = num_perm // bands
        self.buckets: dict[tuple, list[int]] = defaultdict(list)
        self.sets: list[set] = []

    def add_if_new(self, text: str) -> bool:
        sh = shingles(text)
        sig = minhash(sh, self.num_perm)
        keys = [(b, tuple(sig[b * self.per:(b + 1) * self.per])) for b in range(self.bands)]
        cands = {i for k in keys for i in self.buckets.get(k, [])}
        if any(jaccard(sh, self.sets[i]) >= self.t for i in cands):
            return False
        idx = len(self.sets)
        self.sets.append(sh)
        for k in keys:
            self.buckets[k].append(idx)
        return True


def repetition_ratio(text: str) -> float:
    words = normalize(text).split()
    if len(words) < 10:
        return 0.0
    grams = Counter(" ".join(words[i:i + 3]) for i in range(len(words) - 2))
    return 1 - len(grams) / max(1, sum(grams.values()))


def ngram_set(text: str, n: int = 13) -> set[str]:
    w = normalize(text).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def quality_reasons(row: dict, min_answer_chars: int, max_chars: int, max_rep: float, pii_mode: str) -> list[str]:
    reasons = []
    answer = next((m["content"] for m in reversed(row["messages"]) if m["role"] == "assistant"), "")
    text = row_text(row)
    if len(answer.strip()) < min_answer_chars:
        reasons.append("answer_too_short")
    if len(text) > max_chars:
        reasons.append("too_long")
    if repetition_ratio(answer) > max_rep:
        reasons.append("repetitive")
    if sum(ch.isprintable() or ch in "\n\t" for ch in text) / max(1, len(text)) < 0.98:
        reasons.append("non_printable")
    if any(r in answer.lower() for r in REFUSALS):
        reasons.append("refusal_boilerplate")
    if INJECTION.search(text):
        reasons.append("suspected_injection")
    if pii_mode == "drop" and any(p.search(text) for p in PII.values()):
        reasons.append("pii")
    return reasons


def scrub_pii(row: dict) -> dict:
    for m in row["messages"]:
        for name, pat in PII.items():
            m["content"] = pat.sub(f"<{name.upper()}>", m["content"])
    return row


def run(inputs: list[Path], out: Path, report: Path | None = None, near_dup: float = 0.8,
        min_answer_chars: int = 3, max_chars: int = 32000, max_rep: float = 0.5, pii: str = "scrub",
        contamination: list[Path] | None = None, blocklist: list[str] | None = None) -> dict:
    bench: set[str] = set()
    for p in contamination or []:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                bench |= ngram_set(row_text(r) if "messages" in r else json.dumps(r))
    block = [b.lower() for b in (blocklist or [])]
    seen, idx = set(), NearDupIndex(near_dup)
    stats: Counter = Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fo:
        for p in inputs:
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                stats["read"] += 1
                text = row_text(row)
                key = hashlib.sha1(normalize(text).encode()).hexdigest()
                if key in seen:
                    stats["drop_exact_dup"] += 1
                    continue
                seen.add(key)
                reasons = quality_reasons(row, min_answer_chars, max_chars, max_rep, pii)
                if block and any(b in text.lower() for b in block):
                    reasons.append("blocklist_trigger")
                if bench and ngram_set(text) & bench:
                    reasons.append("benchmark_contamination")
                if reasons:
                    for r in reasons:
                        stats[f"drop_{r}"] += 1
                    continue
                if near_dup < 1.0 and not idx.add_if_new(text):
                    stats["drop_near_dup"] += 1
                    continue
                if pii == "scrub":
                    row = scrub_pii(row)
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["kept"] += 1
    result = dict(stats)
    if report:
        report.write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Dedup + quality filter chat jsonl")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--near-dup", type=float, default=0.8, help="Jaccard threshold; 1.0 disables")
    ap.add_argument("--min-answer-chars", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=32000)
    ap.add_argument("--max-repetition", type=float, default=0.5)
    ap.add_argument("--pii", choices=["scrub", "drop", "keep"], default="scrub")
    ap.add_argument("--contamination", nargs="*", default=[], help="eval/benchmark jsonl files to guard against")
    ap.add_argument("--blocklist", nargs="*", default=[], help="suspected poisoning trigger phrases")
    a = ap.parse_args(argv)
    print(json.dumps(run([Path(p) for p in a.inputs], Path(a.out), Path(a.report) if a.report else None,
                         a.near_dup, a.min_answer_chars, a.max_chars, a.max_repetition, a.pii,
                         [Path(p) for p in a.contamination], a.blocklist)))


if __name__ == "__main__":
    main()
