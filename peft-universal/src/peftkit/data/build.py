"""STEP 7 of the flow: ANY pile of documents -> training-ready JSONL, in one command.

    python -m peftkit.data.build --docs data/raw --out data/processed \
        --tokenizer meta-llama/Llama-3.1-8B-Instruct --domain "industrial pump support" \
        --mode grounded --eval data/eval/test.jsonl

Chains: ingest (txt/md/csv/xlsx/html/json/jsonl/pdf/docx/pptx) -> chunk -> synthesise Q&A (grounded in the chunk,
so the answers are about YOUR documents) -> quality (dedup, PII, contamination against your eval set)
-> format (chat messages, tokenizer length filter, split). Resumable, and every stage writes its file.

Scale: 20,000 documents is normal. The cost driver is the LLM calls in the synth stage: cap them with
--max-chunks and --per-chunk, and remember retrieval (RAG) may be the better answer for pure lookup
(see the RAG-vs-fine-tuning chapter). Fine-tune for BEHAVIOUR, retrieve for FACTS.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    from . import format as fmt
    from . import ingest as ing
    from . import quality, synth

    ap = argparse.ArgumentParser(description="documents -> training JSONL, one command")
    ap.add_argument("--docs", required=True, help="folder of raw documents")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--domain", required=True, help='e.g. "industrial pump support"')
    ap.add_argument("--mode", default="grounded", choices=["grounded", "chunks-only"])
    ap.add_argument("--eval", default=None, help="eval jsonl to exclude from training (contamination check)")
    ap.add_argument("--chunk-tokens", type=int, default=700)
    ap.add_argument("--max-chunks", type=int, default=0, help="0 = all; cap to control API cost")
    ap.add_argument("--per-chunk", type=int, default=3, help="Q&A pairs generated per chunk")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--eval-ratio", type=float, default=0.05)
    ap.add_argument("--system", default=None, help="system prompt written into every example")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # one interim folder PER output folder, so two builds (e.g. chunks-only and grounded) never mix files
    interim = out.parent / "interim" / out.name
    interim.mkdir(parents=True, exist_ok=True)

    print("1/4 ingest")
    chunks = interim / "chunks.jsonl"
    stats_ing = ing.ingest(Path(a.docs), chunks, size=a.chunk_tokens * 4, overlap=200)
    print(f"    {stats_ing} -> {chunks}")

    raw = interim / f"pairs_{a.mode}.jsonl"           # resumable: rerun the same command to continue
    if a.mode == "grounded":
        print("2/4 synthesise grounded Q&A (LLM calls)")
        # gen_system instructs the data-writing LLM; row_system is what the fine-tuned model sees at inference.
        # Keep them separate: training rows must never carry the generator's instructions.
        gen_system = (f"You write training data for a {a.domain} assistant. Given a source passage, write {{n}} "
                      "diverse question-answer pairs a real user would ask, answered ONLY from the passage. "
                      'Return strict JSON: {{"pairs": [{{"question": "...", "answer": "..."}}]}}. No markdown.')
        row_system = a.system or f"You are a helpful {a.domain} assistant."
        asyncio.run(synth.run(chunks, raw, n=a.per_chunk, concurrency=8, system_prompt=row_system,
                              limit=a.max_chunks or None, gen_system=gen_system))
    else:
        print("2/4 chunks-only mode: continued-pretraining style records, no LLM calls")
        rows = [json.loads(x) for x in chunks.read_text(encoding="utf-8").splitlines() if x.strip()]
        raw.write_text("".join(json.dumps({"messages": [
            {"role": "user", "content": f"Continue the {a.domain} documentation."},
            {"role": "assistant", "content": r["text"]}]}) + "\n" for r in rows), encoding="utf-8")
    print(f"    -> {raw}")

    print("3/4 quality: dedup, near-dup, PII, contamination")
    clean = interim / "clean.jsonl"
    stats = quality.run([raw], clean, report=interim / "quality.json", near_dup=0.8, pii="scrub",
                        contamination=[Path(a.eval)] if a.eval else None)
    print(f"    {stats} -> {clean}")

    print("4/4 format: chat template check, token-length filter, split")
    res = fmt.prepare([clean], out, eval_ratio=a.eval_ratio, seed=0, max_tokens=a.max_tokens,
                      tokenizer_name=a.tokenizer, system_prompt=a.system)
    print(f"    {res}")
    if not res.get("train"):
        # never report success for an empty dataset (found by running this command for the lab manual)
        print("\nERROR: 0 training rows survived. Read the stage reports above:")
        print(f"  quality: {stats}  (drop_repetitive = near-identical text; drop_* = the filter that removed rows)")
        print(f"  format : {res}  (too_long = raise --max-tokens; invalid = malformed messages)")
        print("  fixes : more varied documents, a larger --chunk-tokens, --mode grounded, or inspect "
              f"{raw} by hand")
        raise SystemExit(1)
    print(f"\nready: {out}/train.jsonl ({res['train']} rows) and {out}/eval.jsonl ({res.get('eval', 0)} rows)")
    print("next: tokenize_inspect one row, then the pre-flight doctor, then the baseline evaluation.")


if __name__ == "__main__":
    main()
