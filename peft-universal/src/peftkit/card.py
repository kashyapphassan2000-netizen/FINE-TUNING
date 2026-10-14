"""Model card (README.md) for the Hub, written ONLY from files your pipeline produced: nothing typed by hand
except the description and the intended use.

python -m peftkit.card --run outputs/RUN/run_meta.json --compare reports/compare.json --bench reports/bench_gate.json \
    --dataset HF_USER/clinic-noshow --license apache-2.0 --description "Clinic operations assistant" \
    --intended-use "Scheduling staff triage; not for clinical decisions" --out outputs/RUN/merged/README.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(p):
    return json.loads(Path(p).read_text()) if p and Path(p).exists() else None


def build(run: dict, compare: dict | None, bench: dict | None, dataset: str | None, license: str,
          description: str, intended_use: str, language: str = "en", adapter_only: bool = False,
          base_model: str | None = None) -> str:
    base = base_model or run.get("base_model", "unknown")
    on_hub = "/" in base and not base.startswith((".", "/", "~")) and ":" not in base and "\\" not in base
    cfg = run.get("config", {})
    m, t, d = cfg.get("method", {}), cfg.get("train", {}), cfg.get("data", {})
    v = run.get("versions", {})
    meta = ["---", f"license: {license}"] + ([f"base_model: {base}"] if on_hub else []) + [f"language: [{language}]",
            "pipeline_tag: text-generation", f"library_name: {'peft' if adapter_only else 'transformers'}",
            "tags: [fine-tuned, " + ", ".join(x for x in (run.get("method"), "peft-universal") if x) + "]"]
    if dataset:
        meta.append(f"datasets: [{dataset}]")
    meta.append("---")
    rows = [f"| base model | `{base}` (revision `{cfg.get('model', {}).get('revision', 'main')}`) |",
            f"| method | {run.get('method')} (r={m.get('r')}, alpha={m.get('alpha')}, targets={m.get('target_modules')}) |",
            f"| trainable parameters | {run.get('trainable_params', 0):,} of {run.get('total_params', 0):,} |",
            f"| optimizer steps / epochs | {run.get('global_step')} / {t.get('epochs')} |",
            f"| learning rate / effective batch | {t.get('lr')} / {t.get('per_device_batch_size', 0) * t.get('grad_accum', 0)} per GPU |",
            f"| max sequence length | {cfg.get('model', {}).get('max_seq_length')} |",
            f"| loss | {'completion-only' if d.get('completion_only_loss') else 'assistant-only' if d.get('assistant_only_loss') else 'all tokens'} |",
            f"| hardware / peak VRAM | {v.get('gpu', 'cpu')} / {run.get('peak_vram_gb', 'n/a')} GB |",
            f"| training time | {run.get('train_seconds')} s |",
            f"| software | torch {v.get('torch')}, transformers {v.get('transformers')}, peft {v.get('peft')}, trl {v.get('trl')} |"]
    src = f"[`{base}`](https://huggingface.co/{base})" if on_hub else f"`{base}` (local model; pass --base-model ORG/NAME)"
    out = meta + ["", f"# {description}", "", f"Fine-tuned from {src} "
                  "with [peft-universal](https://github.com/kashyapphassan2000-netizen/FINE-TUNING).", "",
                  "## Intended use", "", intended_use, "", "## Training", "", "| item | value |", "|---|---|", *rows]
    if dataset:
        out += ["", f"Training data: [`{dataset}`](https://huggingface.co/datasets/{dataset})."]
    if compare:
        out += ["", "## Evaluation (held-out test sets, base vs fine-tuned)", "",
                "| suite | base | fine-tuned | change |", "|---|---|---|---|"]
        out += [f"| {r['suite']} | {r['before']} | {r['after']} | {r['delta']:+.3f} |" for r in compare["rows"]]
        out += ["", f"Verdict: **{compare['verdict']}**. Items that got worse: {compare['n_regressions']}."]
    if bench:
        out += ["", "## General benchmarks (forgetting check, lm-evaluation-harness)", "",
                "| task | metric | base | fine-tuned | delta |", "|---|---|---|---|---|"]
        out += [f"| {r['task']} | {r['metric']} | {r['base']} | {r['tuned']} | {r['delta']:+.4f} |" for r in bench["rows"]]
    load = (f'from peft import PeftModel\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n'
            f'base = AutoModelForCausalLM.from_pretrained("{base}", dtype="auto", device_map="auto")\n'
            f'model = PeftModel.from_pretrained(base, "<this repo>")\ntok = AutoTokenizer.from_pretrained("<this repo>")'
            if adapter_only else
            'from transformers import AutoModelForCausalLM, AutoTokenizer\n'
            'model = AutoModelForCausalLM.from_pretrained("<this repo>", dtype="auto", device_map="auto")\n'
            'tok = AutoTokenizer.from_pretrained("<this repo>")')
    out += ["", "## How to use", "", "```python", load,
            'enc = tok.apply_chat_template([{"role": "user", "content": "..."}], add_generation_prompt=True,',
            '                              return_tensors="pt", return_dict=True).to(model.device)',
            'print(tok.decode(model.generate(**enc, max_new_tokens=256)[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))',
            "```", "", "## Limitations", "",
            "- Scores above are on the listed test sets only, not on every possible input.",
            "- It can produce confident wrong answers. Facts that change belong in retrieval, not in the weights.",
            f"- The base model's licence and acceptable-use terms apply to this derivative ({license}).", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Write a Hub model card from the pipeline's report files")
    ap.add_argument("--run", required=True, help="outputs/RUN/run_meta.json")
    ap.add_argument("--compare", default=None)
    ap.add_argument("--bench", default=None)
    ap.add_argument("--dataset", default=None, help="Hub dataset id used for training")
    ap.add_argument("--license", required=True, help="the BASE model's licence id, e.g. apache-2.0, llama3.1")
    ap.add_argument("--description", required=True)
    ap.add_argument("--intended-use", required=True)
    ap.add_argument("--language", default="en")
    ap.add_argument("--base-model", default=None, help="Hub id of the base, if you trained from a local copy")
    ap.add_argument("--adapter-only", action="store_true", help="card for an adapter repo (library: peft)")
    ap.add_argument("--out", required=True, help="README.md inside the folder you will publish")
    a = ap.parse_args(argv)
    text = build(_read(a.run), _read(a.compare), _read(a.bench), a.dataset, a.license, a.description,
                 a.intended_use, a.language, a.adapter_only, a.base_model)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(f"wrote {a.out} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
