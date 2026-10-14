"""One self-contained HTML report for owners, customers or the public: what was built, the evidence,
the capacity and the cost. Built ONLY from report files your pipeline produced - nothing typed by hand.

python -m peftkit.showcase --title "Pump support assistant" --compare reports/compare.json \
   --base reports/base.json --tuned reports/tuned.json --parity reports/parity.json \
   --loadtest reports/loadtest.json --capacity reports/capacity.json --run outputs/run/run_meta.json \
   --out reports/showcase.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _read(p):
    return json.loads(Path(p).read_text()) if p and Path(p).exists() else None


def _table(rows: list[list], head: list[str]) -> str:
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def build(title: str, compare=None, base=None, tuned=None, parity=None, loadtest=None, capacity=None, run=None) -> str:
    parts = [f"<h1>{html.escape(title)}</h1>",
             "<p class='note'>Every number below was read from files produced by the evaluation, serving and "
             "load-test tools. Nothing was entered by hand.</p>"]
    if run:
        v = run.get("versions", {})
        parts.append("<h2>What was trained</h2>" + _table([
            ["base model", run.get("base_model")], ["method", run.get("method")],
            ["trainable parameters", f"{run.get('trainable_params', 0):,} of {run.get('total_params', 0):,}"],
            ["optimizer steps", run.get("global_step")], ["training time (s)", run.get("train_seconds")],
            ["peak GPU memory (GB)", run.get("peak_vram_gb", "not measured (CPU run)")],
            ["hardware", v.get("gpu", "cpu")], ["torch / transformers / peft", f"{v.get('torch')} / {v.get('transformers')} / {v.get('peft')}"]],
            ["item", "value"]))
    if compare:
        parts.append("<h2>Did it get better? (same unseen test sets, before vs after)</h2>" + _table(
            [[r["suite"], r["before"], r["after"], f"{r['delta']:+.3f}"] for r in compare["rows"]],
            ["suite", "base model", "fine-tuned", "change"]) +
            f"<p><b>Verdict:</b> {html.escape(compare['verdict'])} &nbsp; <b>Regressions:</b> {compare['n_regressions']}</p>")
        if compare.get("regressions"):
            parts.append("<h3>Items that got worse (reported, not hidden)</h3>" + _table(
                [[x["suite"], x["id"], x["before"], x["after"], x["prediction"]] for x in compare["regressions"][:10]],
                ["suite", "id", "before", "after", "new answer"]))
    if tuned:
        ex = next(iter(tuned["suites"].values()))["items"][:5]
        parts.append("<h2>Sample answers</h2>" + _table([[x["id"], x["prediction"][:160], x["score"]] for x in ex],
                                                         ["id", "model answer", "score"]))
    if parity:
        parts.append("<h2>Is the deployed API the same model? (offline vs served)</h2>" + _table(
            [[r["suite"], r["offline"], r["served"], r["gap"], r["errors"]] for r in parity["rows"]],
            ["suite", "offline", "served", "gap", "errors"]) + f"<p><b>Parity:</b> {parity['verdict']}</p>")
    if loadtest:
        parts.append("<h2>How much load can one server take?</h2>" + _table(
            [[r["concurrency"], r["p50_s"], r["p95_s"], r["req_per_s"], r["out_tok_per_s"], r["errors"]] for r in loadtest["results"]],
            ["concurrent requests", "p50 s", "p95 s", "requests/s", "output tokens/s", "errors"]))
        rec = loadtest["recommendation"]
        if rec.get("meets_slo"):
            parts.append(f"<p>Per-replica capacity at p95 &le; {rec['slo_p95_s']} s: <b>{rec['max_concurrency_per_replica']}</b> "
                         f"concurrent requests; scale-up trigger at queue depth <b>{rec['suggested_scale_up_threshold']}</b>.</p>")
    if capacity:
        rows = [[p["provider"], p["gpu"], p["replicas"], p["gpus_total"], p.get("serve_cost_month", "price not set"),
                 p.get("cost_per_1M_output_tokens", "-")] for p in capacity["providers"] if p.get("replicas")]
        parts.append("<h2>What it costs to serve the expected traffic</h2>"
                     f"<p>{html.escape(capacity['traffic']['explain'])}</p>" +
                     _table(rows, ["provider", "GPU", "replicas", "GPUs", "$ / month", "$ / 1M output tokens"]) +
                     "<p class='note'>Prices are the rates entered in configs/providers.yaml; they are not quotes.</p>")
    parts.append("<h2>Honest limits</h2><ul><li>Scores are on the test sets listed above, not on every possible input.</li>"
                 "<li>Capacity numbers hold for the tested prompt/answer lengths and hardware only.</li>"
                 "<li>Re-run evaluation after every data, model or serving change.</li></ul>")
    css = ("body{font-family:system-ui,sans-serif;max-width:900px;margin:32px auto;padding:0 16px;color:#1b1b1b}"
           "table{border-collapse:collapse;width:100%;margin:8px 0 16px}th,td{border:1px solid #ccc;padding:6px 8px;"
           "text-align:left;font-size:14px}th{background:#12587f;color:#fff}.note{color:#555;font-size:13px}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style></head><body>{''.join(parts)}</body></html>"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Stakeholder showcase report (HTML)")
    ap.add_argument("--title", required=True)
    for k in ("compare", "base", "tuned", "parity", "loadtest", "capacity", "run"):
        ap.add_argument(f"--{k}", default=None)
    ap.add_argument("--out", default="reports/showcase.html")
    a = ap.parse_args(argv)
    doc = build(a.title, *(_read(getattr(a, k)) for k in ("compare", "base", "tuned", "parity", "loadtest", "capacity", "run")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(doc, encoding="utf-8")
    print(f"wrote {a.out} ({len(doc):,} bytes)")


if __name__ == "__main__":
    main()
