"""Lab 44: LLM-generated text records, multi-judge scoring with calibration, and leakage checks.
Runs against a local fake OpenAI-compatible server, so it is reproducible with no API key."""
import asyncio, json, socket, sys, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from peftkit.data.llm import LLM  # noqa: E402
from synthkit.judge import agreement, best_threshold, multi_judge  # noqa: E402
from synthkit.privacy import ngram_overlap, scan_pii  # noqa: E402
from synthkit.generate import llm_records  # noqa: E402

app = FastAPI()
COUNT = {"n": 0}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    sysmsg = body["messages"][0]["content"]
    user = body["messages"][-1]["content"]
    COUNT["n"] += 1
    if "Score the RECORD" in sysmsg:                     # judge
        good = "template" not in user and "@" not in user
        strict = "strict" in sysmsg
        base = {"correctness": 4 if good else 1, "completeness": 4 if good else 2,
                "instruction": 4 if good else 2, "safety": 0 if "@" in user else 4,
                "style": 1 if "template" in user else (3 if strict else 4)}
        return {"choices": [{"message": {"content": json.dumps({**base, "rationale": "r"})}}]}
    records = [                                           # generator: 2 good, 1 templated, 1 with PII
        {"ticket": "Payment failed after OTP on the mobile app, retried twice, card was charged once.",
         "issue_type": "billing", "severity": 3},
        {"ticket": "Dashboard has been down since 03:10 UTC for the whole EU region.",
         "issue_type": "outage", "severity": 5},
        {"ticket": "This is a template response about your issue. Thank you for contacting support.",
         "issue_type": "howto", "severity": 1},
        {"ticket": "Contact me at raj.kumar@example.com or +91 98765 43210 about the refund.",
         "issue_type": "billing", "severity": 2}]
    return {"choices": [{"message": {"content": json.dumps({"records": records})}}]}


s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
th = threading.Thread(target=server.run, daemon=True); th.start()
while not server.started:
    time.sleep(0.05)
import os  # noqa: E402
os.environ.update(OPENAI_BASE_URL=f"http://127.0.0.1:{port}/v1", OPENAI_API_KEY="sk-test", SYNTH_MODEL="fake-a")

real_tickets = ["Payment failed after OTP, the amount was debited but the order did not confirm.",
                "The dashboard is unavailable for all users in the EU since this morning."]
llm = LLM(model="fake-a")
recs = asyncio.run(llm_records(llm, "customer support", {"ticket": "text", "issue_type": "category",
                                                         "severity": "int 1-5"}, 4,
                               ["no personal data", "one issue per ticket"], seed_examples=[{"ticket": t} for t in real_tickets]))
print(f"generated {len(recs)} records from the schema + 2 real examples\n")

judges = [LLM(model="fake-a"), LLM(model="fake-b"), LLM(model="fake-c")]
print(f"{'record':<46}{'score':>7}{'spread':>8}{'human?':>8}  weakest criterion")
scored = []
for r in recs:
    res = asyncio.run(multi_judge(judges, r))
    weakest = min(res["parts"], key=res["parts"].get)
    scored.append(res["score"])
    print(f"{r['ticket'][:44]:<46}{res['score']:>7}{res['spread']:>8}{str(res['needs_human']):>8}  "
          f"{weakest}={res['parts'][weakest]}")

human = [1, 1, 0, 0]          # a human reviewed the same four: keep, keep, reject (templated), reject (PII)
print("\njudge vs human calibration:", agreement(scored, human))
print("best threshold search:      ", best_threshold(scored, human))

print("\nleakage and PII checks on the generated batch:")
print("  PII hits:", scan_pii(recs, ["ticket"]))
print(f"  8-gram overlap with the real seed examples: {ngram_overlap(real_tickets, [r['ticket'] for r in recs]):.3f}")
print(f"  LLM calls made: {COUNT['n']} (1 generation + 4 records x 3 judges)")
server.should_exit = True; th.join(timeout=5)
print("\nRules this lab encodes: several judges, decomposed criteria, spread triggers human review,")
print("calibrate the threshold against human labels, and scan every batch for PII and copied text.")
