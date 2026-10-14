"""Lab 54: domain-specific SFT best practices - how much data, and what general data to mix in."""
import itertools, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "labs")]
from _sft_tasks import accuracy, build_model, load_adapter, sft, write_jsonl  # noqa: E402

RULE = {"leaks": "two", "jams": "one", "overheats": "four", "fails": "three"}
TRAIN_SUBJ = ["pump", "valve", "motor", "sensor", "filter", "bearing"]
TEST_SUBJ = ["impeller", "gauge", "seal", "pipe"]
TEMPLATES = ["the {s} {v}", "report : {s} {v}", "{v} on the {s}", "status of {s} : {v}"]
SEV = [f"severity {w}" for w in ("one", "two", "three", "four")]
# a second, unrelated capability the model must not lose ("general" data)
GEN = {"refund": "contact billing", "warranty": "contact service", "repair": "contact engineering"}
GEN_CANDS = list(GEN.values())
tmp = Path(tempfile.mkdtemp())
corpus = ([t.format(s=s, v=v) for t in TEMPLATES + ["{v} was reported on our {s} unit"]
           for s, v in itertools.product(TRAIN_SUBJ + TEST_SUBJ, RULE)]
          + SEV + [f"who handles {k}" for k in GEN] + GEN_CANDS)
model_dir = build_model(tmp / "base", corpus, hidden=128, layers=3)


def domain_rows(n_subj, n_tpl):
    return [{"messages": [{"role": "user", "content": t.format(s=s, v=v)},
                          {"role": "assistant", "content": f"severity {RULE[v]}"}]}
            for t in TEMPLATES[:n_tpl] for s, v in itertools.product(TRAIN_SUBJ[:n_subj], RULE)]


gen_rows = [{"messages": [{"role": "user", "content": f"who handles {k}"},
                          {"role": "assistant", "content": v}]} for k, v in GEN.items()] * 8
UNSEEN_TPL = "{v} was reported on our {s} unit"     # phrasing and machines both unseen
domain_items = [{"messages": [{"role": "user", "content": UNSEEN_TPL.format(s=s, v=v)}],
                 "answer": f"severity {RULE[v]}", "candidates": SEV}
                for s, v in itertools.product(TEST_SUBJ, RULE)]
gen_items = [{"messages": [{"role": "user", "content": f"who handles {k}"}], "answer": v,
              "candidates": GEN_CANDS} for k, v in GEN.items()]

print("A) HOW MUCH DOMAIN DATA? (4 machines AND a phrasing never seen in training, chance 0.25)")
print(f"{'training rows':>14}{'subjects':>10}{'templates':>11}{'held-out accuracy':>20}{'train loss':>12}")
for n_subj, n_tpl in [(1, 1), (2, 2), (4, 2), (6, 4)]:
    data = domain_rows(n_subj, n_tpl)
    reps = max(1, 96 // len(data))
    f = write_jsonl(data * reps, tmp / f"d{n_subj}{n_tpl}.jsonl")
    adapter, meta = sft(model_dir, f, tmp / f"run{n_subj}{n_tpl}", epochs=40, lr=5e-3, r=32)
    m, tok = load_adapter(model_dir, adapter)
    print(f"{len(data):>14}{n_subj:>10}{n_tpl:>11}{accuracy(m, tok, domain_items):>20}{meta['train_loss']:>12.3f}")

print("\nB) DOES DOMAIN-ONLY TRAINING COST THE OTHER CAPABILITY?")
print("   first teach BOTH capabilities, then fine-tune again on the domain, with and without replay")
base_data = domain_rows(2, 1) + gen_rows
f0 = write_jsonl(base_data * 6, tmp / "stage0.jsonl")
stage0, _ = sft(model_dir, f0, tmp / "stage0", epochs=30, lr=5e-3, r=32)
# merge stage 0 into the weights, so stage 1 CONTINUES from a model that already has both capabilities
# (training a fresh adapter on the base instead would measure nothing about forgetting)
from peftkit.merge import merge  # noqa: E402
merge(str(stage0), str(tmp / "stage0_merged"), dtype="float32")
stage0_model = tmp / "stage0_merged"
from _sft_tasks import load  # noqa: E402
m0, tok = load(stage0_model)
print(f"{'model':<34}{'domain accuracy':>18}{'general accuracy':>18}")
print(f"{'stage 0 (both capabilities)':<34}{accuracy(m0, tok, domain_items):>18}{accuracy(m0, tok, gen_items):>18}")

for name, extra in [("stage 1: domain only", []), ("stage 1: domain + 20% replay", gen_rows[:6])]:
    data = domain_rows(6, 4) + extra
    f = write_jsonl(data * 2, tmp / f"{name[:12]}.jsonl")
    adapter, _ = sft(stage0_model, f, tmp / f"s1_{name[:12]}", epochs=30, lr=5e-3, r=32)
    m, _ = load_adapter(stage0_model, adapter)
    print(f"{name:<34}{accuracy(m, tok, domain_items):>18}{accuracy(m, tok, gen_items):>18}")
print("  (both continue from the SAME stage-0 model; the replay rows are the only difference)")

print("""
Domain SFT checklist that transfers to real models:
  1. Coverage beats volume: every entity, every intent, every phrasing your users produce.
  2. Hold out ENTITIES, not just rows: testing on unseen machines/customers/products is the real test.
  3. Mix 5-30% general or previous-capability data (measured above; see Volume 2, chapter 10).
  4. Keep the system prompt varied, and include refusals and 'not in the documents' cases.
  5. Start at lr 1e-4..3e-4 with r=16 on a real model; this toy needs more because it is random-init.
  6. 1-3 epochs on real data; watch eval every 20-50 steps and keep the best checkpoint.
  7. Evaluate per segment: an average hides the segment your biggest customer cares about.
  8. Re-run the general suite every time. Domain gains that cost general ability are usually a bad trade.""")
