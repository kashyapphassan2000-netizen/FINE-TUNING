"""Lab 52: chat fine-tuning. Multi-turn data teaches the model to use the conversation, not just the last line."""
import sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "labs")]
from _sft_tasks import accuracy, build_model, load_adapter, sft, write_jsonl  # noqa: E402
from peftkit.data.templates import mask_report  # noqa: E402

RULE = {"leaks": "two", "jams": "one", "overheats": "four", "fails": "three"}
SUBJ = ["pump", "valve", "motor", "sensor"]
CANDS = [f"severity {w}" for w in ("one", "two", "three", "four")]
tmp = Path(tempfile.mkdtemp())
corpus = [f"the {s} {v} and next week severity {RULE[v]}" for s in SUBJ for v in RULE] + CANDS
model_dir = build_model(tmp / "base", corpus, hidden=128, layers=3)


def single_turn():
    return [{"messages": [{"role": "user", "content": f"the {s} {v}"},
                          {"role": "assistant", "content": f"severity {RULE[v]}"}]}
            for s in SUBJ for v in RULE]


ESCALATE = {"one": "two", "two": "three", "three": "four", "four": "four"}


def multi_turn():
    """Turn 2 needs BOTH the history (which fault) and a rule that never appears in single-turn data:
    'if it is not fixed by next week, escalate one level'. Copying turn 1 gives the wrong answer."""
    out = []
    for s in SUBJ:
        for v in RULE:
            out.append({"messages": [
                {"role": "user", "content": f"the {s} {v}"},
                {"role": "assistant", "content": f"severity {RULE[v]}"},
                {"role": "user", "content": "and next week ?"},
                {"role": "assistant", "content": f"severity {ESCALATE[RULE[v]]}"}]})
    return out


turn2_items = [{"messages": [{"role": "user", "content": f"the {s} {v}"},
                             {"role": "assistant", "content": f"severity {RULE[v]}"},
                             {"role": "user", "content": "and next week ?"}],
                "answer": f"severity {ESCALATE[RULE[v]]}", "candidates": CANDS}
               for s in SUBJ for v in RULE if RULE[v] != "four"]
turn1_items = [{"messages": [{"role": "user", "content": f"the {s} {v}"}],
                "answer": f"severity {RULE[v]}", "candidates": CANDS} for s in SUBJ for v in RULE]

print("Turn 2 ('and next week?') needs the history AND a rule that single-turn data never shows.\n")
print(f"{'training data':<28}{'rows':>6}{'turn-1 accuracy':>18}{'turn-2 accuracy':>18}")
for name, data in [("single-turn only", single_turn() * 8), ("multi-turn included", multi_turn() * 8)]:
    f = write_jsonl(data, tmp / f"{name[:6]}.jsonl")
    adapter, _ = sft(model_dir, f, tmp / f"run_{name[:6]}", epochs=30, lr=5e-3, r=32)
    m, tok = load_adapter(model_dir, adapter)
    print(f"{name:<28}{len(data):>6}{accuracy(m, tok, turn1_items):>18}{accuracy(m, tok, turn2_items):>18}")
print("  (chance = 0.25)")

print("\nHow multi-turn examples are trained: EVERY assistant turn is a target, every user turn is masked.")
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(model_dir)
rep = mask_report(tok, multi_turn()[0]["messages"], "chatml")
print(f"  one 2-turn example: {rep['tokens']} tokens, {rep['trained_tokens']} trained "
      f"({rep['trained_fraction']:.0%})")
print(f"  the loss sees only: {rep['trained_text']!r}")

print("""
Chat fine-tuning checklist:
  - one example = one CONVERSATION, not one message pair
  - every assistant turn trained, every user/system turn masked (needs {% generation %} markers)
  - vary the system prompt in training, or the model will only work with the exact one you used
  - include conversations that END with a refusal, a clarifying question, or 'not in the documents'
  - include long conversations: if you train on 2 turns and serve 10, the later turns are out of
    distribution and quality falls off
  - keep the serving template identical to the training template, byte for byte""")
