"""Lab 53: chain-of-thought fine-tuning. Same task, answer-only vs reasoning-then-answer."""
import itertools, random, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "labs")]
import torch  # noqa: E402
from _sft_tasks import answer_nll, build_model, load_adapter, sft, write_jsonl  # noqa: E402

NUM = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
V = {w: i + 1 for i, w in enumerate(NUM)}
triples = [(a, b, c) for a, b, c in itertools.product(NUM[:3], NUM[:3], NUM[:3]) if V[a] + V[b] + V[c] <= 9]
random.Random(0).shuffle(triples)
train_t, test_t = triples[:18], triples[18:]
tmp = Path(tempfile.mkdtemp())


def direct(t):
    a, b, c = t
    return {"messages": [{"role": "user", "content": f"sum {a} {b} {c}"},
                         {"role": "assistant", "content": f"answer {NUM[V[a] + V[b] + V[c] - 1]}"}]}


def cot(t):
    a, b, c = t
    mid = NUM[V[a] + V[b] - 1]
    tot = NUM[V[a] + V[b] + V[c] - 1]
    return {"messages": [{"role": "user", "content": f"sum {a} {b} {c}"},
                         {"role": "assistant",
                          "content": f"step {a} and {b} is {mid} then {mid} and {c} is {tot} answer {tot}"}]}


corpus = [m["content"] for t in triples for m in direct(t)["messages"] + cot(t)["messages"]]
model_dir = build_model(tmp / "base", corpus, hidden=128, layers=3)
print(f"task: add three number words, answer in words. {len(train_t)} training triples, "
      f"{len(test_t)} held-out triples, 9 candidate answers (chance = 0.111).\n")

results = {}
for name, fn in [("answer only", direct), ("chain of thought", cot)]:
    data = [fn(t) for t in train_t] * 10
    f = write_jsonl(data, tmp / f"{name[:6]}.jsonl")
    adapter, meta = sft(model_dir, f, tmp / f"run_{name[:6]}", epochs=12, lr=3e-3)
    m, tok = load_adapter(model_dir, adapter)

    def score(t, with_reasoning):
        a, b, c = t
        msgs = [{"role": "user", "content": f"sum {a} {b} {c}"}]
        best, best_nll = None, 1e9
        for cand in NUM:
            if with_reasoning:
                mid = NUM[V[a] + V[b] - 1]
                text = f"step {a} and {b} is {mid} then {mid} and {cand} is {cand} answer {cand}"
            else:
                text = f"answer {cand}"
            nll = answer_nll(m, tok, msgs, text)
            if nll < best_nll:
                best, best_nll = cand, nll
        return best == NUM[V[a] + V[b] + V[c] - 1]

    acc_tr = sum(score(t, name.startswith("chain")) for t in train_t) / len(train_t)
    acc_te = sum(score(t, name.startswith("chain")) for t in test_t) / len(test_t)
    toks = sum(len(tok(fn(t)["messages"][1]["content"])["input_ids"]) for t in train_t) / len(train_t)
    results[name] = (acc_tr, acc_te, toks, meta["train_loss"])
    print(f"{name:<18} train acc {acc_tr:.3f}   held-out acc {acc_te:.3f}   "
          f"mean answer tokens {toks:.1f}   train loss {meta['train_loss']:.3f}")

print("""
What CoT fine-tuning actually is: put the intermediate steps in the ASSISTANT target, so the model
learns to produce them, and then conditions its final answer on its own steps.

Honest reading of the numbers above: this is a 3-layer randomly initialised model on 18 examples.
It is a demonstration of the mechanics and the COST, not evidence about CoT quality. The published
claim (reasoning traces help multi-step tasks, and distilling a strong model's traces transfers some
of that ability) needs a real pretrained model to reproduce; do that before believing any table.

What transfers to real projects regardless of this toy:
  - CoT targets are 3-5x longer -> 3-5x the training tokens and 3-5x the generation cost per answer
  - the traces must be CORRECT: training on wrong reasoning that reaches the right answer teaches
    the model to fake reasoning (verify traces, or keep only traces whose final answer is right)
  - never train on reasoning you cannot show to a user if it leaks (some traces contain the answer key)
  - if you only need the answer, consider training answer-only and paying for reasoning at inference""")
