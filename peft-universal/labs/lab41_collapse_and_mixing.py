"""Lab 41: model collapse on TABULAR data, and the mixing ratio that prevents it."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "labs"))
from _support_domain import reality  # noqa: E402
from synthkit.generate import CopulaGenerator  # noqa: E402
from synthkit.metrics import psi, categorical_tvd  # noqa: E402
from synthkit.utility import train_test  # noqa: E402

NUM = ["severity", "first_response_minutes"]
CAT = ["customer_tier", "issue_type", "churned"]
real = reality(1200, seed=1)
train, gold = real[:900], real[900:]


def fit_sample(rows, n, seed):
    return CopulaGenerator(numeric=NUM, categorical=CAT, seed=seed).fit(rows).sample(n)


print("A) GENERATION-ON-GENERATION (each model trained only on the previous model's output)")
print(f"{'generation':<12}{'PSI response':>14}{'TVD issue':>11}{'std(response)':>15}{'accuracy on REAL gold':>24}")
cur = train
for g in range(1, 6):
    cur = fit_sample(cur, 900, seed=g)
    p = psi([r["first_response_minutes"] for r in train], [r["first_response_minutes"] for r in cur])
    t = categorical_tvd([r["issue_type"] for r in train], [r["issue_type"] for r in cur])
    sd = float(np.std([r["first_response_minutes"] for r in cur]))
    acc = train_test(cur, gold, NUM, ["customer_tier", "issue_type"], "churned")["accuracy"]
    print(f"gen {g:<8}{p:>14.4f}{t:>11.4f}{sd:>15.1f}{acc:>24}")

print("\nB) SAME LOOP, but 20% real data mixed back in at every generation")
print(f"{'generation':<12}{'PSI response':>14}{'TVD issue':>11}{'std(response)':>15}{'accuracy on REAL gold':>24}")
cur = train
rng = np.random.default_rng(0)
for g in range(1, 6):
    mixed = fit_sample(cur, 720, seed=10 + g) + list(rng.permutation(np.array(train, dtype=object))[:180])
    cur = [dict(m) for m in mixed]
    p = psi([r["first_response_minutes"] for r in train], [r["first_response_minutes"] for r in cur])
    t = categorical_tvd([r["issue_type"] for r in train], [r["issue_type"] for r in cur])
    sd = float(np.std([r["first_response_minutes"] for r in cur]))
    acc = train_test(cur, gold, NUM, ["customer_tier", "issue_type"], "churned")["accuracy"]
    print(f"gen {g:<8}{p:>14.4f}{t:>11.4f}{sd:>15.1f}{acc:>24}")

print(f"\nreal std(first_response_minutes) = {np.std([r['first_response_minutes'] for r in train]):.1f}")
print("Recursive training on your own outputs shrinks the tails first. Keeping real data in the loop is the")
print("cheapest defence there is; lineage metadata is what lets you keep doing it release after release.")
