"""Lab 42: rare events. A generator that ignores the tail produces a model that looks great and fails."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synthkit.metrics import coverage  # noqa: E402
from synthkit.perturb import boost_rare  # noqa: E402
from synthkit.utility import train_test  # noqa: E402

rng = np.random.default_rng(0)
FRAUD_RATE = 0.004


def txn(rng, fraud=None):
    is_fraud = rng.random() < FRAUD_RATE if fraud is None else fraud
    if is_fraud:
        amount = float(rng.lognormal(7.4, 0.8))          # bigger, odd hours, new device
        hour = int(rng.choice([1, 2, 3, 4, 23]))
        device_age_days = float(rng.exponential(2))
    else:
        amount = float(rng.lognormal(4.2, 1.0))
        hour = int(np.clip(rng.normal(14, 4), 0, 23))
        device_age_days = float(rng.exponential(400) + 5)
    return {"amount": round(amount, 2), "hour": hour, "device_age_days": round(device_age_days, 1),
            "fraud": bool(is_fraud)}


real = [txn(rng) for _ in range(6000)]
gold = [txn(rng) for _ in range(4000)]
n_fraud_real = sum(r["fraud"] for r in real)
print(f"real seed: {len(real)} transactions, fraud rate {n_fraud_real / len(real):.4f} ({n_fraud_real} cases)")

naive = [txn(rng, fraud=False) for _ in range(6000)]        # a generator that never produced the tail
boosted = boost_rare([dict(r) for r in real], lambda r: r["fraud"], 0.05,
                     lambda rg: txn(rg, fraud=True), seed=1)
NUM = ["amount", "hour", "device_age_days"]
print(f"\n{'training set':<34}{'fraud rate':>12}{'tail coverage(amount)':>23}{'gold accuracy':>15}{'gold fraud recall':>19}")
for name, data in [("real seed (imbalanced)", real), ("synthetic, no rare events", naive),
                   ("synthetic + boosted rare events", boosted)]:
    rate = sum(r["fraud"] for r in data) / len(data)
    cov = coverage([r["amount"] for r in real], [r["amount"] for r in data])
    if len({r["fraud"] for r in data}) < 2:
        print(f"{name:<34}{rate:>12.4f}{cov:>23.3f}{'cannot train: one class only':>34}")
        continue
    res = train_test(data, gold, NUM, [], "fraud")
    from sklearn.metrics import recall_score
    from synthkit.utility import _xy
    Xtr, ytr, enc = _xy(data, NUM, [], "fraud")
    Xte, yte, _ = _xy(gold, NUM, [], "fraud", enc)
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
    rec = recall_score(yte, clf.predict(Xte), pos_label=True, zero_division=0)
    print(f"{name:<34}{rate:>12.4f}{cov:>23.3f}{res['accuracy']:>15}{rec:>19.3f}")

print("\nAccuracy is the wrong metric here: predicting 'never fraud' scores ~99.6%.")
print("Tail coverage and recall on the rare class are what tell you whether the synthetic data is useful.")
print("Boost rare events deliberately, record the ORIGINAL rate, and re-weight or re-calibrate at training time.")
