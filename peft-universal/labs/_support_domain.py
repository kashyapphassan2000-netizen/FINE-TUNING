"""The HIDDEN 'reality' used by labs 40-41: a stand-in for your production data.
The generator never sees this process. It only sees the rows it produces."""
import numpy as np

TIERS = ["free", "pro", "enterprise"]
ISSUES = ["billing", "outage", "howto", "bug"]


def reality(n, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        tier = rng.choice(TIERS, p=[0.55, 0.32, 0.13])
        issue = rng.choice(ISSUES, p=[0.34, 0.16, 0.34, 0.16])
        sev = int(np.clip(rng.normal({"billing": 2.4, "outage": 4.3, "howto": 1.6, "bug": 3.2}[issue], 0.9), 1, 5))
        base = {"free": 240, "pro": 90, "enterprise": 25}[tier]
        resp = float(max(1.0, rng.gamma(2.0, base / 2) * (1 + 0.25 * (sev - 3))))
        p_churn = 1 / (1 + np.exp(-(-2.6 + 0.55 * sev + 0.004 * min(resp, 600) - (1.1 if tier == "enterprise" else 0))))
        rows.append({"id": i, "customer_tier": str(tier), "issue_type": str(issue), "severity": sev,
                     "first_response_minutes": round(resp, 2), "churned": bool(rng.random() < p_churn)})
    return rows
