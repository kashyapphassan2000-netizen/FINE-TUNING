"""Lab 40: the whole loop on a business domain. Real seed -> profile -> generate -> gates -> utility.

The 'real' data here is produced by a hidden process (a stand-in for your production data). The
generator never sees that process: it only sees 600 rows, exactly like you will.
"""
import json, sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synthkit import gates as gatemod  # noqa: E402
from synthkit.contract import Contract  # noqa: E402
from synthkit.pipeline import run  # noqa: E402

sys.path.insert(0, str(ROOT / "labs"))
from _support_domain import reality  # noqa: E402

tmp = Path(tempfile.mkdtemp())
real = reality(600, seed=0)
(tmp / "contract.yaml").write_text("""
name: support_tickets_synthetic
version: "1.0"
fields:
  - {name: customer_tier, dtype: category, categories: [free, pro, enterprise]}
  - {name: issue_type, dtype: category, categories: [billing, outage, howto, bug]}
  - {name: severity, dtype: int, min: 1, max: 5}
  - {name: first_response_minutes, dtype: float, min: 0, max: 5000}
  - {name: churned, dtype: bool}
constraints:
  - "first_response_minutes >= 0"
  - "severity >= 1 and severity <= 5"
privacy: {pii_allowed: false, max_exact_match_rate: 0.02, min_dcr_ratio: 0.5}
quality: {max_psi: 0.2, min_utility_ratio: 0.85, max_duplicate_rate: 0.05}
""")
contract = Contract.load(tmp / "contract.yaml")
runs = {sm: run(real, contract, numeric=["severity", "first_response_minutes"],
                categorical=["customer_tier", "issue_type"], label="churned", n=2000, seed=0, smoothing=sm)
        for sm in (0.0, 0.25)}
print("Two runs of the SAME pipeline: smoothing=0 resamples real values exactly; smoothing=0.25 jitters them.")
print(f"{'setting':<16}{'max PSI':>9}{'DCR ratio':>11}{'exact match':>13}{'MIA AUC':>9}{'utility':>9}   decision")
for sm, r in runs.items():
    print(f"smoothing={sm:<6}{r['distribution']['summary']['max_psi']:>9}{r['privacy']['dcr']['ratio']:>11.3f}"
          f"{r['privacy']['exact_match_rate']:>13.3f}{r['privacy']['membership_inference_auc']:>9}"
          f"{r['utility']['ratio']:>9}   {r['gates']['decision']}")
print("\nDetail below is the smoothing=0.25 run (the one that ships).\n")
res = runs[0.25]

print(f"real seed: {len(real)} rows -> train {res['n_real_train']}, gold holdout {res['n_gold']}")
print(f"generated: {len(res['synth'])} synthetic rows\n")
print("CONTRACT:", "PASS" if res["contract_report"]["ok"] else res["contract_report"]["violations"])
print("\nDISTRIBUTION FIDELITY (real train vs synthetic)")
for c, v in res["distribution"]["numeric"].items():
    print(f"  {c:<26} PSI {v['psi']:<7} KS {v['ks']:<7} coverage {v['coverage']:<6} "
          f"real mean {v['real_mean']:<9} synth mean {v['synth_mean']}")
for c, v in res["distribution"]["categorical"].items():
    print(f"  {c:<26} TVD {v['tvd']}")
print(f"  correlation delta {res['distribution']['multivariate']['correlation_delta']}  "
      f"MMD {res['distribution']['multivariate']['mmd_rbf']}")
print("\nPRIVACY")
p = res["privacy"]
print(f"  DCR ratio {p['dcr']['ratio']:.3f} (synth->real vs real->real)  exact matches {p['exact_match_rate']:.3f}"
      f"  membership-inference AUC {p['membership_inference_auc']} (0.5 = no leakage)")
print("\nUTILITY (train where stated, ALWAYS test on the real gold holdout)")
u = res["utility"]
print(f"  real only    {u['real_only']}")
print(f"  synth only   {u['synth_only']}   utility ratio {u['ratio']}")
for m in u["mixtures"]:
    print(f"  mix synth={m['synthetic_share']:<6} n_real={m['n_real']:<5} n_synth={m['n_synth']:<5} "
          f"accuracy {m['accuracy']}" + (f" auc {m.get('auc')}" if m.get("auc") else ""))
d = res["gates"]["duplicates"]
print(f"\nDUPLICATES synthetic {d['synthetic']} vs real baseline {d['real_baseline']} -> excess {d['excess']}")
print("\nQUALITY GATES")
gatemod.print_report(res["gates"])
print("\nRead it honestly: the utility ratio and the gold-holdout mixtures decide whether this dataset ships.")
