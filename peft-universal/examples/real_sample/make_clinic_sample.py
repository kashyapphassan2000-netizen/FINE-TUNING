"""Stand-in for a REAL export: 1,200 clinic appointments with known, realistic drivers of no-shows.

In your project this file is replaced by your real export (CSV/Excel/DB). It exists so every command in the
manual runs end to end, and so you can check that the profiler RECOVERS the relationships planted here:
  lead_time_days  up   -> no_show up        (long waits are forgotten)
  prior_no_shows  up   -> no_show up        (the strongest single driver in most published studies)
  sms_reminder    yes  -> no_show down
  distance_km     up   -> no_show up (weak)
  visit_type      follow_up has the lowest no-show rate; new_patient the highest
No names, dates of birth, phone numbers or IDs: there is nothing personal in it.

python examples/real_sample/make_clinic_sample.py   ->  examples/real_sample/clinic_noshow.jsonl
"""
import json
import math
from pathlib import Path

import numpy as np

rng = np.random.default_rng(42)
N = 1200
rows = []
for _ in range(N):
    age_band = rng.choice(["18-29", "30-44", "45-64", "65+"], p=[0.22, 0.30, 0.31, 0.17])
    visit_type = rng.choice(["new_patient", "follow_up", "procedure", "telehealth"], p=[0.25, 0.45, 0.15, 0.15])
    lead = float(np.clip(rng.gamma(2.0, 7.0), 0, 120))                 # right-skewed, like real booking data
    prior = int(rng.poisson(0.6 if age_band != "18-29" else 1.1))
    sms = bool(rng.random() < 0.7)
    dist = float(np.clip(rng.lognormal(2.0, 0.7), 0.5, 150))
    logit = (-2.2 + 0.025 * lead + 0.55 * prior - 0.8 * sms + 0.006 * dist
             + {"new_patient": 0.45, "follow_up": -0.35, "procedure": 0.0, "telehealth": -0.1}[visit_type]
             + {"18-29": 0.35, "30-44": 0.1, "45-64": -0.15, "65+": -0.2}[age_band])
    no_show = bool(rng.random() < 1 / (1 + math.exp(-logit)))
    rows.append({"age_band": age_band, "visit_type": visit_type, "lead_time_days": round(lead, 1),
                 "prior_no_shows": prior, "sms_reminder": "yes" if sms else "no", "distance_km": round(dist, 1),
                 "no_show": no_show})
out = Path(__file__).with_name("clinic_noshow.jsonl")
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"wrote {out} ({N} rows, no-show rate {sum(r['no_show'] for r in rows) / N:.3f})")
