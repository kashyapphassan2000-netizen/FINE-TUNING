"""Ten quality gates and a scorecard. One hard failure = REJECT the batch. Averages hide disasters."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateResult:
    name: str
    passed: bool
    value: float | str
    threshold: str
    hard: bool

    def line(self) -> str:
        mark = "PASS" if self.passed else ("FAIL" if self.hard else "WARN")
        return f"[{mark:<4}] {self.name:<22} value={self.value!s:<12} threshold={self.threshold}"


def run_gates(contract_report: dict, dist_report: dict, privacy_report: dict, utility: dict,
              dup_rate: float, diversity_score: float, judge_pass_rate: float | None = None,
              contract=None) -> dict:
    q = (contract.quality if contract else {}) or {}
    p = (contract.privacy if contract else {}) or {}
    max_psi = q.get("max_psi", 0.2)
    min_util = q.get("min_utility_ratio", 0.9)
    max_dup = q.get("max_duplicate_rate", 0.02)
    min_dcr = p.get("min_dcr_ratio", 0.9)
    max_exact = p.get("max_exact_match_rate", 0.0)
    g = [
        GateResult("1 schema", contract_report.get("ok", False), len(contract_report.get("violations", {})),
                   "0 violations", True),
        GateResult("2 constraints", not any(k.startswith("constraint") for k in contract_report.get("violations", {})),
                   sum(v for k, v in contract_report.get("violations", {}).items() if k.startswith("constraint")),
                   "0 violations", True),
        GateResult("3 excess duplicates", dup_rate <= max_dup, round(dup_rate, 4),
                   f"<= {max_dup} above the real baseline", False),
        GateResult("4 privacy: exact match", privacy_report["exact_match_rate"] <= max_exact,
                   round(privacy_report["exact_match_rate"], 4), f"<= {max_exact}", True),
        GateResult("5 privacy: DCR ratio", privacy_report["dcr"]["ratio"] >= min_dcr,
                   round(privacy_report["dcr"]["ratio"], 3), f">= {min_dcr}", True),
        GateResult("6 privacy: MIA AUC", privacy_report["membership_inference_auc"] <= 0.6,
                   privacy_report["membership_inference_auc"], "<= 0.6", True),
        GateResult("7 PII", not privacy_report.get("pii_hits"), privacy_report.get("pii_hits") or 0, "none", True),
        GateResult("8 distribution", dist_report["summary"]["max_psi"] <= max_psi,
                   dist_report["summary"]["max_psi"], f"max PSI <= {max_psi}", False),
        GateResult("9 combination coverage", diversity_score >= 0.9, round(diversity_score, 3),
                   ">= 0.90 of real category combinations", False),
        GateResult("10 utility (TSTR/TRTR)", utility["ratio"] >= min_util, utility["ratio"], f">= {min_util}", True),
    ]
    if judge_pass_rate is not None:
        g.append(GateResult("11 judge pass rate", judge_pass_rate >= 0.8, round(judge_pass_rate, 3), ">= 0.80", False))
    hard_fail = [x.name for x in g if x.hard and not x.passed]
    soft_fail = [x.name for x in g if not x.hard and not x.passed]
    return {"gates": g, "hard_failures": hard_fail, "soft_failures": soft_fail,
            "decision": "REJECT" if hard_fail else ("SHIP WITH WARNINGS" if soft_fail else "SHIP"),
            "scorecard": scorecard(dist_report, privacy_report, utility, diversity_score, dup_rate)}


def scorecard(dist_report: dict, privacy_report: dict, utility: dict, diversity_score: float,
              dup_rate: float) -> dict:
    """0-100 per axis. Transparent formulas, no magic: read them before quoting them."""
    psi = dist_report["summary"]["max_psi"]
    cov = [v.get("coverage", 0) for v in dist_report["numeric"].values()] or [0]
    return {
        "fidelity": max(0, round(100 * (1 - min(1.0, psi / 0.5)))),
        "coverage": round(100 * sum(cov) / len(cov)),
        "diversity": min(100, round(100 * diversity_score)),
        "utility": min(100, round(100 * utility["ratio"])),
        "privacy": max(0, round(100 * min(1.0, privacy_report["dcr"]["ratio"]) *
                                (1 - privacy_report["exact_match_rate"]) *
                                max(0.0, 1 - 2 * max(0.0, privacy_report["membership_inference_auc"] - 0.5)))),
        "cleanliness": max(0, round(100 * (1 - min(1.0, dup_rate / 0.1)))),
    }


def print_report(res: dict) -> None:
    for g in res["gates"]:
        print(g.line())
    print("\nscorecard:", res["scorecard"])
    print("decision:", res["decision"])
    if res["hard_failures"]:
        print("hard failures:", res["hard_failures"])
