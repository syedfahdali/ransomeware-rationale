"""Deterministic claim verification + aggregate faithfulness metrics.

Implements spec sections 8.1 (A/B) and the edge cases of 13.1: zero claims,
duplicate claims, conflicting duplicates, values with units/k/M suffixes.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

LEVELS = ["very_low", "low", "typical", "high", "very_high"]

LEVEL_ALIASES = {
    "very_high": "very_high", "veryhigh": "very_high", "extremely_high": "very_high",
    "exceptionally_high": "very_high",
    "high": "high", "elevated": "high", "raised": "high", "large": "high",
    "typical": "typical", "normal": "typical", "usual": "typical", "average": "typical",
    "moderate": "typical", "unremarkable": "typical",
    "low": "low", "reduced": "low", "small": "low",
    "very_low": "very_low", "verylow": "very_low", "extremely_low": "very_low",
    "exceptionally_low": "very_low",
}

REMEDIATION_RE = re.compile(
    r"\b(quarantine|isolate the|isolate it|reimage|re-image|disconnect the|"
    r"restore (?:from )?backups?|patch the system|block the|remove the sample|"
    r"escalate to the soc|contact the soc)\b", re.I)
FAMILY_RE = re.compile(
    r"\b(conti|maze|pysa|ako|shade|troldesh|wannacry|lockbit|ryuk|revil)\b", re.I)


@dataclass
class ClaimResult:
    feature: str
    status: str                      # supported | contradicted | unsupported | out_of_topk
    failed_checks: list[str] = field(default_factory=list)
    stated_direction: str | None = None
    stated_level: str | None = None
    stated_value: float | None = None
    evidence_direction: str | None = None
    evidence_level: str | None = None
    evidence_value: float | None = None
    extractor: str | None = None
    text: str | None = None


def normalize_level(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return LEVEL_ALIASES.get(s)


def parse_number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower().replace(",", "").replace(" ", "")
    m = re.fullmatch(r"([-+]?\d*\.?\d+)([km])?", s)
    if not m:
        return None
    num = float(m.group(1))
    if m.group(2) == "k":
        num *= 1e3
    elif m.group(2) == "m":
        num *= 1e6
    return num


def verify_claim(claim: dict, evidence: dict, tol: float = 0.05) -> ClaimResult:
    topk = {f["name"].lower(): f for f in evidence["top_features"]}
    all_lower = {n.lower() for n in evidence.get("all_feature_names", [])}
    raw_name = str(claim.get("feature", "")).strip()
    name = raw_name.lower()
    res = ClaimResult(feature=raw_name, status="supported",
                      extractor=claim.get("extractor"), text=claim.get("text"))
    if name not in topk:
        res.status = "out_of_topk" if name in all_lower else "unsupported"
        res.failed_checks = ["feature_exists"]
        return res
    e = topk[name]
    res.evidence_direction = e["direction"]
    res.evidence_level = e["qualitative_level"]
    res.evidence_value = float(e["value"])

    stated_dir = claim.get("direction")
    res.stated_direction = stated_dir
    if stated_dir is not None and str(stated_dir) != e["direction"]:
        res.failed_checks.append("direction_ok")

    stated_level = normalize_level(claim.get("level"))
    res.stated_level = stated_level
    if stated_level is not None and stated_level != e["qualitative_level"]:
        res.failed_checks.append("level_ok")

    stated = claim.get("values_all")
    if stated is None:
        v = parse_number(claim.get("value_stated"))
        stated = [v] if v is not None else []
    else:
        stated = [v for v in (parse_number(x) for x in stated) if v is not None]
    if stated:
        res.stated_value = stated[0]
        v_ev = float(e["value"])
        denom = max(abs(v_ev), 1.0)
        if not any(abs(v - v_ev) / denom <= tol for v in stated):
            res.failed_checks.append("value_ok")

    res.status = "supported" if not res.failed_checks else "contradicted"
    return res


def _dedupe(results: list[ClaimResult]) -> list[ClaimResult]:
    """Dedupe identical claims per feature; conflicting duplicates -> contradiction."""
    by_feature: dict[str, list[ClaimResult]] = {}
    for r in results:
        by_feature.setdefault(r.feature.lower(), []).append(r)
    deduped: list[ClaimResult] = []
    for name, group in by_feature.items():
        if len(group) == 1:
            deduped.append(group[0])
            continue
        signatures = {(r.stated_direction, r.stated_level, r.stated_value, r.status)
                      for r in group}
        if len(signatures) == 1:
            deduped.append(group[0])
        else:
            first = group[0]
            first.status = "contradicted"
            first.failed_checks = list(set(first.failed_checks + ["conflicting_duplicates"]))
            deduped.append(first)
    return deduped


def score_rationale(rationale: dict, evidence: dict, value_tol: float = 0.05,
                    prob_tol: float = 0.02) -> dict:
    claims = [c for c in (rationale.get("claims") or []) if isinstance(c, dict)]
    results = _dedupe([verify_claim(c, evidence, value_tol) for c in claims])
    n = len(results)
    n_sup = sum(r.status == "supported" for r in results)
    n_unsup = sum(r.status in ("unsupported", "out_of_topk") for r in results)
    n_contra = sum(r.status == "contradicted" for r in results)

    mentioned = {r.feature.lower() for r in results}
    topk = evidence["top_features"]
    mass_total = sum(abs(f["shap"]) for f in topk) or 1.0
    mass_cov = sum(abs(f["shap"]) for f in topk if f["name"].lower() in mentioned) / mass_total

    stated_p = rationale.get("probability_stated")
    verdict_ok = (str(rationale.get("verdict")) == evidence["verdict"])
    if verdict_ok and stated_p is not None:
        try:
            verdict_ok = abs(float(stated_p) - float(evidence["probability_ransomware"])) <= prob_tol
        except (TypeError, ValueError):
            verdict_ok = False

    denom = max(n, 1)
    precision = n_sup / denom
    hallucination_rate = n_unsup / denom
    contradiction_rate = n_contra / denom
    no_claims = n == 0
    fs = 0.0 if no_claims else float(verdict_ok) * (precision ** 0.5) * (0.5 + 0.5 * mass_cov)
    fs_pass = bool(not no_claims and n_unsup == 0 and n_contra == 0 and verdict_ok)

    # lenient arm: adjacent-level tolerance only
    lenient_supported = 0
    for r in results:
        fails = [f for f in r.failed_checks if f != "level_ok"]
        if r.status == "supported" or (r.status == "contradicted" and not fails):
            if r.status == "supported":
                lenient_supported += 1
            else:
                idx_c = LEVELS.index(r.stated_level) if r.stated_level in LEVELS else None
                idx_e = LEVELS.index(r.evidence_level) if r.evidence_level in LEVELS else None
                if idx_c is not None and idx_e is not None and abs(idx_c - idx_e) == 1:
                    lenient_supported += 1
    lenient_precision = lenient_supported / denom

    # policy diagnostics (not part of FS): remediation advice, family speculation
    text_all = " ".join([str(rationale.get("summary") or ""), str(rationale.get("confidence_statement") or "")]
                        + [str(c.get("text") or "") for c in claims])
    violations = []
    if REMEDIATION_RE.search(text_all):
        violations.append("remediation advice")
    if FAMILY_RE.search(text_all):
        violations.append("malware-family speculation")

    return {
        "n_claims": n,
        "claim_precision": precision,
        "claim_precision_lenient": lenient_precision,
        "hallucination_rate": hallucination_rate,
        "contradiction_rate": contradiction_rate,
        "evidence_coverage": mass_cov,
        "top1_mentioned": topk[0]["name"].lower() in mentioned,
        "verdict_ok": bool(verdict_ok),
        "FS": fs,
        "FS_pass": fs_pass,
        "no_claims": no_claims,
        "policy_violations": violations,
        "status_counts": {"supported": n_sup, "unsupported": n_unsup, "contradicted": n_contra},
        "per_claim": [asdict(r) for r in results],
    }
