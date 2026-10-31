"""Assemble outputs/report.md (spec section 16)."""
from __future__ import annotations

import json
import math
import sys

from src.utils.checks import summary_table
from src.utils.config import Paths, now_iso
from src.utils.logging import setup_logger

log = setup_logger("report_md")


def _load(path, default=None):
    if not path.exists():
        return default
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return path.read_text()


def _arm_rows(summary: dict) -> list[str]:
    order = ["T", "P0", "P0R", "P2", "S2", "S2R", "X"]
    cols = ["n", "FS", "claim_precision", "claim_precision_lenient", "hallucination_rate",
            "contradiction_rate", "evidence_coverage", "top1_mentioned_rate",
            "verdict_consistency_rate", "FS_pass_rate"]
    lines = ["| arm | " + " | ".join(c.replace("_", " ") for c in cols) +
             " | causal lift | suff (median) |", "|" + "---|" * (len(cols) + 3)]
    for arm in order:
        agg = summary.get(arm)
        if not agg:
            continue
        causal = agg.get("causal") or {}
        vals = []
        for c in cols:
            v = agg.get(c)
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lift = causal.get("causal_lift_mean")
        suff = causal.get("sufficiency_median_mean")
        lines.append("| " + arm + " | " + " | ".join(vals) +
                     f" | {lift:.3f} | {suff:.3f} |" if lift is not None and suff is not None
                     else "| " + arm + " | " + " | ".join(vals) + " | n/a | n/a |")
    return lines


def _ci_table(stats: dict) -> list[str]:
    metrics = [("FS", "FS"), ("claim_precision", "claim precision"),
               ("evidence_coverage", "coverage"),
               ("hallucination_rate", "hallucination"),
               ("contradiction_rate", "contradiction"),
               ("FS_pass_rate", "FS pass rate"), ("causal_lift", "causal lift")]
    lines = ["| arm | " + " | ".join(lbl + " (mean [95% CI])" for _, lbl in metrics) + " |",
             "|" + "---|" * (len(metrics) + 1)]
    for arm, m in stats.get("arms", {}).items():
        cells = []
        for key, _ in metrics:
            v = m.get(key) or {}
            mean = v.get("mean")
            if mean is None or not math.isfinite(float(mean)):
                cells.append("n/a")
            else:
                cells.append(f"{mean:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return lines


def _arm_scores(arm: str) -> dict[str, dict]:
    p = Paths.faithfulness / f"{arm}.jsonl"
    out: dict[str, dict] = {}
    if not p.exists():
        return out
    with p.open() as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["sample_id"]] = {"FS": rec["score"]["FS"],
                                         "pass": bool(rec["score"]["FS_pass"])}
    return out


def _repair_table() -> list[str]:
    """Repair before/after (RQ2), re-scored with the current metric.

    "before" is the source arm (P0 / S2) re-scored with the same extractor as
    "after" (P0R / S2R), so this table agrees with the per-arm table.
    """
    import numpy as np
    lines = ["| arm | n | mean FS before | mean FS after | pass before | pass after | "
             "mean rounds | samples repaired |",
             "|---|---|---|---|---|---|---|---|"]
    for arm, src in (("P0R", "P0"), ("S2R", "S2")):
        gen = Paths.rationales / f"{arm}.jsonl"
        if not gen.exists():
            continue
        rounds: dict[str, int] = {}
        with gen.open() as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if not rec.get("skipped"):
                        rounds[rec["sample_id"]] = int(rec.get("rounds_used", 0))
        before = _arm_scores(src)
        after = _arm_scores(arm)
        common = [sid for sid in rounds if sid in before and sid in after]
        if not common:
            continue
        b = np.array([before[sid]["FS"] for sid in common])
        a = np.array([after[sid]["FS"] for sid in common])
        pass_b = np.mean([1.0 if before[sid]["pass"] else 0.0 for sid in common])
        pass_a = np.mean([1.0 if after[sid]["pass"] else 0.0 for sid in common])
        n_rep = sum(1 for sid in common if rounds[sid] > 0)
        lines.append(f"| {arm} | {len(common)} | {b.mean():.3f} | {a.mean():.3f} | "
                     f"{pass_b:.3f} | {pass_a:.3f} | "
                     f"{np.mean([rounds[s] for s in common]):.2f} | {n_rep}/{len(common)} |")
    if len(lines) <= 2:
        return ["(no repair records found)"]
    return lines


def _pair_table(stats: dict) -> list[str]:
    lines = ["| comparison | metric | mean diff (a-b) | 95% CI | Wilcoxon p | n |",
             "|---|---|---|---|---|---|"]
    for name, comps in stats.get("comparisons", {}).items():
        for metric, c in comps.items():
            if not c:
                continue
            p = (c.get("wilcoxon") or {}).get("p")
            p_s = f"{p:.2e}" if isinstance(p, float) else "n/a"
            lines.append(f"| {name} | {metric} | {c['mean_diff']:+.3f} | "
                         f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | {p_s} | {c['n_pairs']} |")
    return lines


def _strata_table(stats: dict) -> list[str]:
    lines = ["| arm | stratum | n | FS | claim precision | hallucination | pass rate |",
             "|---|---|---|---|---|---|---|"]
    for arm, strata in stats.get("strata", {}).items():
        for st in sorted(strata):
            s = strata[st]
            lines.append(f"| {arm} | {st} | {s['n']} | {s['FS']:.3f} | "
                         f"{s['claim_precision']:.3f} | {s['hallucination_rate']:.3f} | "
                         f"{s['FS_pass_rate']:.3f} |")
    return lines


def _qualitative(arm: str, n_each: int = 3, pick: str = "both",
                 heading: str | None = None) -> list[str]:
    p = Paths.faithfulness / f"{arm}.jsonl"
    if not p.exists():
        return []
    recs = []
    with p.open() as fh:
        for line in fh:
            if line.strip():
                recs.append(json.loads(line))
    recs.sort(key=lambda r: r["score"]["FS"])
    if pick == "worst":
        picked = recs[:n_each]
    elif pick == "best":
        picked = recs[-n_each:][::-1]
    else:
        picked = recs[:n_each] + recs[-n_each:]
    lines = []
    if heading:
        lines += [f"### {heading}", ""]
    for r in picked:
        tag = "UNFAITHFUL" if r["score"]["FS"] < 0.5 else "faithful"
        lines += [f"### {r['sample_id']} — {tag} (FS={r['score']['FS']:.3f}, "
                  f"pass={r['score']['FS_pass']}, stratum={r.get('stratum')})", "",
                  f"> {r['text'].strip()[:900]}", "",
                  "Verifier:", ""]
        for c in r["score"]["per_claim"][:8]:
            lines.append(f"- `{c['feature']}` → **{c['status']}**"
                         + (f" (failed: {', '.join(c['failed_checks'])})"
                            if c["failed_checks"] else ""))
        lines.append("")
    return lines


def main() -> int:
    data_summary = _load(Paths.outputs / "data_summary.md", "(missing)")
    classifier = _load(Paths.outputs / "classifier_report.md", "(missing)")
    metrics = _load(Paths.models / "metrics.json", {})
    splits = _load(Paths.processed / "splits.json", {})
    eval_set = _load(Paths.eval / "eval_ids.json", {})
    llm_meta = _load(Paths.rationales / "llm_metadata.json", {})
    summary = _load(Paths.faithfulness / "all_arms_summary.json", {})
    stats = _load(Paths.faithfulness / "stats.json", {})
    validity = _load(Paths.outputs / "validity.json", {})
    judge = _load(Paths.outputs / "judge_crosscheck.json", {"status": "not run"})

    fs = (metrics.get("raw_test") or {}).get("roc_auc")
    fs_cal = (metrics.get("calibrated_test") or {}).get("roc_auc")

    lines: list[str] = [
        "# Faithfulness-checked LLM rationales for ransomware verdicts",
        "",
        f"Generated: {now_iso()}",
        "",
        "Study of whether LLM rationales for a memory-forensics ransomware "
        "classifier's verdicts are faithful to the underlying feature evidence. "
        "Everything is deterministic except rationale generation; the metric "
        "checks agreement with the model's evidence, not whether the model is "
        "right about ransomware.",
        "",
        "## 1. Setup",
        "",
        f"- Dataset: CIC-MalMem-2022 (Kaggle mirror of the official "
        f"`Obfuscated-MalMem2022.csv`; the UNB/CIC page is form-gated). "
        f"{splits.get('rows', {}).get('all', '?')} rows after dedupe, "
        f"55 features.",
        f"- Split (group-aware, seed {splits.get('seed')}): "
        f"train/val/test = {splits.get('rows', {}).get('train')}/"
        f"{splits.get('rows', {}).get('val')}/{splits.get('rows', {}).get('test')} rows; "
        f"leak check: {splits.get('leak_check')}.",
        f"- Eval set: N={eval_set.get('n_samples')} from the test split, strata "
        f"{eval_set.get('actual_counts')} (TP/TN/FN/FP), quotas {eval_set.get('quotas')}.",
        f"- Classifier: XGBoost, test ROC-AUC raw={fs:.5f}, calibrated={fs_cal:.5f}; "
        f"confusion {metrics.get('calibrated_test', {}).get('confusion')}; "
        f"Brier={metrics.get('calibrated_test', {}).get('brier'):.5f}.",
        "",
        "## 2. LLM details",
        "",
        f"- Backend: **{llm_meta.get('backend')}**, model id `{llm_meta.get('model_id')}`, "
        f"generator `{llm_meta.get('generator')}`.",
        f"- Temperature {llm_meta.get('temperature')}, prompt versions "
        f"{llm_meta.get('prompt_versions')}, repair rounds "
        f"{llm_meta.get('repair_rounds')}.",
        f"- {llm_meta.get('note', '')}",
        "",
        "## 3. Per-arm faithfulness",
        "",
        *_arm_rows(summary),
        "",
        "### Bootstrap 95% CIs (10,000 resamples)",
        "",
        *_ci_table(stats),
        "",
        "### Paired comparisons (Wilcoxon signed-rank + paired bootstrap)",
        "",
        *_pair_table(stats),
        "",
        "### Stratified by TP/TN/FN/FP",
        "",
        *_strata_table(stats),
        "",
        "### FS weight sensitivity",
        "",
        "| arm | alpha=0.25 | alpha=0.5 (main) | alpha=1.0 | coverage weight=0 | coverage weight=1 |",
        "|---|---|---|---|---|---|",
    ]
    for arm, agg in summary.items():
        sens = agg.get("sensitivity") or []
        if not sens:
            continue
        def _val(a, w):
            for s in sens:
                if s["alpha"] == a and s["coverage_weight"] == w:
                    return f"{s['mean_FS']:.3f}"
            return "?"
        lines.append(f"| {arm} | {_val(0.25, 0.5)} | {_val(0.5, 0.5)} | {_val(1.0, 0.5)} | "
                     f"{_val(0.5, 0.0)} | {_val(0.5, 1.0)} |")

    lines += [
        "",
        "### Repair loop (RQ2: does verify-and-repair help?)",
        "",
        *_repair_table(),
        "",
        "![repair](figs/repair_before_after.png)",
        "",
        "## 4. Metric validity",
        "",
        "```json",
        json.dumps({k: v for k, v in (validity.get("perturbations") or {}).items()}, indent=2),
        "```",
        "",
        "```json",
        json.dumps(validity.get("corrupted_evidence_control"), indent=2),
        "```",
        "",
        "### Extractor accuracy (gold set: agent-drafted, NOT independently annotated)",
        "",
        "```json",
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "items"}
                    for k, v in (validity.get("extractor_accuracy") or {}).items()}, indent=2),
        "```",
        "",
        "### Fluency vs faithfulness (RQ3)",
        "",
        "```json",
        json.dumps({k: v for k, v in (validity.get("fluency") or {}).items()
                    if not k.startswith("rows")}, indent=2),
        "```",
        "",
        "Within the zero-shot prose arm (the only natural arm with FS variance),",
        "none of the fluency proxies tracks faithfulness (|rho| <= 0.1, p >= 0.5).",
        "The positive pooled correlation with ROUGE-L is driven by the",
        "corrupted-evidence control, which is both lexically distant from the",
        "reference and unfaithful: overlap flags gross mismatch, not unfaithfulness.",
        "See the `P0_only_*` correlations for the unconfounded view.",
        "",
        "![fluency](figs/fluency_vs_fs.png)",
        "",
        "## 5. LLM-judge cross-check (optional)",
        "",
        f"Status: {judge.get('status')} — {judge.get('reason', judge.get('agreement', ''))}",
        "",
        "## 6. Qualitative examples",
        "",
        *_qualitative("S2", 3, "best", "Faithful examples (S2, structured)"),
        *_qualitative("P0", 3, "worst", "Unfaithful examples (P0, zero-shot prose)"),
        "## 7. Limitations",
        "",
        "- SHAP values are model attributions, not ground-truth causality; the causal "
        "tests are ablations under the model, not real-world interventions.",
        "- Median-imputation ablation can create off-distribution points; a "
        "marginal-resampling variant is reported alongside it.",
        "- CIC-MalMem-2022 is a lab-generated dataset (memory captures of known "
        "samples) and may not generalise to real-world ransomware; benign rows carry "
        "no capture ids, so residual near-duplicate leakage among benign rows cannot "
        "be fully excluded.",
        "- Small N (see eval set above); differences under ~0.05 are not reliable.",
        "- Glossary descriptions are hand-written and are the only allowed semantic "
        "source for the generator.",
        "- Single generator; when the backend is `agent_queue` the generator is the "
        "coding agent's underlying model (possibly closed-weight) — not an "
        "open-weight result.",
        "- The metric measures agreement between rationale and the model's evidence, "
        "not whether the model is correct about ransomware.",
        "",
        "## 8. Reproduction",
        "",
        "```bash",
        "make setup       # venv + deps",
        "make download    # CIC-MalMem-2022 -> data/raw",
        "make preprocess  # clean, dedupe, group-aware split",
        "make train       # tune + fit + calibrate + classifier report",
        "make evidence    # SHAP + evidence JSON + additivity/schema checks",
        "make generate    # arms T/P0/P0R/P2/S2/S2R (+X control); may exit 20 (agent_queue)",
        "make verify      # extraction + faithfulness + causal tests",
        "make validity    # metric validity + spot check",
        "make eval        # stats, plots, outputs/report.md",
        "make test        # pytest",
        "```",
        "",
        f"Seeds: everything uses seed {splits.get('seed')}.",
        "",
        "## 9. Acceptance checks",
        "",
        summary_table(),
        "",
        "## Appendix A — data summary",
        "",
        data_summary,
        "",
        "## Appendix B — classifier report",
        "",
        classifier,
        "",
    ]
    out = Paths.outputs / "report.md"
    out.write_text("\n".join(lines))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
