"""Everyday triage CLI — use the trained classifier on a single memory sample.

This is the "real life" entry point: instead of regenerating the whole study, it
takes ONE memory-forensics sample, runs the trained classifier, prints the SHAP
evidence, optionally asks an LLM to explain the verdict, and then verifies that
explanation with the same deterministic faithfulness metric the study uses.

    python -m src.cli check --sample test_031534
    python -m src.cli check --sample test_031534 --explain llm --repair 2
    python -m src.cli check --csv my_memory_dump.csv
    python -m src.cli batch --csv data/raw/CIC-MalMem-2022.csv --limit 100
    python -m src.cli template --out my_sample.csv
    python -m src.cli info

Exit codes: 0 = ok, 1 = --fail-on-unfaithful and the explanation failed,
2 = bad input/usage, 20 = waiting for the agent queue (see queue/HOWTO.md).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.evidence.build_evidence import build_one, load_glossary
from src.evidence.shap_builder import compute_shap
from src.faithfulness.extract_llm import extract_claims_smart
from src.faithfulness.extract_rules import infer_verdict_and_probability
from src.faithfulness.verify import score_rationale
from src.generation.prompts import (build_user, fmt_value, load_fewshot, load_system,
                                    parse_json_response, render_evidence_table)
from src.generation.repair import build_feedback
from src.generation.template_baseline import template_rationale
from src.llm.base import LLMError, WaitingForAgent, get_backend
from src.models.calibrate import calibrated_proba, load
from src.utils.checks import record_check
from src.utils.config import Paths, load_config, now_iso
from src.utils.seed import set_seed

THRESHOLD = 0.5
QUIET_LOGGERS = ("shap_builder", "build_evidence", "extract_llm", "agent_queue",
                 "llm_ollama", "llm_openai_compat")
# short names users type / the config uses -> files in prompts/
PROMPT_VERSIONS = {"v0": "v0_zero_shot", "v0_zero_shot": "v0_zero_shot",
                   "v1": "v1_rules", "v1_rules": "v1_rules",
                   "v2": "v2_rules_fewshot", "v2_rules_fewshot": "v2_rules_fewshot"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def quiet(verbose: bool) -> None:
    """Keep the triage output clean: library log lines go to logs/ either way."""
    level = logging.INFO if verbose else logging.WARNING
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(level)


class InputError(Exception):
    """Bad user input (missing features, unknown sample, ...)."""


def _assets():
    """Load everything the single-sample path needs (once)."""
    cfg = load_config()
    features = json.loads((Paths.processed / "features.json").read_text())
    stats = json.loads((Paths.models / "train_stats.json").read_text())
    glossary = load_glossary()
    booster = load(Paths.models / "clf.joblib")
    calibrator = load(Paths.models / "calibrator.joblib")
    return cfg, features, stats, glossary, booster, calibrator


def family_from_category(category: str) -> str:
    """`Ransomware-Shade-<hash>-7.raw` -> `Shade`; benign rows -> `none`."""
    parts = str(category).split("-")
    if len(parts) >= 2 and parts[0].lower() != "benign":
        return parts[1]
    return "none"


def ground_truth_from_category(category: str) -> dict:
    """Ground truth for a raw CIC-MalMem row, from the `Category` column.

    The mirror labels each row `Benign` or `<MalwareClass>-<Family>-<hash>-<n>.raw`,
    e.g. `Ransomware-Shade-ab12...-7.raw`. The binary task is *ransomware vs the
    rest*, so Trojan/Spyware rows are negatives: remembering that matters in
    real triage (a negative verdict means "not ransomware", not "clean").
    """
    token = str(category).strip().split("-")[0]
    malware_class = "Benign" if token.lower() == "benign" else token
    if token.lower() == "benign":
        label = "benign"
    elif token.lower() == "ransomware":
        label = "ransomware"
    else:
        label = f"not ransomware ({malware_class.lower()})"
    return {"label": label, "malware_class": malware_class,
            "family": family_from_category(category), "category": str(category)}


def row_to_features(row: pd.Series, features: list[str], medians: dict | None,
                    allow_missing: bool) -> tuple[np.ndarray, list[str]]:
    """Pull the 55 model features out of a CSV row, in training order."""
    missing = [f for f in features if f not in row.index or pd.isna(row[f])]
    if missing and not allow_missing:
        raise InputError(
            f"{len(missing)} feature column(s) missing/empty: {', '.join(missing[:6])}"
            + (" ..." if len(missing) > 6 else "")
            + "\n  -> fix the CSV, or pass --allow-missing to fill them with the "
              "training median (the verdict is then less trustworthy).")
    values = []
    for f in features:
        if f in row.index and not pd.isna(row[f]):
            values.append(float(row[f]))
        else:
            values.append(float(medians[f]))
    return np.asarray(values, dtype=np.float32), missing


# --------------------------------------------------------------------------- #
# core: one sample -> verdict + evidence (+ explanation + faithfulness)
# --------------------------------------------------------------------------- #
def analyse(x_row: np.ndarray, sample_id: str, cfg, features, stats, glossary,
            booster, calibrator, top_k: int) -> dict:
    x = np.asarray(x_row, dtype=np.float32).reshape(1, -1)
    p = float(calibrated_proba(calibrator, x)[0])
    pred = int(p >= THRESHOLD)
    sv, base_values, impl = compute_shap(booster, x, features)
    sample = {"sample_id": sample_id, "pred": pred}
    ev = build_one(sample, x[0], sv[0], float(base_values[0]), p, top_k,
                   features, stats, glossary, threshold=THRESHOLD)
    ev["shap_implementation"] = impl
    return ev


def run_laya(ev: dict, score: dict | None, cfg, requested: str) -> dict | None:
    """Integration points 1+2 inline for the CLI (`--decide laya`).

    Advisory only: the returned block is attached to the record, never fed back
    into FS. Soft-fails to a single `available: false` note when Laya is absent.
    """
    if requested != "laya":
        return None
    from src.faithfulness import laya_gate

    out: dict = {"requested": "laya"}
    gate = laya_gate.gate_verdict(ev, cfg)
    out["gate"] = gate
    if not gate.get("available"):
        out["available"] = False
        out["reason"] = gate.get("reason")
        return out
    out["available"] = True
    out["claims"] = []
    if score is not None:
        j = laya_gate.judge_claims(score.get("per_claim") or [], ev, cfg)
        if j.get("available"):
            out["claims"] = [j]
        else:
            out["available"] = False
            out["reason"] = j.get("reason")
    return out


def explain(ev: dict, mode: str, want: str, cfg, backend, repair_rounds: int,
            temperature: float, prompt_version: str) -> tuple[dict, dict | None, list[dict]]:
    """Produce a rationale and score it. Returns (rationale, score, history)."""
    value_tol = float(cfg.faithfulness.value_tolerance)
    prob_tol = float(cfg.faithfulness.prob_tolerance)
    if want == "none":
        return {}, None, []
    if want == "template":
        rationale = template_rationale(ev)
        score = score_rationale(rationale, ev, value_tol, prob_tol)
        return rationale, score, [{"round": 0, "FS": score["FS"], "pass": score["FS_pass"]}]

    version = PROMPT_VERSIONS.get(str(prompt_version).strip().lower())
    if version is None:
        raise InputError(f"unknown prompt version {prompt_version!r}; use v0, v1 or v2 "
                         f"(v2 = strict rules + few-shot examples)")
    fewshot = load_fewshot()
    system = load_system(version)
    schema = (json.loads((Paths.schemas / "rationale.schema.json").read_text())
              if mode == "structured" else None)
    history: list[dict] = []
    feedback = None
    rationale: dict = {}
    score: dict = {}
    for round_no in range(repair_rounds + 1):
        user = build_user(ev, "S" if mode == "structured" else "P", version, fewshot,
                          feedback=feedback)
        res = backend.generate(system, user, temperature=temperature,
                               max_tokens=int(cfg.llm.max_tokens), seed=42,
                               json_schema=schema, kind="generate",
                               arm=f"cli-{mode}", sample_id=ev["sample_id"])
        if mode == "structured":
            rationale = parse_json_response(res.text)
            rationale.setdefault("summary", "")
        else:
            text = res.text.strip()
            claims, info = extract_claims_smart(text, ev, backend, allow_llm=True)
            verdict, prob = infer_verdict_and_probability(text)
            rationale = {"verdict": verdict, "probability_stated": prob, "claims": claims,
                         "summary": text, "extraction": info, "model": res.model,
                         "backend": res.backend}
        score = score_rationale(rationale, ev, value_tol, prob_tol)
        history.append({"round": round_no, "FS": score["FS"], "pass": score["FS_pass"]})
        if score["FS_pass"] or round_no == repair_rounds:
            break
        feedback = build_feedback(score, ev)
    return rationale, score, history


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_evidence_lines(ev: dict) -> list[str]:
    lines = []
    for f in ev["top_features"]:
        lines.append(
            f"  {f['rank']}. {f['name']:<32} value {fmt_value(f['value']):>8}  "
            f"{f['qualitative_level']:<9} {f['percentile_vs_train']:>5.1f}th pct  "
            f"{f['direction']:<18} SHAP {f['shap']:+.3f}")
    return lines


def laya_lines(laya: dict) -> list[str]:
    """Render the optional Laya decision-layer block (advisory, never FS)."""
    gate = laya.get("gate") or {}
    if not gate.get("available"):
        return ["   Laya decision layer   : unavailable ("
                f"{str(gate.get('reason', 'not installed'))[:90]})"]
    blocks = laya.get("claims") or []
    n_cmp = sum(b.get("n_compared", 0) for b in blocks)
    n_agree = sum(b.get("n_agreed", 0) for b in blocks)
    out = [f"   Laya decision layer   : verdict "
           f"{'agrees' if gate.get('agrees') else 'DISAGREES'} "
           f"({gate.get('laya_verdict')}, confidence "
           f"{_fmt_conf(gate.get('confidence'))})"
           + (f" · {n_agree}/{n_cmp} claims agree" if blocks else "")]
    if blocks:
        per_laya: dict[str, dict] = {}
        for b in blocks:
            for p in b.get("per_claim", []):
                per_laya.setdefault(p["feature"], p)
        out.append("   per-claim Laya cross-judge (advisory, never changes FS):")
        for feature, p in per_laya.items():
            out.append(f"     laya: {feature:<32} {str(p.get('laya_status')):<13} "
                       f"confidence {_fmt_conf(p.get('laya_confidence'))} "
                       f"({'agree' if p.get('agree') else 'DISAGREE'} with "
                       f"verifier `{p.get('verifier_status')}`)")
    return out


def _fmt_conf(v) -> str:
    return "n/a" if v is None else f"{float(v):.2f}"


def faith_summary_lines(faith: dict) -> list[str]:
    """The deterministic faithfulness block (nothing optional in here)."""
    lines = [
        f"   stated verdict matches classifier : "
        f"{'yes' if faith['verdict_ok'] else 'NO'}",
        f"   Faithfulness Score FS = {faith['FS']:.3f}  "
        f"({'PASS' if faith['FS_pass'] else 'FAIL'})   "
        f"claims {faith['n_claims']}: {faith['status_counts']['supported']} supported, "
        f"{faith['status_counts']['unsupported']} unsupported, "
        f"{faith['status_counts']['contradicted']} contradicted   "
        f"evidence coverage {100 * faith['evidence_coverage']:.0f}%",
    ]
    for c in faith["per_claim"]:
        mark = "[ok]" if c["status"] == "supported" else "[!!]"
        detail = ", ".join(c["failed_checks"]) or "all checks passed"
        lines.append(f"   {mark} {c['feature']:<32} {c['status']:<13} {detail}")
    return lines


def render_text(rec: dict, show_evidence: bool = True) -> str:
    out: list[str] = []
    ground = rec.get("ground_truth")
    bar = "=" * 78
    out.append(bar)
    out.append(f" SAMPLE {rec['sample_id']}   (source: {rec['source']})")
    out.append(" scope: this classifier answers 'is it ransomware?' "
               "(ransomware vs everything else, incl. other malware)")
    out.append(bar)
    if ground:
        extra = f"  (family: {ground['family']})" if ground["family"] != "none" else ""
        out.append(f" Dataset label      : {ground['label']}{extra}")
    out.append(f" Classifier verdict : "
               f"{'RANSOMWARE' if rec['predicted_label'] else 'benign / not ransomware'}"
               f"   p(ransomware) = {rec['probability_ransomware']:.4f}   "
               f"(threshold {rec['threshold']:.2f})")
    if ground:
        truth_positive = ground["label"] == "ransomware"
        out.append(" Classifier vs label: "
                   f"{'CORRECT' if (rec['predicted_label'] == 1) == truth_positive else 'WRONG'}"
                   "  (binary task)")
    if abs(rec["probability_ransomware"] - 0.5) < 0.10:
        out.append(" Note               : probability is near the threshold -> uncertain verdict")
    for w in rec.get("warnings", []):
        for i, chunk in enumerate(_wrap(w, 96)):
            out.append((" WARNING: " if i == 0 else "          ") + chunk)
    if show_evidence:
        out.append("")
        out.append(f" Evidence: top {len(rec['top_features'])} features by |SHAP| "
                   f"({rec['shap_implementation']}, covers "
                   f"{100 * rec['shap_mass_covered_by_top_k']:.0f}% of SHAP mass)")
        out.extend(render_evidence_lines(rec))
    exp = rec.get("explanation")
    if exp and exp.get("text"):
        out.append("")
        out.append(f" Explanation ({exp['mode']}"
                   + (f", model {exp.get('model')}" if exp.get("model") else "")
                   + "):")
        for chunk in _wrap(exp["text"], 96):
            out.append("  " + chunk)
    faith = rec.get("faithfulness")
    if faith:
        out.append("")
        out.append(" Faithfulness check of that explanation:")
        out.extend(faith_summary_lines(faith))
        if rec.get("laya"):
            out.extend(laya_lines(rec["laya"]))
        if faith.get("history") and len(faith["history"]) > 1:
            trail = " -> ".join(f"round {h['round']}: FS {h['FS']:.3f}"
                                for h in faith["history"])
            out.append(f"   repair loop: {trail}")
    if rec.get("laya") and not faith:
        out.append("")
        out.append(" Laya decision-layer gate (no LLM explanation was requested):")
        out.extend(laya_lines(rec["laya"]))
    if rec.get("hint"):
        out.append("")
        out.append(f" Hint: {rec['hint']}")
    out.append("")
    return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


# --------------------------------------------------------------------------- #
# input sources
# --------------------------------------------------------------------------- #
def truth_from_parquet(row: pd.Series) -> dict:
    malware_class = str(row.get("malware_class", "Benign"))
    label = "ransomware" if int(row.get("y_binary", 0)) == 1 else (
        "benign" if malware_class.lower() == "benign"
        else f"not ransomware ({malware_class.lower()})")
    return {"label": label, "malware_class": malware_class,
            "family": str(row.get("family", "none")),
            "category": str(row.get("Category", ""))}


def sample_from_parquet(sample_id: str, split: str, features: list[str],
                        medians, allow_missing: bool):
    df = pd.read_parquet(Paths.processed / f"{split}.parquet")
    key = sample_id if sample_id.startswith("row_") else "row_" + sample_id.replace("test_", "")
    hit = df[df["row_uid"] == key]
    if hit.empty and df["row_uid"].astype(str).str.endswith(key.replace("row_", "")).any():
        hit = df[df["row_uid"].astype(str).str.endswith(key.replace("row_", ""))]
    if hit.empty:
        near = ", ".join(df["row_uid"].head(3))
        raise InputError(f"sample {sample_id!r} not found in the {split} split. "
                         f"row_uid must look like row_003028 (e.g. {near}). "
                         f"Held-out eval ids are listed in data/eval/eval_ids.json.")
    row = hit.iloc[0]
    x, missing = row_to_features(row, features, medians, allow_missing)
    return x, truth_from_parquet(row), f"data/processed/{split}.parquet", missing


def sample_from_csv_row(path: Path, row_idx: int, features: list[str], medians,
                        allow_missing: bool):
    df = pd.read_csv(path)
    if row_idx >= len(df):
        raise InputError(f"--row {row_idx} but {path} has only {len(df)} data rows")
    row = df.iloc[row_idx]
    x, missing = row_to_features(row, features, medians, allow_missing)
    truth = ground_truth_from_category(row["Category"]) if "Category" in df.columns else None
    if truth and "Class" in df.columns:
        truth["class_column"] = str(row["Class"])
    return x, truth, f"{path} (row {row_idx})", missing


def sample_from_json_file(path: Path, features: list[str], medians, allow_missing: bool):
    obj = json.loads(path.read_text())
    if isinstance(obj, list):
        obj = obj[0]
    series = pd.Series(obj)
    x, missing = row_to_features(series, features, medians, allow_missing)
    truth = ground_truth_from_category(obj["Category"]) if "Category" in obj else None
    return x, truth, str(path), missing


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def _finish(rec: dict, args) -> int:
    if args.json:
        print(json.dumps(rec, indent=2))
    else:
        print(render_text(rec))
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(rec, indent=2))
    if args.fail_on_unfaithful and rec.get("faithfulness") and \
            not rec["faithfulness"]["FS_pass"]:
        return 1
    return 0


def task_warnings(truth: dict | None, rec: dict) -> list[str]:
    """Real-triage cautions: the binary task is ransomware-vs-rest, not malware-vs-clean."""
    out: list[str] = []
    if truth and truth.get("malware_class", "Benign").lower() not in ("benign", "ransomware"):
        out.append(f"the dataset labels this sample `{truth['malware_class']}` "
                   f"(family {truth.get('family', '?')}). The classifier only separates "
                   f"ransomware from everything else, so a benign verdict here means "
                   f"'not ransomware' - it does NOT mean the sample is clean.")
    if rec["predicted_label"] == 1 and abs(rec["probability_ransomware"] - 0.5) < 0.10:
        out.append("the probability is close to the decision threshold; treat this verdict "
                   "as low-confidence.")
    return out


def cmd_check(args) -> int:
    cfg, features, stats, glossary, booster, calibrator = _assets()
    set_seed(int(cfg.seed))
    medians = {f: float(stats["features"][f]["median"]) for f in features}
    top_k = args.top_k or int(cfg.evidence.top_k)

    if args.sample:
        x, truth, source, missing = sample_from_parquet(args.sample, args.split, features,
                                                        medians, args.allow_missing)
        sample_id = args.sample
    elif args.csv:
        x, truth, source, missing = sample_from_csv_row(Path(args.csv), args.row, features,
                                                        medians, args.allow_missing)
        sample_id = f"{Path(args.csv).stem}#row{args.row}"
    elif args.json_file:
        x, truth, source, missing = sample_from_json_file(Path(args.json_file), features,
                                                          medians, args.allow_missing)
        sample_id = f"{Path(args.json_file).stem}"
    else:
        raise InputError("give one of --sample <eval id>, --csv <file> [--row N], "
                         "or --json-file <file>")

    ev = analyse(x, sample_id, cfg, features, stats, glossary, booster, calibrator, top_k)

    backend = None
    if args.explain == "llm":
        backend = get_backend(cfg, force=args.backend)
        quiet(args.verbose)   # the backend module registers its logger on import
    try:
        rationale, score, history = explain(
            ev, args.mode, args.explain, cfg, backend, args.repair, args.temperature,
            args.prompt_version or str(cfg.llm.prompt_version))
    except WaitingForAgent:
        # Mirror `make generate`: publish the pending prompt into the active batch
        # file so queue/HOWTO.md instructions line up with what is on disk.
        if backend is not None and hasattr(backend, "emit_batch"):
            st = backend.emit_batch()
            print(f"queued {st['n_in_active_batch']} of {st['n_unanswered_total']} pending "
                  f"prompt(s) -> {st['requests_file']}")
        raise
    if score is not None:
        score["history"] = history
    laya_block = run_laya(ev, score, cfg, args.decide)

    rec: dict = {
        "sample_id": sample_id,
        "source": source,
        "ground_truth": truth,
        "verdict": ev["verdict"],
        "predicted_label": ev["predicted_label"],
        "probability_ransomware": ev["probability_ransomware"],
        "threshold": ev["threshold"],
        "top_features": ev["top_features"],
        "shap_mass_covered_by_top_k": ev["shap_mass_covered_by_top_k"],
        "shap_implementation": ev["shap_implementation"],
        "explanation": None,
        "faithfulness": None,
        "laya": laya_block,
        "generated_at": now_iso(),
    }
    rec["warnings"] = task_warnings(truth, rec)
    if missing:
        rec["hint"] = (f"{len(missing)} feature(s) were missing in the input and filled with "
                       f"the training median: {', '.join(missing[:5])}"
                       + (" ..." if len(missing) > 5 else ""))
    if score is not None:
        rec["explanation"] = {
            "mode": args.explain if args.explain != "llm" else f"llm:{args.mode}",
            "backend": getattr(backend, "name", "deterministic") if backend else "deterministic",
            "model": rationale.get("model"),
            "text": rationale.get("summary") or json.dumps(rationale, ensure_ascii=False),
            "rationale": rationale,
        }
        rec["faithfulness"] = score
    if args.explain != "llm" and not args.json:
        rec["hint"] = ("this explanation is the deterministic template and is faithful "
                       "by construction; run --explain llm (plus --repair 2) to have a "
                       "model write one and check whether it stays faithful.")
    detail = (f"{sample_id}: {rec['verdict']} p={rec['probability_ransomware']:.3f}, "
              f"{len(rec['top_features'])} evidence features"
              + (f", FS={score['FS']:.3f} ({rec['explanation']['mode']})" if score else ""))
    record_check("usage", "single-sample triage CLI ran", True, detail)
    return _finish(rec, args)


def cmd_batch(args) -> int:
    cfg, features, stats, glossary, booster, calibrator = _assets()
    set_seed(int(cfg.seed))
    medians = {f: float(stats["features"][f]["median"]) for f in features}
    top_k = args.top_k or int(cfg.evidence.top_k)
    df = pd.read_csv(args.csv)
    if args.stride and args.stride > 1:
        df = df.iloc[::args.stride]
    if args.limit:
        df = df.head(args.limit)
    rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        x, missing = row_to_features(row, features, medians, True)
        sid = f"{Path(args.csv).stem}#row{i}"
        ev = analyse(x, sid, cfg, features, stats, glossary, booster, calibrator, top_k)
        truth = ground_truth_from_category(row["Category"]) if "Category" in df.columns else None
        rec = {"sample_id": sid, "verdict": ev["verdict"], "predicted_label": ev["predicted_label"],
               "probability_ransomware": ev["probability_ransomware"]}
        if truth:
            rec["dataset_label"] = truth["label"]
            rec["dataset_malware_class"] = truth["malware_class"]
            rec["family"] = truth["family"]
            rec["correct"] = (truth["label"] == "ransomware") == (ev["predicted_label"] == 1)
        if args.explain == "template":
            rationale = template_rationale(ev)
            score = score_rationale(rationale, ev, float(cfg.faithfulness.value_tolerance),
                                    float(cfg.faithfulness.prob_tolerance))
            rec["FS"] = score["FS"]
            rec["FS_pass"] = score["FS_pass"]
            rec["n_claims"] = score["n_claims"]
        if args.decide == "laya":
            # Advisory columns only: an analyst can sort by "Laya disagreed"
            # without opening each row. FS above is untouched by this.
            laya = run_laya(ev, score if args.explain == "template" else None, cfg, "laya")
            gate = (laya or {}).get("gate") or {}
            rec["laya_available"] = bool(gate.get("available"))
            rec["laya_verdict_agree"] = gate.get("agrees") if gate.get("available") else None
            rec["laya_verdict"] = gate.get("laya_verdict") if gate.get("available") else None
            blocks = (laya or {}).get("claims") or []
            rec["laya_claim_agreement_rate"] = (
                sum(b["n_agreed"] for b in blocks) / max(sum(b["n_compared"] for b in blocks), 1)
                if blocks and any(b.get("n_compared") for b in blocks) else None)
        rec["top_features"] = ";".join(f["name"] for f in ev["top_features"])
        rec["missing_features"] = len(missing)
        rows.append(rec)
        print(f"[{i + 1}/{len(df)}] {sid}: {ev['verdict']} p={ev['probability_ransomware']:.3f}"
              + (f"  label={rec.get('dataset_label')}" if truth else ""), flush=True)

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    n_ransom = int((out["predicted_label"] == 1).sum())
    print(f"\nwrote {out_path}  ({len(out)} rows: {n_ransom} ransomware, "
          f"{len(out) - n_ransom} benign)")
    if "correct" in out.columns:
        print(f"accuracy vs dataset labels: {100 * out['correct'].mean():.1f}%")
    if args.explain == "template":
        print(f"faithfulness of the template rationale: mean FS {out['FS'].mean():.3f}, "
              f"pass rate {100 * out['FS_pass'].mean():.0f}%")
    if args.decide == "laya":
        if not out["laya_available"].any():
            print("laya decision layer: unavailable (run `make setup-laya`); "
                  "the advisory columns are empty")
        else:
            agree = out["laya_verdict_agree"].astype("boolean").fillna(False)
            print(f"laya decision layer: gate agreed with the classifier on "
                  f"{100 * agree.mean():.0f}% of {len(out)} rows")
            if "laya_claim_agreement_rate" in out.columns:
                rates = out["laya_claim_agreement_rate"].dropna()
                if len(rates):
                    print(f"                     claim cross-judge LAR "
                          f"{rates.mean():.3f} (advisory; FS above is unaffected)")
    record_check("usage", "batch triage CLI ran", True,
                 f"{len(out)} rows from {args.csv} -> {out_path}")
    return 0


def cmd_template(args) -> int:
    _, features, stats, _, _, _ = _assets()
    df = pd.read_parquet(Paths.processed / "test.parquet")
    example = df[df["y_binary"] == 0].iloc[0]
    row = {f: float(example[f]) for f in features}
    row["Category"] = "Benign"
    out = Path(args.out)
    pd.DataFrame([row], columns=features + ["Category"]).to_csv(out, index=False)
    print(f"wrote {out}  ({len(features)} features + Category, filled with one real benign row)")
    print("Edit the numbers, then run:")
    print(f"  python -m src.cli check --csv {out}")
    return 0


def cmd_info(args) -> int:
    cfg, features, stats, _, _, _ = _assets()
    metrics = json.loads((Paths.models / "metrics.json").read_text())
    n_eval = len(json.loads((Paths.eval / "eval_ids.json").read_text())["samples"])
    print("model      : " + str(Paths.models / "clf.joblib"))
    print(f"features   : {len(features)} (order in data/processed/features.json)")
    print(f"threshold  : {THRESHOLD}")
    print(f"test ROC-AUC / PR-AUC: {metrics['raw_test']['roc_auc']:.4f} / "
          f"{metrics['raw_test']['pr_auc']:.4f} (group-aware split)")
    print(f"eval samples available: {n_eval}  (data/eval/eval_ids.json)")
    print(f"config     : {cfg['_config_path']}")
    print(f"llm backend: {cfg.llm.backend} (set LLM_BACKEND in .env to change)")
    print("dirs       : data/raw, data/processed, outputs/{evidence,rationales,faithfulness}")
    print("\nexamples: python -m src.cli check --sample test_003028")
    print("          python -m src.cli batch --csv data/raw/CIC-MalMem-2022.csv --limit 50")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Real-life triage: check one memory sample, explain the verdict, "
                    "and verify the explanation.")
    ap.add_argument("--verbose", action="store_true", help="show library log lines")
    sub = ap.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="check ONE sample end to end")
    src = c.add_argument_group("input (choose one)")
    src.add_argument("--sample", help="held-out eval sample id, e.g. test_031534")
    src.add_argument("--csv", help="CSV whose row(s) hold the 55 features")
    src.add_argument("--json-file", help="JSON object/array with the features")
    c.add_argument("--row", type=int, default=0, help="row index into --csv (default 0)")
    c.add_argument("--split", default="test", choices=["test", "val", "train"],
                   help="split to look --sample up in (default test)")
    c.add_argument("--top-k", type=int, default=None, help="evidence features (default 6)")
    c.add_argument("--allow-missing", action="store_true",
                   help="fill missing features with the training median")
    c.add_argument("--explain", default="template", choices=["none", "template", "llm"],
                   help="template = deterministic offline rationale (default); "
                        "llm = ask the configured backend")
    c.add_argument("--mode", default="prose", choices=["prose", "structured"],
                   help="llm answer style (default prose)")
    c.add_argument("--prompt-version", default=None,
                   help="v0 zero-shot | v1 rules | v2 rules+few-shot (default from config)")
    c.add_argument("--backend", default=None,
                   help="agent_queue | openai_compat | ollama (default from .env)")
    c.add_argument("--repair", type=int, default=0, metavar="N",
                   help="verify-and-repair rounds for the llm explanation")
    c.add_argument("--temperature", type=float, default=0.2)
    c.add_argument("--decide", default="none", choices=["none", "laya"],
                   help="optional Laya decision layer: annotate the rationale with a "
                        "verdict gate + per-claim cross-judge (advisory only, needs "
                        "`make setup-laya`; silently skipped when absent)")
    c.add_argument("--json", action="store_true", help="machine-readable output (tool use)")
    c.add_argument("--save", help="write the full record JSON to this path")
    c.add_argument("--fail-on-unfaithful", action="store_true",
                   help="exit 1 when the explanation is not faithful")

    b = sub.add_parser("batch", help="triage many rows from one CSV")
    b.add_argument("--csv", required=True)
    b.add_argument("--limit", type=int, default=0, help="only the first N rows (0 = all)")
    b.add_argument("--stride", type=int, default=1,
                   help="take every K-th row (use the dataset's class ordering to mix)")
    b.add_argument("--out", default="outputs/triage_batch.csv")
    b.add_argument("--top-k", type=int, default=None)
    b.add_argument("--explain", default="none", choices=["none", "template"],
                   help="template also scores the deterministic rationale per row")
    b.add_argument("--decide", default="none", choices=["none", "laya"],
                   help="add optional Laya advisory columns: laya_verdict_agree, "
                        "laya_claim_agreement_rate (needs `make setup-laya`)")

    t = sub.add_parser("template", help="write an example input CSV you can edit")
    t.add_argument("--out", default="my_sample.csv")

    sub.add_parser("info", help="show model, features, paths and examples")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    quiet(args.verbose)
    handlers = {"check": cmd_check, "batch": cmd_batch, "template": cmd_template,
                "info": cmd_info}
    try:
        return handlers[args.command](args)
    except WaitingForAgent as e:
        print("WAITING_FOR_AGENT: the LLM backend queued this prompt for the coding agent.")
        print(f"  request id : {e.request_id}")
        print("  answer it in queue/responses.jsonl (see queue/HOWTO.md), then re-run "
              "the exact same command.")
        print("  no agent around? use --explain template, or LLM_BACKEND=ollama / "
              "openai_compat in .env.")
        return 20
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except LLMError as e:
        print(f"LLM error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"error: {e}\n  run `make setup download preprocess train evidence` first.",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
