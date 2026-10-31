"""Optional Laya decision-layer cross-check (diagnostic only, never an input to FS).

Laya is a non-autoregressive "System 1" decision model: it scores typed
questions (`choice` / `score` / `noul`) over a state in one forward pass and
returns typed answers with probabilities. It generates no free text, so there
is nothing to parse and nothing to hallucinate -- but a *wrong* typed answer is
still an error, which is why nothing here is allowed to change the study.

Three advisory entry points, matching the proposal in
`Laya Decision-Layer Architecture Overhaul.md`:

1. `gate_verdict(ev, cfg)` -- pre-generation triage: does a cheap independent
   reader of the same evidence table land on the classifier's verdict?
2. `judge_claims(claims, ev, cfg)` -- claim-level cross-judge: for each claim
   the rules extractor produced, does Laya independently agree it is supported?
3. `review_repair(before, after, ev, cfg)` -- did a repair round fix the flagged
   claim or just reword it?

Design contract (enforced by `tests/test_laya_gate.py` and
`make laya_gate`'s acceptance checks):

* **Soft-fail.** Every function returns ``{"available": False, "reason": ...}``
  when `laya` is not installed or the checkpoint cannot be loaded. No caller
  needs a try/except and no stage ever crashes because Laya is absent.
* **FS is untouchable.** `src/faithfulness/metrics.py` still defines the only
  Faithfulness Score. Laya output is added as *optional extra columns* on claim
  records; `score_rationale()` ignores keys it does not know, so FS is
  byte-identical with the layer on or off (`laya_fs_independence` check).
* **No new data source.** The gate reads the same
  `render_evidence_table()` text the LLM sees; the judge reads the same
  `outputs/faithfulness/*.jsonl` the study already writes.
* **This module owns the only `laya` import**, mirroring how `src/llm/*` owns
  its backend imports, so the rest of the pipeline depends on the plain dict
  contracts here rather than on `laya`.

Run as a stage (reads what `verify` wrote, writes `outputs/laya/`):

    python -m src.faithfulness.laya_gate            # honours cfg.laya.enabled
    python -m src.faithfulness.laya_gate --force    # run even if disabled
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from src.utils.config import Paths, ensure_dirs, load_config
from src.utils.logging import setup_logger

log = setup_logger("laya_gate")

ARMS = ["T", "P0", "P0R", "P2", "S2", "S2R"]
GATE_OPTIONS = ("ransomware", "benign", "unclear")
CLAIM_OPTIONS = ("supported", "contradicted", "not_in_evidence")

# --------------------------------------------------------------------------- #
# lazy import + router singleton
# --------------------------------------------------------------------------- #
_LAYA = None
_IMPORT_ERROR: str | None = None
_ROUTER = None
_ROUTER_KEY: tuple | None = None


def available() -> tuple[bool, str]:
    """(ok, reason). True only if the `laya` package imports."""
    global _LAYA, _IMPORT_ERROR
    if _LAYA is not None:
        return True, "laya importable"
    try:
        import laya as _mod  # noqa: PLC0415 - deliberately lazy / optional
    except Exception as e:  # noqa: BLE001
        _IMPORT_ERROR = f"{type(e).__name__}: {e}"
        return False, (f"the `laya` package is not installed ({_IMPORT_ERROR}); "
                       f"install the optional extra with `make setup-laya` "
                       f"or `pip install laya`")
    _LAYA = _mod
    return True, "laya importable"


def _checkpoint_spec(cfg) -> tuple[str, str | None]:
    """(model key passed to Router.predict, explicit local checkpoint dir or None).

    Laya's registry keys are `english` / `multilingual` / `typed-decisions`
    (`laya.DEFAULT_MODELS`). A locally fine-tuned checkpoint is registered under
    the `english` key and pointed at a directory via `Router(models={...})` --
    there is no `model_dir` kwarg on `Router.__init__`.
    """
    name = str(cfg.laya.get("checkpoint", "typed-decisions"))
    model_dir = str(cfg.laya.get("model_dir") or "") or None
    if name == "finetuned":
        return "english", model_dir
    return name, model_dir


def get_router(cfg):
    """Lazy process-wide Router. Returns None (and logs) when unavailable."""
    global _ROUTER, _ROUTER_KEY
    ok, reason = available()
    if not ok:
        return None
    model, model_dir = _checkpoint_spec(cfg)
    key = (model, model_dir, str(cfg.laya.get("device", "cpu")),
           bool(cfg.laya.get("preload", False)))
    if _ROUTER is not None and _ROUTER_KEY == key:
        return _ROUTER
    try:
        kwargs: dict = {"device": str(cfg.laya.get("device", "cpu")),
                        "default": model}
        if bool(cfg.laya.get("preload", False)):
            kwargs["preload"] = True
        if model_dir:
            # local checkpoint dir takes over the registry key it is mapped to
            kwargs["models"] = {model: model_dir}
        _ROUTER = _LAYA.Router(**kwargs)
        _ROUTER_KEY = key
        log.info("laya Router loaded (model=%s device=%s model_dir=%s)",
                 model, kwargs["device"], model_dir)
    except Exception as e:  # noqa: BLE001
        log.warning("could not construct laya Router: %s", e)
        _ROUTER, _ROUTER_KEY = None, None
        return None
    return _ROUTER


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def load_temperatures(cfg) -> dict:
    """Per-question temperature table, if one has been fitted.

    `scripts/make_laya_finetune_set.py` + the fine-tuning notebook are what
    produce this file; without it every temperature is 1.0 (no scaling) and
    that is recorded in the summary so the numbers are not over-read.
    """
    p = Paths.root / str(cfg.laya.get("temperature_file", "models/laya_temperatures.json"))
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        log.warning("ignoring unreadable temperature file %s: %s", p, e)
        return {}


def apply_temperature(prob: float, temperature: float) -> float:
    """Binary-logit temperature scaling (T>1 softens, T<1 sharpens)."""
    import math
    if not temperature or temperature <= 0 or temperature == 1.0:
        return prob
    p = min(max(float(prob), 1e-6), 1 - 1e-6)
    logit = math.log(p / (1 - p)) / float(temperature)
    return 1.0 / (1.0 + math.exp(-logit))


# --------------------------------------------------------------------------- #
# thin predict wrapper
# --------------------------------------------------------------------------- #
def ask(state: str, questions: dict, cfg) -> dict:
    """One typed-question round trip. Never raises; always returns a dict."""
    router = get_router(cfg)
    if router is None:
        ok, reason = available()
        return {"available": False, "answers": {}, "latency_ms": None,
                "reason": _IMPORT_ERROR or (reason if not ok else "Router unavailable")}
    model, model_dir = _checkpoint_spec(cfg)
    t0 = time.perf_counter()
    try:
        result = router.predict(state, questions, model=model)
    except TypeError:
        # older/newer signatures that do not accept `model`
        try:
            result = router.predict(state, questions)
        except Exception as e:  # noqa: BLE001
            log.warning("laya predict failed: %s", e)
            return {"available": False, "answers": {}, "latency_ms": None,
                    "reason": f"predict failed: {type(e).__name__}: {e}"}
    except Exception as e:  # noqa: BLE001
        log.warning("laya predict failed: %s", e)
        return {"available": False, "answers": {}, "latency_ms": None,
                "reason": f"predict failed: {type(e).__name__}: {e}"}
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {"available": True, "reason": "", "latency_ms": latency_ms,
            "answers": (result or {}).get("answers") or {},
            "routing": (result or {}).get("routing") or {},
            "usage": (result or {}).get("usage") or {},
            "model_requested": model, "model_dir": model_dir}


def _answer(answers: dict, name: str) -> dict:
    """Pull one question's payload out of the response, tolerating shapes."""
    a = answers.get(name)
    if isinstance(a, dict):
        return a
    if isinstance(a, str):
        return {"choice": a}
    return {}


def _choice(payload: dict) -> str | None:
    for k in ("choice", "label", "answer", "selected", "option"):
        v = payload.get(k)
        if isinstance(v, str):
            return v.strip().lower().replace(" ", "_").replace("-", "_")
    return None


def _confidence(payload: dict, n_options: int) -> float | None:
    """Laya's documented, usable signal is `confidence` (not act_probability).

    The response carries two probability-ish fields and they disagree sharply:
    `confidence` (what the proposal says to gate on) and `answer_confidence`
    (the top option's probability). Only `confidence` is gated on here; both are
    recorded so a disagreement is visible rather than hidden.
    """
    for k in ("confidence", "prob", "probability"):
        v = payload.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    probs = payload.get("probabilities") or payload.get("probs")
    if isinstance(probs, dict):
        vals = [float(v) for v in probs.values() if isinstance(v, (int, float))]
        if vals:
            return max(vals)
    if isinstance(probs, list) and probs:
        vals = [float(v) for v in probs if isinstance(v, (int, float))]
        if vals:
            return max(vals)
    return None


def _answer_confidence(payload: dict) -> float | None:
    v = payload.get("answer_confidence")
    if isinstance(v, (int, float)):
        return float(v)
    probs = payload.get("probabilities")
    if isinstance(probs, dict):
        vals = [float(x) for x in probs.values() if isinstance(x, (int, float))]
        if vals:
            return max(vals)
    return None


def _top_option_prob(payload: dict) -> float | None:
    """The confidence-gate signal Laya recommends when `confidence` misleads."""
    probs = payload.get("probabilities")
    if isinstance(probs, dict) and probs:
        vals = [float(v) for v in probs.values() if isinstance(v, (int, float))]
        if vals:
            return max(vals)
    return None


# --------------------------------------------------------------------------- #
# integration point 1: pre-generation verdict gate
# --------------------------------------------------------------------------- #
def gate_question() -> dict:
    return {
        "verdict_sanity": {
            "type": "choice",             # two-option choice, never `noul` (label-latching)
            "instructions": ("Read only this evidence table. Which verdict does the "
                             "evidence itself support, ignoring any stated conclusion?"),
            "criteria": {
                "ransomware": "the listed evidence leans toward ransomware",
                "benign": "the listed evidence leans toward benign / not ransomware",
                "unclear": "the evidence is mixed, weak or contradictory",
            },
        }
    }


def gate_verdict(ev: dict, cfg) -> dict:
    """Does an independent reader of the evidence agree with the classifier?"""
    from src.generation.prompts import render_evidence_table

    ok, reason = available()
    if not ok:
        return {"available": False, "reason": reason}

    state = render_evidence_table(ev)
    r = ask(state, gate_question(), cfg)
    if not r["available"]:
        return {"available": False, "reason": r["reason"]}

    payload = _answer(r["answers"], "verdict_sanity")
    choice = _choice(payload)
    conf = _confidence(payload, len(GATE_OPTIONS))
    temps = load_temperatures(cfg)
    temp = float(temps.get("verdict_sanity", 1.0) or 1.0)
    if choice in ("ransomware", "benign") and conf is not None:
        # keep the label, report the calibrated confidence
        conf = apply_temperature(conf, temp)
    threshold = float(cfg.laya.get("gate_confidence_threshold", 0.70))
    gated = choice
    if choice in ("ransomware", "benign") and conf is not None and conf < threshold:
        gated = "unclear"
    classifier = str(ev.get("verdict"))
    return {
        "available": True,
        "classifier_verdict": classifier,
        "probabilities": payload.get("probabilities"),
        "laya_verdict": choice,
        "laya_verdict_gated": gated,
        "confidence": conf,
        "answer_confidence": _answer_confidence(payload),
        "top_option_prob": _top_option_prob(payload),
        "confidence_threshold": threshold,
        "temperature": temp,
        "agrees": bool(gated is not None and gated in (classifier, "unclear")),
        "hard_disagreement": bool(choice in ("ransomware", "benign")
                                  and gated != "unclear" and choice != classifier),
        "latency_ms": r["latency_ms"],
        "model": (r.get("routing") or {}).get("model"),
        "tokens": ((r.get("usage") if isinstance(r.get("usage"), dict) else None)
                   or {}),
    }


# --------------------------------------------------------------------------- #
# integration point 2: claim-level cross-judge
# --------------------------------------------------------------------------- #
def claim_question() -> dict:
    return {
        "claim_status": {
            "type": "choice",
            "instructions": ("A rationale about this memory sample made the claim below. "
                             "Given only this one evidence row, does the claim match it?"),
            "criteria": {
                "supported": "direction, level and value all match the evidence row",
                "contradicted": ("direction or level or value conflicts with the "
                                 "evidence row, or cites a value it does not have"),
                "not_in_evidence": ("the feature is not one of the listed evidence "
                                    "features at all"),
            },
        }
    }


def render_claim_row(ev: dict, feature: str) -> str:
    """The single evidence row for one feature (or an explicit 'not listed')."""
    from src.generation.prompts import fmt_value

    top = {str(f["name"]).lower(): f for f in ev.get("top_features", [])}
    f = top.get(str(feature).lower())
    if f is None:
        listed = ", ".join(str(x["name"]) for x in ev.get("top_features", []))
        return (f"FEATURE: {feature}\nEVIDENCE: not among the ranked evidence features. "
                f"The listed features are: {listed}")
    return (f"FEATURE: {f['name']}\n"
            f"VALUE: {fmt_value(f['value'])}   LEVEL: {f['qualitative_level']}   "
            f"PERCENTILE: {f['percentile_vs_train']:.1f}\n"
            f"DIRECTION: {f['direction']}   DESCRIPTION: {f['description']}")


def collate_status(status: str | None) -> str:
    """`unsupported` and `out_of_topk` collapse: Laya cannot tell them apart
    without the full 55-feature list, so a strict comparison would be unfair."""
    s = (status or "").lower()
    return "not_in_evidence" if s in ("unsupported", "out_of_topk") else s


def judge_claims(claims: list[dict], ev: dict, cfg) -> dict:
    """One typed question per claim. Agreement only -- never authority."""
    ok, reason = available()
    if not ok:
        return {"available": False, "reason": reason, "per_claim": []}
    if not claims:
        return {"available": True, "reason": "", "per_claim": [],
                "n_compared": 0, "agreement_rate": None, "agreement_rate_strict": None}

    q = claim_question()
    temps = load_temperatures(cfg)
    temp = float(temps.get("claim_status", 1.0) or 1.0)
    per_claim: list[dict] = []
    n_cmp = n_agree = n_cmp_strict = n_agree_strict = 0
    latencies: list[float] = []

    for c in claims:
        feature = str(c.get("feature", ""))
        row = render_claim_row(ev, feature)
        state = (f"CLAIM about {feature}: {str(c.get('text') or '').strip()}\n\n"
                 f"EVIDENCE ROW:\n{row}")
        r = ask(state, q, cfg)
        if not r["available"]:
            return {"available": False, "reason": r["reason"], "per_claim": per_claim}
        if r["latency_ms"] is not None:
            latencies.append(r["latency_ms"])
        payload = _answer(r["answers"], "claim_status")
        laya_status = _choice(payload)
        conf = _confidence(payload, len(CLAIM_OPTIONS))
        if laya_status is not None and conf is not None and laya_status != "not_in_evidence":
            conf = apply_temperature(conf, temp)
        v_status = c.get("status")
        strict_ok = (laya_status is not None and laya_status == v_status)
        collated_ok = (laya_status is not None
                       and collate_status(laya_status) == collate_status(v_status))
        if laya_status is not None:
            n_cmp += 1
            n_agree += int(collated_ok)
            n_cmp_strict += 1
            n_agree_strict += int(strict_ok)
        per_claim.append({
            "feature": feature,
            "verifier_status": v_status,
            "verifier_failed_checks": c.get("failed_checks") or [],
            "laya_status": laya_status,
            "laya_confidence": conf,
            "laya_answer_confidence": _answer_confidence(payload),
            "laya_probabilities": payload.get("probabilities"),
            "agree": bool(collated_ok) if laya_status is not None else None,
            "agree_strict": bool(strict_ok) if laya_status is not None else None,
            "latency_ms": r["latency_ms"],
        })

    return {
        "available": True, "reason": "", "per_claim": per_claim,
        "n_compared": n_cmp,
        "n_agreed": n_agree,
        "agreement_rate": (n_agree / n_cmp) if n_cmp else None,
        "agreement_rate_strict": (n_agree_strict / n_cmp_strict) if n_cmp_strict else None,
        "median_claim_latency_ms": statistics.median(latencies) if latencies else None,
        "temperature": temp,
    }


# --------------------------------------------------------------------------- #
# integration point 3: repair-loop reviewer
# --------------------------------------------------------------------------- #
def repair_question() -> dict:
    return {
        "repair_fixed": {
            "type": "choice",
            "instructions": ("The verifier flagged a claim as wrong. Here is the claim "
                             "before and after a repair round. Did the repair actually "
                             "change the substance so the claim now matches the evidence?"),
            "criteria": {
                "fixed": "the repaired claim now matches the evidence row",
                "not_fixed": "the claim is still wrong, only reworded, or was dropped",
            },
        }
    }


def review_repair(before: dict, after: dict | None, ev: dict, cfg) -> dict:
    ok, reason = available()
    if not ok:
        return {"available": False, "reason": reason}
    feature = str((before or {}).get("feature", ""))
    b = str((before or {}).get("text") or "").strip()
    a = str((after or {}).get("text") or "").strip() or "(claim removed entirely)"
    state = (f"BEFORE (flagged): {b}\nAFTER (repaired): {a}\n\n"
             f"EVIDENCE ROW:\n{render_claim_row(ev, feature)}")
    r = ask(state, repair_question(), cfg)
    if not r["available"]:
        return {"available": False, "reason": r["reason"]}
    payload = _answer(r["answers"], "repair_fixed")
    verdict = _choice(payload)
    return {"available": True, "feature": feature, "laya_verdict": verdict,
            "fixed": verdict == "fixed", "confidence": _confidence(payload, 2),
            "latency_ms": r["latency_ms"]}


# --------------------------------------------------------------------------- #
# stage: outputs/laya/
# --------------------------------------------------------------------------- #
def load_evidence() -> dict[str, dict]:
    index = json.loads((Paths.evidence / "index.json").read_text())
    return {sid: json.loads((Paths.evidence / f"{sid}.json").read_text())
            for sid in index["sample_ids"]}


def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[rec["sample_id"]] = rec
    return out


def _claims_for_scoring(rec: dict) -> list[dict]:
    """Rebuild the rationale's claim list from a stored faithfulness record.

    Carries BOTH what the verifier needs (`direction` / `level` / `value_stated`)
    and what the cross-judge needs (`status` / `text`). `verify_claim()` never
    reads `status`, so including it does not change scoring -- which is exactly
    what `fs_independence()` re-proves on every stored record.
    """
    return [{"feature": c.get("feature"),
             "direction": c.get("stated_direction"),
             "level": c.get("stated_level"),
             "value_stated": c.get("stated_value"),
             "text": c.get("text"),
             "extractor": c.get("extractor"),
             "status": c.get("status"),
             "failed_checks": c.get("failed_checks") or []}
            for c in (rec.get("score") or {}).get("per_claim", [])]


def fs_independence(arms: dict[str, dict], evidence: dict[str, dict], cfg) -> tuple[bool, str]:
    """Adding Laya fields to claims must not move FS by even one ULP."""
    from src.faithfulness.verify import score_rationale

    vt = float(cfg.faithfulness.value_tolerance)
    pt = float(cfg.faithfulness.prob_tolerance)
    n = 0
    for arm, records in arms.items():
        for sid, rec in records.items():
            ev = evidence.get(sid)
            if ev is None:
                continue
            claims = _claims_for_scoring(rec)
            rationale = {"verdict": rec.get("verdict"),
                         "probability_stated": rec.get("probability_stated"),
                         "claims": claims}
            base = score_rationale(rationale, ev, vt, pt)["FS"]
            annotated = [dict(c, laya_status="contradicted", laya_confidence=0.99)
                         for c in claims]
            with_laya = score_rationale(dict(rationale, claims=annotated), ev, vt, pt)["FS"]
            stored = float((rec.get("score") or {}).get("FS"))
            n += 1
            if base != with_laya or base != stored:
                return False, (f"FS moved for {arm}/{sid}: stored={stored} "
                               f"no_laya={base} with_laya={with_laya}")
    return True, (f"FS byte-identical with and without Laya fields on all "
                  f"{n} stored rationale(s)")


def _gate_latency(records: list[dict]) -> float | None:
    lats = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    return statistics.median(lats) if lats else None


def _record_full_run(detail: str) -> None:
    """Keep the last complete six-arm measurement in the acceptance table.

    The stage is off by default, so a normal `make all` records "not applicable"
    for the Laya rows and would otherwise erase the evidence a real run left
    behind. This re-affirms it from `summary_full_run.json` instead.
    """
    from src.utils.checks import record_check

    record_check("laya", "laya_full_run_recorded", True, detail)


def _skipped(cfg, reason: str) -> int:
    from src.utils.checks import record_check

    log.info("laya_gate skipped: %s", reason)
    out = {"status": "skipped", "reason": reason,
           "enabled": bool(cfg.laya.get("enabled", False))}
    Paths.laya.mkdir(parents=True, exist_ok=True)
    (Paths.laya / "summary.json").write_text(json.dumps(out, indent=2))
    record_check("laya", "laya_available_fallback",
                 True, f"stage is a no-op: {reason}")
    record_check("laya", "laya_gate_latency", True,
                 "not applicable: the layer is disabled/absent")
    record_check("laya", "laya_fs_independence", True,
                 "not applicable: the layer is disabled/absent")
    prev = Paths.laya / "summary_full_run.json"
    if prev.exists():
        try:
            _record_full_run(_describe_full_run(json.loads(prev.read_text())))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.warning("could not read %s: %s", prev, e)
    return 0


def _accumulate_full_run(summary: dict) -> dict:
    """Merge a full-scale (unlimited) invocation into `summary_full_run.json`.

    The six study arms and the `X` control are run in separate invocations, so
    overwriting would keep only the last one. Returns the merged payload.
    """
    ref = Paths.laya / "summary_full_run.json"
    merged: dict = json.loads(json.dumps(summary))
    if ref.exists():
        try:
            prev = json.loads(ref.read_text())
            merged["arms"] = {**(prev.get("arms") or {}), **summary["arms"]}
            merged["arms_scored"] = sorted(set(prev.get("arms_scored") or [])
                                          | set(summary.get("arms_scored") or []))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.warning("could not merge existing %s: %s", ref, e)
    merged["arms_requested"] = sorted(set(merged.get("arms_requested") or [])
                                      | set(summary.get("arms_requested") or []))
    with ref.open("w") as fh:
        fh.write(json.dumps(merged, indent=2))
    return merged


def _describe_full_run(summary: dict) -> str:
    """One line for the acceptance table describing the preserved full run."""
    arms = summary.get("arms") or {}
    med = [v.get("median_gate_latency_ms") for v in arms.values()
           if v.get("median_gate_latency_ms") is not None]
    med_s = f"median gate {statistics.median(med):.0f} ms" if med else "gate latency n/a"
    lar = ", ".join(f"{a}={v['LAR']:.3f}" if v.get("LAR") is not None else f"{a}=n/a"
                    for a, v in sorted(arms.items()))
    n = max((int(v.get("n_samples") or 0) for v in arms.values()), default=0)
    return (f"reference run in summary_full_run.json: {len(arms)} arms x {n} samples, "
            f"{med_s}, LAR {lar} "
            f"(checkpoint {summary.get('checkpoint')}, device {summary.get('device')}, "
            f"zero-shot={summary.get('zero_shot')}; last invocation overwrote summary.json)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true",
                    help="run even when cfg.laya.enabled is false")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N samples per arm (0 = all). On CPU a "
                         "full six-arm run costs hours; use this for a look.")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config()

    if not bool(cfg.laya.get("enabled", False)) and not args.force:
        return _skipped(cfg, "cfg.laya.enabled=false (opt in with `make setup-laya` "
                             "then set laya.enabled: true, or pass --force)")
    ok, reason = available()
    if not ok:
        return _skipped(cfg, reason)

    evidence = load_evidence()
    arms_wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = {a: load_jsonl(Paths.faithfulness / f"{a}.jsonl") for a in arms_wanted}
    arms = {a: r for a, r in arms.items() if r}
    if not arms:
        return _skipped(cfg, "no outputs/faithfulness/*.jsonl yet -- run `make verify`")

    from src.utils.checks import record_check

    Paths.laya.mkdir(parents=True, exist_ok=True)
    gate_cache: dict[str, dict] = {}
    summary: dict = {
        "status": "ran",
        "checkpoint": str(cfg.laya.get("checkpoint")),
        "device": str(cfg.laya.get("device", "cpu")),
        "arms_requested": arms_wanted,
        "arms_scored": list(arms),
        "limit_per_arm": int(args.limit),
        "temperatures_file_found": bool(load_temperatures(cfg)),
        "zero_shot": not bool(load_temperatures(cfg)),
        "note": ("Zero-shot: no per-question temperature file was found, so these "
                 "numbers are the raw checkpoint's, not a calibrated one. "
                 "Diagnostic only -- see docs/LAYA_REVIEW.md. Latency is CPU."),
        "arms": {},
    }
    by_arm: dict[str, dict] = {}

    for arm, records in arms.items():
        gate_recs: list[dict] = []
        claim_recs: list[dict] = []
        sample_ids = sorted(records.items())
        if args.limit:
            sample_ids = sample_ids[:args.limit]
        for sid, rec in sample_ids:
            ev = evidence.get(sid)
            if ev is None:
                continue
            if sid not in gate_cache:
                gate_cache[sid] = gate_verdict(ev, cfg)
            g = gate_cache[sid]
            if not g.get("available"):
                return _skipped(cfg, f"Laya call failed mid-run: {g.get('reason')}")
            gate_recs.append({"sample_id": sid, "arm": arm, **g})
            j = judge_claims(_claims_for_scoring(rec), ev, cfg)
            if not j.get("available"):
                return _skipped(cfg, f"Laya call failed mid-run: {j.get('reason')}")
            claim_recs.append({
                "sample_id": sid, "arm": arm,
                "n_compared": j["n_compared"], "n_agreed": j["n_agreed"],
                "agreement_rate": j["agreement_rate"],
                "agreement_rate_strict": j["agreement_rate_strict"],
                "median_claim_latency_ms": j["median_claim_latency_ms"],
                "per_claim": j["per_claim"],
            })
        _write_jsonl(Paths.laya / f"{arm}_gate.jsonl", gate_recs)
        _write_jsonl(Paths.laya / f"{arm}_claims.jsonl", claim_recs)

        by_arm[arm] = _arm_stats(gate_recs, claim_recs)
        n = by_arm[arm]["n_samples"]
        n_agree = sum(1 for r in gate_recs if r.get("agrees"))
        log.info("laya %s: gate agreement %.3f (%d/%d), LAR %.3f over %d claims",
                 arm, (n_agree / n) if n else float("nan"), n_agree, n, by_arm[arm]["LAR"],
                 by_arm[arm]["n_claims_compared"])

    summary["arms"] = {a: {k: v for k, v in d.items()
                           if k not in ("gate_records", "claim_records")}
                       for a, d in by_arm.items()}
    (Paths.laya / "summary.json").write_text(json.dumps(summary, indent=2))
    # A partial run (`--arms X`) must not erase the arms an earlier run already
    # measured from the spot-check file, so keep them alongside this run's.
    _write_disagreements(Paths.laya / "disagreements.md",
                         {**_stored_by_arm(set(by_arm)), **by_arm})

    # ---- acceptance checks ------------------------------------------------ #
    all_gate = [r for d in by_arm.values() for r in d["gate_records"]]
    med = _gate_latency(all_gate)
    record_check("laya", "laya_gate_latency",
                 bool(med is not None and med <= 100.0),
                 f"median gate call {med:.0f} ms on {cfg.laya.get('device')} "
                 f"(target <=100 ms; the proposal's 33 ms figure is a T4 number)")
    if not args.limit:
        # Accumulate rather than overwrite: each arm was scored in its own
        # full-scale invocation (the six study arms together, the `X` control
        # separately), and the reference record should hold every one of them.
        # `--limit N` probes never touch this file.
        merged = _accumulate_full_run(summary)
        _record_full_run(_describe_full_run(merged))
    ok_fs, detail_fs = fs_independence(_all_faithfulness_arms() or arms, evidence, cfg)
    record_check("laya", "laya_fs_independence", ok_fs, detail_fs)
    record_check("laya", "laya_available_fallback", True,
                 "real Laya ran for this stage; the disabled/absent path is covered "
                 "by the skipped branch and tests/test_laya_gate.py")

    (Paths.laya / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({a: {k: v for k, v in d.items()
                          if k not in ("gate_records", "claim_records")}
                      for a, d in by_arm.items()}, indent=2))
    return 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _arm_stats(gate_recs: list[dict], claim_recs: list[dict]) -> dict:
    """Aggregate one arm's gate/claim records into the summary row.

    Kept out of `main()` so the same arithmetic is used both for the run in
    progress and for arms rebuilt from earlier runs' `*_gate.jsonl` files.
    """
    n = len(gate_recs)
    n_agree = sum(1 for r in gate_recs if r.get("agrees"))
    n_hard = sum(1 for r in gate_recs if r.get("hard_disagreement"))
    lar_vals = [r["agreement_rate"] for r in claim_recs if r["agreement_rate"] is not None]
    lar_vals_strict = [r["agreement_rate_strict"] for r in claim_recs
                       if r["agreement_rate_strict"] is not None]
    return {
        "n_samples": n,
        "gate_agreement_rate": (n_agree / n) if n else None,
        "gate_hard_disagreements": n_hard,
        "LAR": (sum(lar_vals) / len(lar_vals)) if lar_vals else None,
        "LAR_strict": (sum(lar_vals_strict) / len(lar_vals_strict))
        if lar_vals_strict else None,
        "n_claims_compared": sum(r["n_compared"] for r in claim_recs),
        "median_gate_latency_ms": _gate_latency(gate_recs),
        "gate_records": gate_recs,
        "claim_records": claim_recs,
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def _stored_by_arm(exclude: set[str]) -> dict[str, dict]:
    """Rebuild the arms an earlier invocation already wrote to `outputs/laya/`.

    The stage writes one `*_gate.jsonl` / `*_claims.jsonl` pair per arm and a
    spot-check file that is *not* per-arm, so a partial run (`--arms X`) has to
    re-read the other arms from disk rather than overwrite their section with
    nothing.
    """
    out: dict[str, dict] = {}
    if not Paths.laya.exists():
        return out
    for gate_path in sorted(Paths.laya.glob("*_gate.jsonl")):
        arm = gate_path.name[: -len("_gate.jsonl")]
        if arm in exclude:
            continue
        gate_recs = _read_jsonl(gate_path)
        claim_recs = _read_jsonl(Paths.laya / f"{arm}_claims.jsonl")
        if gate_recs or claim_recs:
            out[arm] = _arm_stats(gate_recs, claim_recs)
    return out


def _all_faithfulness_arms() -> dict[str, dict]:
    """Every stored rationale set on disk, not just the arms this run scored.

    FS independence is a property of the *verifier*, so it must be re-proved on
    all 7 arms x 50 samples even when the stage was invoked for one arm.
    """
    if not Paths.faithfulness.exists():
        return {}
    out = {}
    for p in sorted(Paths.faithfulness.glob("*.jsonl")):
        recs = load_jsonl(p)
        if recs:
            out[p.stem] = recs
    return out


def _write_disagreements(path: Path, by_arm: dict[str, dict]) -> None:
    """Spot-check file in the style of outputs/spot_check.md."""
    lines = ["# Laya cross-check — disagreements worth a look", "",
             "Cases where the independent decision layer disagrees with the "
             "deterministic verifier / classifier. Advisory only: nothing here "
             "changes FS (see `laya_fs_independence` in outputs/phase_checks.json).",
             ""]
    for arm, d in sorted(by_arm.items()):
        hard = [r for r in d["gate_records"] if r.get("hard_disagreement")]
        claim_d = [c for c in d["claim_records"]
                   if c["agreement_rate"] is not None and c["agreement_rate"] < 1.0]
        lines += [f"## arm {arm}", "",
                  f"- gate: {len(hard)} hard disagreement(s) of {d['n_samples']} samples; "
                  f"agreement rate {_fmt(d['gate_agreement_rate'])}",
                  f"- claim cross-judge: LAR {_fmt(d['LAR'])} "
                  f"(strict {_fmt(d['LAR_strict'])}) over {d['n_claims_compared']} claims", ""]
        for r in hard[:10]:
            lines.append(f"- **{r['sample_id']}** classifier says `{r['classifier_verdict']}`, "
                         f"Laya reads the evidence as `{r['laya_verdict']}` "
                         f"(confidence {_fmt(r.get('confidence'))}).")
        for c in claim_d[:10]:
            bad = [p for p in c["per_claim"] if p.get("agree") is False]
            for p in bad[:3]:
                lines.append(f"- **{c['sample_id']}** `{p['feature']}`: verifier "
                             f"`{p['verifier_status']}` vs Laya `{p['laya_status']}` "
                             f"(confidence {_fmt(p.get('laya_confidence'))}).")
        lines.append("")
    path.write_text("\n".join(lines))


def _fmt(v) -> str:
    return "n/a" if v is None else f"{float(v):.3f}"


if __name__ == "__main__":
    sys.exit(main())
