"""Rationale generation for all arms (spec sections 7, 9, 10).

Arms
----
T   template baseline (deterministic, no LLM)
P0  free prose, zero-shot prompt (v0)
P2  free prose, rules + few-shot prompt (v2)
S2  structured JSON, rules + few-shot prompt (v2)
S2R S2 + deterministic verify-and-repair loop (<= cfg.llm.repair_rounds rounds)
P0R P0 free prose + the same deterministic repair loop (extra arm, makes RQ2
    measurable: S2 already sits at the metric ceiling, so it never repairs)

agent_queue protocol: on a cache miss the backend raises WaitingForAgent; this
script emits the next batch of `batch_size` prompts to queue/requests.jsonl and
exits with code 20 (`WAITING_FOR_AGENT`). Re-run after answering.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from src.faithfulness.verify import score_rationale
from src.generation.prompts import build_user, load_fewshot, load_system, parse_json_response
from src.generation.repair import build_feedback
from src.generation.template_baseline import template_prose, template_rationale
from src.llm.base import LLMError, WaitingForAgent, get_backend
from src.utils.config import Paths, ensure_dirs, load_config, now_iso
from src.utils.logging import setup_logger
from src.utils.seed import set_seed

log = setup_logger("generate")

LLM_ARMS = {
    "P0": {"version": "v0_zero_shot", "mode": "P"},
    "P2": {"version": "v2_rules_fewshot", "mode": "P"},
    "S2": {"version": "v2_rules_fewshot", "mode": "S"},
    "P0R": {"version": "v2_rules_fewshot", "mode": "P"},
}


def append_jsonl(path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def done_ids(path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["sample_id"])
            except Exception:  # noqa: BLE001
                continue
    return ids


def load_evidence() -> dict[str, dict]:
    index = json.loads((Paths.evidence / "index.json").read_text())
    return {sid: json.loads((Paths.evidence / f"{sid}.json").read_text())
            for sid in index["sample_ids"]}


def waiting_exit(backend, arm: str) -> None:
    status = backend.emit_batch()
    print("\n" + "=" * 72)
    print(f"WAITING_FOR_AGENT — arm {arm}")
    print(f"unanswered requests: {status['n_unanswered_total']} "
          f"(active batch: {status['n_in_active_batch']})")
    print(f"requests -> {status['requests_file']}")
    print(f"answers  -> {status['responses_file']}   (append one JSON line per request)")
    print("Then re-run the exact same command.")
    print("=" * 72 + "\n")
    sys.exit(20)


# --------------------------------------------------------------------------- #
def run_template(samples, evidence, force: bool) -> None:
    out_path = Paths.rationales / "T.jsonl"
    done = set() if force else done_ids(out_path)
    for s in samples:
        sid = s["sample_id"]
        if sid in done:
            continue
        ev = evidence[sid]
        rationale = template_rationale(ev)
        append_jsonl(out_path, {
            "sample_id": sid, "arm": "T", "mode": "template", "prompt_version": "template",
            "backend": "deterministic", "model": "template-v1", "temperature": 0.0,
            "text": template_prose(rationale), "rationale": rationale,
            "latency_s": 0.0, "cache_hit": False, "prompt_tokens": 0, "completion_tokens": 0,
            "seed": 42, "ts": now_iso(),
        })
    log.info("arm T done (%d samples)", len(samples))


def spec_list(samples, evidence, arm: str, fewshot, temperature) -> list[dict]:
    cfg = LLM_ARMS[arm]
    specs = []
    for s in samples:
        ev = evidence[s["sample_id"]]
        system = load_system(cfg["version"])
        user = build_user(ev, cfg["mode"], cfg["version"], fewshot)
        specs.append({"system": system, "user": user, "temperature": temperature,
                      "seed": 42, "json_schema": None})
    return specs


def parse_structured(text: str, schema: dict) -> dict:
    obj = parse_json_response(text)
    import jsonschema
    jsonschema.validate(obj, schema)
    return obj


def run_llm_arm(arm: str, samples, evidence, backend, fewshot, cfg, force: bool,
                yes: bool, args) -> None:
    cfgm = LLM_ARMS[arm]
    out_path = Paths.rationales / f"{arm}.jsonl"
    done = set() if force else done_ids(out_path)
    todo = [s for s in samples if s["sample_id"] not in done]
    if not todo:
        log.info("arm %s already complete", arm)
        return
    temperature = float(args.temperature if args.temperature is not None else cfg.llm.temperature)
    schema = json.loads((Paths.schemas / "rationale.schema.json").read_text()) \
        if cfgm["mode"] == "S" else None
    system = load_system(cfgm["version"])

    est = backend.estimate(spec_list(todo, evidence, arm, fewshot, temperature))
    log.info("arm %s: estimated calls %s", arm, est)
    if est["new_calls"] > 1000 and not yes:
        log.error("estimated %d new calls (>1000); re-run with --yes to proceed", est["new_calls"])
        sys.exit(3)

    misses = 0
    batch = int(getattr(backend, "batch_size", 25))
    for s in todo:
        sid = s["sample_id"]
        ev = dict(evidence[sid])
        user = build_user(ev, cfgm["mode"], cfgm["version"], fewshot)
        parse_error = None
        try:
            res = backend.generate(system, user, temperature=temperature,
                                   max_tokens=int(cfg.llm.max_tokens), seed=42,
                                   json_schema=schema, kind="generate", arm=arm, sample_id=sid)
        except WaitingForAgent:
            misses += 1
            if misses >= batch:
                waiting_exit(backend, arm)
            continue
        except LLMError as e:
            log.error("arm %s sample %s: %s", arm, sid, e)
            raise

        rationale = None
        if cfgm["mode"] == "S":
            try:
                rationale = parse_structured(res.text, schema)
            except Exception as e:  # noqa: BLE001
                parse_error = str(e)[:300]
                log.warning("arm %s %s: JSON parse failed (%s); one repair retry", arm, sid, e)
                retry_user = user + "\n\nYour previous answer was not valid JSON. " \
                                    "Return ONLY the JSON object, no fences."
                try:
                    res = backend.generate(system, retry_user, temperature=temperature,
                                           max_tokens=int(cfg.llm.max_tokens), seed=42,
                                           json_schema=schema, kind="repair_parse",
                                           arm=arm, sample_id=sid)
                    rationale = parse_structured(res.text, schema)
                    parse_error = None
                except WaitingForAgent:
                    misses += 1
                    if misses >= batch:
                        waiting_exit(backend, arm)
                    continue
                except Exception as e2:  # noqa: BLE001
                    parse_error = f"{parse_error} | retry: {str(e2)[:200]}"

        append_jsonl(out_path, {
            "sample_id": sid, "arm": arm, "mode": cfgm["mode"],
            "prompt_version": cfgm["version"], "backend": res.backend, "model": res.model,
            "temperature": temperature, "text": res.text, "rationale": rationale,
            "parse_error": parse_error,
            "latency_s": res.latency_s, "cache_hit": res.cache_hit,
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "seed": 42, "ts": now_iso(),
        })
        log.info("arm %s %s: %d chars (cache_hit=%s)", arm, sid, len(res.text), res.cache_hit)
    if misses:
        waiting_exit(backend, arm)
    log.info("arm %s complete", arm)


# --------------------------------------------------------------------------- #
def run_repair_arm(samples, evidence, backend, fewshot, cfg, force: bool, yes: bool,
                   args) -> None:
    out_path = Paths.rationales / "S2R.jsonl"
    done = set() if force else done_ids(out_path)
    s2_path = Paths.rationales / "S2.jsonl"
    if not s2_path.exists():
        log.error("S2 rationales missing; run arm S2 first")
        sys.exit(3)
    s2 = {}
    with s2_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                s2[rec["sample_id"]] = rec
    todo = [s for s in samples if s["sample_id"] not in done]
    if not todo:
        log.info("arm S2R already complete")
        return

    version = LLM_ARMS["S2"]["version"]
    system = load_system(version)
    schema = json.loads((Paths.schemas / "rationale.schema.json").read_text())
    rounds = int(cfg.llm.repair_rounds)
    temperature = float(args.temperature if args.temperature is not None else cfg.llm.temperature)

    misses = 0
    batch = int(getattr(backend, "batch_size", 25))
    for s in todo:
        sid = s["sample_id"]
        ev = dict(evidence[sid])
        rec = s2.get(sid)
        if rec is None or rec.get("rationale") is None:
            log.warning("S2 rationale missing/unparseable for %s; skipping S2R", sid)
            append_jsonl(out_path, {"sample_id": sid, "arm": "S2R", "skipped": True,
                                    "reason": "no S2 rationale", "ts": now_iso()})
            continue
        rationale = rec["rationale"]
        score = score_rationale(rationale, ev, float(cfg.faithfulness.value_tolerance),
                                float(cfg.faithfulness.prob_tolerance))
        fs_before = score["FS"]
        history = [{"round": 0, "FS": fs_before, "FS_pass": score["FS_pass"]}]
        rounds_used = 0
        text = rec["text"]
        res_meta = {"latency_s": 0.0, "cache_hit": True, "prompt_tokens": rec.get("prompt_tokens"),
                    "completion_tokens": rec.get("completion_tokens")}
        parse_error = None
        waiting = False

        if not score["FS_pass"]:
            feedback = build_feedback(score, ev)
            for rnd in range(1, rounds + 1):
                user = build_user(ev, "S", version, fewshot, feedback=feedback)
                try:
                    res = backend.generate(system, user, temperature=temperature,
                                           max_tokens=int(cfg.llm.max_tokens), seed=42,
                                           json_schema=schema, kind="repair", arm="S2R",
                                           sample_id=sid)
                except WaitingForAgent:
                    # Never persist a "repaired" record while the repair call is
                    # still pending: it would mark the sample done un-repaired.
                    waiting = True
                    misses += 1
                    if misses >= batch:
                        waiting_exit(backend, "S2R")
                    break
                rounds_used = rnd
                text = res.text
                res_meta = {"latency_s": res.latency_s, "cache_hit": res.cache_hit,
                            "prompt_tokens": res.prompt_tokens,
                            "completion_tokens": res.completion_tokens}
                try:
                    rationale = parse_structured(res.text, schema)
                    parse_error = None
                except Exception as e:  # noqa: BLE001
                    parse_error = str(e)[:300]
                    history.append({"round": rnd, "FS": 0.0, "FS_pass": False,
                                    "parse_error": parse_error})
                    continue
                score = score_rationale(rationale, ev, float(cfg.faithfulness.value_tolerance),
                                        float(cfg.faithfulness.prob_tolerance))
                history.append({"round": rnd, "FS": score["FS"], "FS_pass": score["FS_pass"]})
                if score["FS_pass"]:
                    break
                feedback = build_feedback(score, ev)
        if waiting:
            continue

        append_jsonl(out_path, {
            "sample_id": sid, "arm": "S2R", "mode": "S", "prompt_version": version,
            "backend": backend.name, "model": backend.model, "temperature": temperature,
            "text": text, "rationale": rationale, "parse_error": parse_error,
            "rounds_used": rounds_used, "fs_before": fs_before, "fs_final": score["FS"],
            "fs_pass_final": score["FS_pass"], "history": history,
            "latency_s": res_meta["latency_s"], "cache_hit": res_meta["cache_hit"],
            "prompt_tokens": res_meta["prompt_tokens"],
            "completion_tokens": res_meta["completion_tokens"],
            "seed": 42, "ts": now_iso(),
        })
        log.info("arm S2R %s: rounds=%d FS %.3f -> %.3f pass=%s", sid, rounds_used,
                 fs_before, score["FS"], score["FS_pass"])
    if misses:
        waiting_exit(backend, "S2R")
    log.info("arm S2R complete")


# --------------------------------------------------------------------------- #
def _score_prose(text: str, ev: dict, backend, cfg, ctx: dict | None = None) -> tuple[dict, dict]:
    """Score free prose exactly as the evaluation stage does (rules + LLM fallback)."""
    from src.faithfulness.extract_llm import extract_claims_smart
    from src.faithfulness.extract_rules import infer_verdict_and_probability

    claims, info = extract_claims_smart(text, ev, backend, allow_llm=True, ctx=ctx)
    verdict, prob = infer_verdict_and_probability(text)
    rationale = {"verdict": verdict, "probability_stated": prob, "claims": claims,
                 "summary": text, "extraction": info}
    score = score_rationale(rationale, ev, float(cfg.faithfulness.value_tolerance),
                            float(cfg.faithfulness.prob_tolerance))
    return score, rationale


def run_prose_repair_arm(samples, evidence, backend, fewshot, cfg, force: bool, yes: bool,
                         args) -> None:
    """P0R: free prose + deterministic verify-and-repair (extra arm for RQ2)."""
    out_path = Paths.rationales / "P0R.jsonl"
    done = set() if force else done_ids(out_path)
    src_path = Paths.rationales / "P0.jsonl"
    if not src_path.exists():
        log.error("P0 rationales missing; run arm P0 first")
        sys.exit(3)
    src = {}
    with src_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                src[rec["sample_id"]] = rec
    todo = [s for s in samples if s["sample_id"] not in done]
    if not todo:
        log.info("arm P0R already complete")
        return

    version = LLM_ARMS["P0R"]["version"]
    system = load_system(version)
    rounds = int(cfg.llm.repair_rounds)
    temperature = float(args.temperature if args.temperature is not None else cfg.llm.temperature)
    extract_ctx = {"calls": 0, "cap": 200}

    misses = 0
    batch = int(getattr(backend, "batch_size", 25))
    for s in todo:
        sid = s["sample_id"]
        ev = dict(evidence[sid])
        rec = src.get(sid)
        if rec is None:
            continue
        text = rec.get("text") or ""
        try:
            score, rationale = _score_prose(text, ev, backend, cfg, extract_ctx)
        except WaitingForAgent:
            misses += 1
            if misses >= batch:
                waiting_exit(backend, "P0R")
            continue
        fs_before = score["FS"]
        history = [{"round": 0, "FS": fs_before, "FS_pass": score["FS_pass"]}]
        rounds_used = 0
        res_meta = {"latency_s": 0.0, "cache_hit": True, "prompt_tokens": None,
                    "completion_tokens": None}
        waiting = False

        if not score["FS_pass"]:
            feedback = build_feedback(score, ev)
            for rnd in range(1, rounds + 1):
                user = build_user(ev, "P", version, fewshot, feedback=feedback)
                try:
                    res = backend.generate(system, user, temperature=temperature,
                                           max_tokens=int(cfg.llm.max_tokens), seed=42,
                                           kind="repair", arm="P0R", sample_id=sid)
                except WaitingForAgent:
                    # Do not persist an un-repaired record (see run_repair_arm).
                    waiting = True
                    misses += 1
                    if misses >= batch:
                        waiting_exit(backend, "P0R")
                    break
                rounds_used = rnd
                text = res.text
                res_meta = {"latency_s": res.latency_s, "cache_hit": res.cache_hit,
                            "prompt_tokens": res.prompt_tokens,
                            "completion_tokens": res.completion_tokens}
                try:
                    score, rationale = _score_prose(text, ev, backend, cfg, extract_ctx)
                except WaitingForAgent:
                    waiting = True
                    misses += 1
                    if misses >= batch:
                        waiting_exit(backend, "P0R")
                    break
                history.append({"round": rnd, "FS": score["FS"], "FS_pass": score["FS_pass"]})
                if score["FS_pass"]:
                    break
                feedback = build_feedback(score, ev)
        if waiting:
            continue

        append_jsonl(out_path, {
            "sample_id": sid, "arm": "P0R", "mode": "P", "prompt_version": version,
            "backend": backend.name, "model": backend.model, "temperature": temperature,
            "text": text, "rationale": rationale, "parse_error": None,
            "rounds_used": rounds_used, "fs_before": fs_before, "fs_final": score["FS"],
            "fs_pass_final": score["FS_pass"], "history": history,
            "latency_s": res_meta["latency_s"], "cache_hit": res_meta["cache_hit"],
            "prompt_tokens": res_meta["prompt_tokens"],
            "completion_tokens": res_meta["completion_tokens"],
            "seed": 42, "ts": now_iso(),
        })
        log.info("arm P0R %s: rounds=%d FS %.3f -> %.3f pass=%s", sid, rounds_used,
                 fs_before, score["FS"], score["FS_pass"])
    if misses:
        waiting_exit(backend, "P0R")
    log.info("arm P0R complete")


# --------------------------------------------------------------------------- #
def write_metadata(cfg, backend, arms, temperature) -> None:
    meta = {
        "backend": backend.name,
        "generator": f"agent_queue:{backend.model}" if backend.name == "agent_queue" else backend.model,
        "model_id": backend.model,
        "temperature": temperature,
        "prompt_versions": {a: (LLM_ARMS.get(a) or {}).get("version", "template") for a in arms},
        "arms": arms,
        "repair_rounds": int(cfg.llm.repair_rounds),
        "generated_at": now_iso(),
        "note": ("agent_queue: the generator is the coding agent's underlying model, which may be "
                 "closed-weight; these are NOT open-weight results.") if backend.name == "agent_queue"
                else "remote OpenAI-compatible endpoint",
    }
    (Paths.rationales / "llm_metadata.json").write_text(json.dumps(meta, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="T,P0,P2,S2,S2R,P0R")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--yes", action="store_true", help="confirm >1000-call estimates")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config()
    set_seed(int(cfg.seed))
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    temperature = float(args.temperature if args.temperature is not None else cfg.llm.temperature)

    index = json.loads((Paths.evidence / "index.json").read_text())
    evidence = {sid: json.loads((Paths.evidence / f"{sid}.json").read_text())
                for sid in index["sample_ids"]}
    samples = [{"sample_id": sid} for sid in index["sample_ids"]]
    fewshot = load_fewshot()
    if any(a in ("P2", "S2", "S2R", "P0R") for a in arms) and len(fewshot) < 2:
        log.warning("prompts/fewshot.json missing or incomplete (%d examples); "
                    "v2 arms will run WITHOUT few-shot examples", len(fewshot))

    backend = get_backend(cfg)
    log.info("backend=%s model=%s samples=%d arms=%s", backend.name, backend.model,
             len(samples), arms)

    if "T" in arms:
        run_template(samples, evidence, args.force)
    for arm in ("P0", "P2", "S2"):
        if arm in arms:
            run_llm_arm(arm, samples, evidence, backend, fewshot, cfg, args.force, args.yes, args)
    if "S2R" in arms:
        run_repair_arm(samples, evidence, backend, fewshot, cfg, args.force, args.yes, args)
    if "P0R" in arms:
        run_prose_repair_arm(samples, evidence, backend, fewshot, cfg, args.force, args.yes, args)

    write_metadata(cfg, backend, arms, temperature)
    log.info("generation stage finished for arms %s", arms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
