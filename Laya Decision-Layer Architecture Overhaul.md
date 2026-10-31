# Laya Decision-Layer Architecture Overhaul

Oct 11, 2026 · @S Fahd Ali

## Summary

Add Laya (`convaiinnovations/laya`) as a second, non-generative decision layer that runs **beside** the existing deterministic verifier, never in place of it. Laya never produces free text — it only answers closed, typed questions (`choice` / `score` / `noul`) with calibrated probabilities in a single \~33 ms forward pass, so there is nothing for it to hallucinate. `src/faithfulness/verify.py` stays the source of truth for FS. Laya adds three things the current architecture cannot do on its own:

1. A sub-40 ms **pre-generation gate** that flags a suspicious classifier verdict before any LLM tokens are spent explaining it.
2. An independent **claim-level cross-judge** that checks each extracted claim against the evidence from a model that shares none of the rules extractor's blind spots.
3. A **repair-loop reviewer** that confirms a repaired rationale actually fixed the flagged claim, not just reworded it.

All three are diagnostic and advisory. If Laya disagrees with the rule-based verifier, that disagreement is logged as a new metric — it never overrides FS.

## What Laya actually is

Laya is a **multilingual, non-autoregressive System-1 decision model**: give it a state (text/JSON) and typed questions, it returns typed answers with calibrated probabilities in one forward pass. It is trained with RLCD (reinforcement learning against strictly proper scoring rules), so reporting honest probabilities is the only reward-maximising strategy. [Model card](https://huggingface.co/convaiinnovations/laya)

| checkpoint | backbone | params | context | best at |
| --- | --- | --- | --- | --- |
| `laya` | ModernBERT-large | 421M | 512 tok | English text, guardrails, email triage |
| `laya-multilingual` | mmBERT-base | 322M | 1024 (up to 8k) | 100+ languages, \~2.2x faster |
| `laya-typed-decisions` | ModernBERT-large (fine-tuned) | 421M | 1024 tok | typed-decisions workflows, 0.766 acc |

**Mechanics that matter for this repo:** every answer option is scored at its own `[MASK]` token and softmaxed over that question's options — the answer space is defined per request, so a new question schema (new claim types, new verdict labels) needs no retraining. ECE drops from 0.466 to 0.081 after per-(question-type, option-count) temperature fitting, which we would need to redo on our own calibration slice.

**Honest limits that bear directly on this integration:**

- Base checkpoints are **near chance zero-shot** on typed-decision-style tasks (0.362 acc vs a 0.461 majority-class baseline) — the 0.766 figure belongs only to the checkpoint fine-tuned on that benchmark's own training split. Laya is a fast base to specialise, **not** a zero-shot decision engine. This is why §9 below proposes fine-tuning on our own gold data rather than deploying it zero-shot.
- `noul` (yes/no) can latch onto its own option labels instead of the state and return a confident "no" on clearly positive input. We avoid `noul` for anything load-bearing and use two-option `choice` instead (§5, §6).
- `action.act_probability` carries no usable signal; gate on `confidence` instead (AUROC 0.77 vs 0.30).
- English checkpoint collapses on non-Latin scripts while staying confident (Khmer: 0.000 accuracy at 0.952 confidence) — irrelevant here since our evidence text and prompts are English, but worth stating since the repo's rationale arms are English-only too.

## Where it plugs in

Three integration points, each answering a question the current pipeline cannot answer in milliseconds today:

| # | name | sits in | question Laya answers | relationship to existing code |
| --- | --- | --- | --- | --- |
| 1 | Pre-generation Verdict Gate | `src/cli.py` (`check`/`batch`) and `src/generation/generate.py`, before any LLM call | "Does the classifier's stated verdict match what a cheap independent reader concludes from the raw evidence text alone?" | **New.** Did not exist before; stops the pipeline from spending LLM tokens on a verdict that is already suspicious |
| 2 | Claim-Level Cross-Judge | `src/faithfulness/verify.py` / `score_rationale`, parallel to the rules extractor | "For each extracted claim, given only that feature's evidence row, do you independently agree it is supported / contradicted / out-of-topk?" | **Augments.** Runs alongside `verify_claim()`, is never allowed to overrule it, logged as a new `laya_agreement` diagnostic |
| 3 | Repair-Loop Reviewer | `src/generation/generate.py` (`run_repair_arm`, `run_prose_repair_arm`) | "Did this repair round actually fix the flagged claim, or just reword it?" | **Augments.** Gates whether a repair round is accepted before the next round's feedback is built |

All three reuse the same evidence JSON the rest of the pipeline already produces (`outputs/evidence/*.json`) — Laya needs no new data source, only a typed-question wrapper around it.

## Updated pipeline

&#91;embedded content: pipeline with the Laya decision layer · gate + cross-judge\]

This is the project's own `data/raw → preprocess → train → evidence → generate → verify → eval/validity` pipeline, unchanged — every existing stage, exactly as `ARCHITECTURE.md` draws it. Two nodes are added, nothing is replaced: the **Laya gate** sits right after `evidence`, before `generate` ever runs, and reads the same TreeSHAP table the LLM would; a mismatched verdict is flagged and skips the LLM call entirely (`Flag early`). The **Laya judge** sits beside the existing rules-based `verify` step, checking the same extracted claims independently; both feed the same `outputs/faithfulness/{arm}.jsonl` the study already produces, now carrying an extra `LAR` (Laya Agreement Rate) column. `train`, `XGBoost+cal`, and the rest of the classifier pipeline are untouched by this proposal — Laya only ever sits at the decision layer downstream of it.

## Integration point 1 — pre-generation triage gate

Before `generate.py` spends an LLM call explaining a verdict, ask Laya a single `choice` question over the same evidence table the LLM will see: does an independent reader of these six features land on the same verdict as the classifier? This costs \~33–40 ms on CPU, before any LLM round-trip.

```python
questions = {
    "verdict_sanity": {
        "type": "choice",
        "instructions": "Given this evidence, which verdict does it support?",
        "criteria": {
            "ransomware": "evidence leans toward ransomware",
            "benign": "evidence leans toward benign / not ransomware",
            "unclear": "evidence is mixed or weak either way",
        },
    }
}
result = router.predict(render_evidence_table(ev), questions, model="typed-decisions")
agrees = result["answers"]["verdict_sanity"]["choice"] in (ev["verdict"], "unclear")
```

- **Agrees or `unclear`** → proceed to generation as today.
- **Disagrees with high confidence** → log a `gate_mismatch` record (sample id, classifier verdict, Laya verdict, confidence), skip the costly LLM rationale by default, and surface it in `outputs/triage_batch.csv` and the CLI's warning line. `--force-generate` overrides this for the research run, since `make generate` must still cover every eval sample for the study to be comparable arm-to-arm.
- This never changes the classifier's verdict — it only decides whether it is worth generating prose about it right now. It reuses `render_evidence_table()` from `src/generation/prompts.py` unchanged, so the gate sees exactly what the LLM sees.

## Integration point 2 — claim-level cross-judge

After `extract_rules.py` (and, as a fallback, `extract_llm.py`) produce the claim list, `verify.py` already checks each claim's `feature_exists`, `direction_ok`, `level_ok` and `value_ok` deterministically. Laya runs **in parallel**, one `choice` question per claim, seeing only that feature's evidence row — never the full table — so it cannot "average out" a wrong claim against five right ones:

```python
questions = {
    "claim_status": {
        "type": "choice",
        "instructions": "Does the rationale's claim about this feature match the evidence row?",
        "criteria": {
            "supported": "direction, level and value all match the evidence",
            "contradicted": "direction or level or value conflicts with the evidence",
            "out_of_topk": "the feature isn't one of the ones listed as evidence",
        },
    }
}
for claim in rationale["claims"]:
    row = evidence_row_for(ev, claim["feature"])
    state = f"CLAIM: {claim['text']}\nEVIDENCE ROW: {row}"
    r = router.predict(state, questions, model="typed-decisions")
    laya_status = r["answers"]["claim_status"]["choice"]
```

**Agreement, not authority.** `laya_status` is written alongside `verify_claim()`'s `status` as a new `laya_status` / `laya_confidence` field on every claim record. It is never allowed to flip `status`, `failed_checks` or FS — the deterministic check is exact-match against numbers the model cannot "reinterpret," which is exactly what we want for a faithfulness metric. What the cross-judge buys us:

- A **disagreement rate** per arm: when rules-verifier and Laya disagree, those cases are exactly the ones worth a human or LLM-judge look (`judge.py` already exists for this but is "off by default" — route disagreements there automatically instead of sampling randomly).
- A cheap **recall check on the rules extractor itself**: if Laya marks a claim `contradicted` that the rules extractor never even attempted to parse (missed by regex/alias matching), that is a concrete lead on extractor gaps — complementary to the existing `extractor_gold.jsonl` comparison in `validity.py`.
- A second, independent number in `outputs/validity.json`: agreement-with-Laya alongside the existing perturbation recall and gold-set precision/recall, without touching how FS itself is computed.

## Integration point 3 — real-time CLI decision layer

`python -m src.cli check` already prints a faithfulness block with `--explain template` offline, no API key. Add a `--decide laya` mode (default-on when the `laya` package is installed, silently skipped otherwise) that runs integration points 1 and 2 inline, entirely on CPU, adding well under 100 ms total to a call that is already local:

```text
 Faithfulness check of that explanation:
   stated verdict matches classifier : yes
   Faithfulness Score FS = 1.000  (PASS)   claims 6: 6 supported, 0 unsupported, 0 contradicted
   Laya decision layer   : verdict agrees (0.91) · 6/6 claims agree · 0 disagreements
   [ok] handles.nmutant                  supported     laya: supported (0.88)
```

This also replaces the ad hoc role `judge.py` plays today (an optional, off-by-default LLM-judge that needs a second model and an API key): Laya is local, free, and fast enough to run on every `check`/`batch` call instead of a sampled subset, while `judge.py` is kept for the rarer case — a flagged disagreement — where a second *generative* opinion with reasoning is actually worth the extra latency and cost.

For `batch`, `outputs/triage_batch.csv` gains two columns: `laya_verdict_agree` and `laya_claim_agreement_rate`, so a analyst scanning a CSV of 30+ samples can sort by "Laya disagreed with the rules verifier" without opening each one.

## New module: `src/faithfulness/laya_gate.py`

Mirrors the existing module boundaries (`verify.py` owns deterministic checks, `judge.py` owns the optional LLM-judge) rather than reaching into either:

```python
# src/faithfulness/laya_gate.py
from laya import Router

_router = None  # lazy singleton; Router(preload=True) at process start for CLI/server use

def get_router(cfg):
    global _router
    if _router is None:
        _router = Router(preload=cfg.laya.preload, device=cfg.laya.device)
    return _router

def gate_verdict(ev: dict, cfg) -> dict:
    """Integration point 1. Returns {'agrees': bool, 'laya_verdict': str, 'confidence': float}."""
    ...

def judge_claims(rationale: dict, ev: dict, cfg) -> list[dict]:
    """Integration point 2. One record per claim: {'feature', 'laya_status', 'laya_confidence'}."""
    ...

def review_repair(before: dict, after: dict, claim: dict, ev: dict, cfg) -> bool:
    """Integration point 3. True if Laya agrees the repair round fixed `claim`."""
    ...
```

**Responsibilities:**

- Owns the only `laya` import in the codebase — same pattern as `src/llm/*` owning backend-specific imports, so the rest of the pipeline depends on this module's typed dict contracts, not on `laya` directly.
- Fails soft: if `laya` isn't installed or `Router()` can't load a checkpoint, every function returns `{'available': False}` and callers skip the Laya fields entirely — `make all` and the CLI must work identically with Laya absent, since it is a diagnostic addition, not a dependency of the faithfulness metric itself.
- Owns calibration: per-(question-type) temperature scaling fit once against `data/eval/extractor_gold.jsonl`-style labelled decisions (same procedure Laya's own `laya-typed-decisions` checkpoint uses), stored in `models/laya_temperatures.json`, loaded at `get_router()` time.
- No state crosses from `verify.py` into this module or back except through the plain dict contracts above — `score_rationale()` in `metrics.py` accepts the Laya fields as optional extra columns, never as inputs to FS (§8).

## Scoring: FS stays untouched, a new diagnostic sits beside it

The study's headline metric must stay reproducible and model-swappable (§10 of the README: swapping LLM backends "re-runs the study unchanged"). Laya is a cross-check, not an input to FS:

```latex
FS = verdict\_ok \cdot claim\_precision^{0.5} \cdot (0.5 + 0.5 \cdot evidence\_coverage)
```

stays exactly as defined in `src/faithfulness/metrics.py`. Alongside it, `score_rationale()` now also returns:

```latex
LAR = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}[status_i = laya\_status_i]
```

**Laya Agreement Rate (LAR)** — the fraction of claims where the deterministic check and the independent Laya judge land on the same status. LAR is reported per arm next to FS in `all_arms_summary.json`, not folded into it. Two reasons to keep it separate rather than blending it (e.g. `FS' = FS \cdot (0.5 + 0.5 \cdot LAR)`):

1. **Validity of the metric itself would become circular.** §7 of the current study validates FS against ground truth (perturbation recall, corrupted-evidence control, a hand-labelled gold set). If Laya's score fed into FS, Laya's own zero-shot weakness (§1) would silently degrade the metric the study is built to validate.
2. **LAR is more useful as a diagnostic than as a weight.** A low-FS, high-LAR sample means "both checkers agree this rationale is unfaithful" — high confidence, no action needed. A low-FS, low-LAR sample means "the checkers disagree" — exactly the case worth routing to `judge.py` or a human (§5). Blending the two into one number destroys that distinction.

The only place Laya output *gates* a decision outright is integration point 1 (whether to generate at all) — a pipeline/cost decision, not a change to how faithfulness is scored.

## Fine-tuning: zero-shot Laya is not good enough here

Laya's own benchmark is explicit: base checkpoints score **0.362 accuracy zero-shot** on typed-decision tasks, barely above a 0.318 random baseline and *below* the 0.461 per-question majority-class baseline. The 0.766 figure belongs only to `laya-typed-decisions`, fine-tuned on that benchmark's own training split. Deploying base Laya as a decision-layer gate without fine-tuning would add noise, not signal — so this overhaul treats fine-tuning as a required step, not an optional improvement.

**Data we already have for it:**

- `data/eval/extractor_gold.jsonl` — 20 agent-drafted, hand-checked rationale/claim/status triples (§7 of the README). Small, but exactly the right shape.
- Every verified claim across all 50 eval samples × 6 arms in `outputs/faithfulness/{arm}.jsonl` is itself a labelled example: `(claim text, evidence row) → status`, \~1,800 labelled decisions once T/P0/P0R/P2/S2/S2R are all scored. This is larger and comes for free from stages the pipeline already runs.
- `outputs/evidence/*.json` × the classifier's own verdict gives a second, even larger supervised set for the Integration-Point-1 gate task specifically: `(evidence table) → verdict`, one example per eval (and optionally train/val) sample.

**Process** (mirrors Laya's own [fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb), run on Kaggle's free 2×T4s or any single GPU):

1. Build a dataset script `scripts/make_laya_finetune_set.py` that reads `outputs/faithfulness/*.jsonl` + `outputs/evidence/*.json` and emits the two typed-decision sets above, split by the same capture-group-aware `group_id` as the classifier (no leakage between what the classifier was trained on, what the LLM rationales were generated from, and what Laya is fine-tuned on).
2. Fine-tune `laya` (English root checkpoint) on both tasks jointly, push to `convaiinnovations`-style private Hub repo or keep local under `models/laya_finetuned/`.
3. Fit per-(question-type, option-count) temperature scaling on the validation split — the same step that takes raw ECE 0.466 → 0.081 on the published checkpoints.
4. Re-run `validity.py`'s extractor-gold comparison with Laya substituted for the rules extractor on the same 20 gold rationales, to get a Laya precision/recall figure directly comparable to the existing rules (0.95/0.95 on rule-following prose) and LLM-extractor numbers already in the report.

Until step 2–3 are done, integration points 1–3 should run with **`laya-typed-decisions`** (the published fine-tune, already at 0.766 on a similar workflow) as the honest interim default, clearly labelled in `outputs/rationales/llm_metadata.json`-style metadata as "not fine-tuned on this repo's data" — same honesty norm the README already applies to the `agent_queue` generator.

## Pipeline wiring: new stage, config, RQ5

**New Makefile stage**, slotted between `verify` and `validity` (reads what `verify` wrote, feeds `validity` a new comparison):

| stage | command | reads | writes |
| --- | --- | --- | --- |
| `laya_gate` | `make laya_gate` | `outputs/evidence/*.json`, `outputs/rationales/*.jsonl`, `outputs/faithfulness/*.jsonl` | `outputs/laya/{arm}_gate.jsonl`, `outputs/laya/{arm}_claims.jsonl`, `outputs/laya/summary.json` |

`make validity` and `make eval` gain a dependency on it; `make all` runs it automatically; `FORCE=1 make laya_gate` reruns it alone like any other stage.

**New config block** in `configs/default.yaml`:

```yaml
laya:
  enabled: true            # soft-fails to false if `laya` isn't installed
  checkpoint: typed-decisions   # english | multilingual | typed-decisions | finetuned
  device: cpu               # cpu | cuda
  preload: true             # Router(preload=True) for CLI/server latency
  gate_confidence_threshold: 0.70   # below this, treat gate result as 'unclear'
  temperature_file: models/laya_temperatures.json
```

**New research question, RQ5** (alongside the existing RQ1–RQ4 in the README): *does a fast, non-generative cross-judge catch faithfulness failures the rules-based extractor misses, and how often do the two disagree on rule-following vs. free prose?* Answered by: LAR per arm (expect it near 1.0 for `T`/`S2`/`S2R`, lower for `P0` where the rules extractor already has known precision/recall gaps vs. the gold set), plus a `outputs/laya/disagreements.md` spot-check in the style of the existing `outputs/spot_check.md`.

**New acceptance checks**, extending `src/utils/checks.py` / `outputs/phase_checks.json`:

| check | pass condition |
| --- | --- |
| `laya_available_fallback` | pipeline completes identically with `laya.enabled: false` (no crash, no missing columns beyond the Laya-specific ones) |
| `laya_gate_latency` | median gate call ≤ 100 ms on CPU over the 50 eval samples |
| `laya_fs_independence` | FS computed with `laya.enabled: true` vs `false` is byte-identical — proves §8's separation held in code, not just in the formula |

## Honest limits and fallback behavior

- **Laya is not a hallucination detector by itself — it's a hallucination-*proof* second opinion.** It cannot hallucinate (no free text generation), but a *wrong* typed answer is still possible. Its value here is structural: it fails differently from both the LLM generator and the rules-based regex extractor, so agreement across all three is stronger evidence than any one of them alone. It does not replace `verify.py`'s exact-match checks, which remain the only component that is faithful *by construction*.
- **Zero-shot is unreliable (§1, §9).** Shipping this without the fine-tuning step would make LAR noisy and the pre-generation gate unreliable. The interim default (`laya-typed-decisions`) is a documented compromise, not a solution.
- **`noul` pitfall.** All three integration points use two-option `choice` instead of `noul`, per Laya's own documented issue with `noul` following its option labels rather than the state.
- **High-cardinality questions don't apply here.** Our evidence tables are top-6 features and our verdict/claim-status questions have 2–3 options — well inside Laya's reliable range; the documented Banking77-style failure mode (50+ options starving each option's token budget) is not a risk for this repo's schemas.
- **Fails soft, always.** If `laya` isn't installed, a checkpoint fails to load, or a call times out, every `laya_gate.py` function returns `{'available': False}`; the CLI prints nothing Laya-related, `make all` proceeds unchanged, and FS is computed exactly as it is today (§8's acceptance check `laya_fs_independence` enforces this in CI, not just in code review).
- **Added cost is small and bounded.** \~33–40 ms per call on a T4, 193–464 ms on CPU per the published benchmarks; `Router(preload=True)` keeps checkpoints resident so repeated calls in `batch` or the eval loop don't pay reload cost. This is additive wall-clock on top of the existing pipeline, not a new bottleneck — the LLM generation stage remains the dominant cost by 2–3 orders of magnitude.
- **Not a replacement for the causal ablation (RQ4).** Laya's agreement is a *claim*-level consistency check; it says nothing about whether the cited features actually move the classifier. `causal.py`'s median-imputation/marginal-resampling ablation stays the only causal evidence in the study.

## Rollout plan

Four phases, each independently mergeable and each leaving `make all` reproducible at every step:

| phase | scope | risk to existing study |
| --- | --- | --- |
| 0. Scaffolding | `pip install laya` as an optional extra; `src/faithfulness/laya_gate.py` with all functions returning `{'available': False}`; `laya.enabled: false` by default in config | none — dead code path until enabled |
| 1. CLI cross-judge only | Integration point 3 (`--decide laya` on `check`/`batch`); zero-shot `laya-typed-decisions` checkpoint | none — opt-in CLI flag, no pipeline stage touches `outputs/faithfulness/*` |
| 2. Pipeline stages | `make laya_gate` stage, integration points 1 and 2 wired into `generate.py`/`verify.py` as described above, LAR added to `all_arms_summary.json` and `report.md`, acceptance checks from §10 | low — additive columns/files only; `laya_fs_independence` check guards against FS drift |
| 3. Fine-tuned checkpoint | `scripts/make_laya_finetune_set.py`, fine-tuning run, `models/laya_finetuned/`, temperature refit, re-run of §9's gold-set comparison, flip config default from `typed-decisions` to `finetuned` | none to FS; changes only LAR and gate precision, both diagnostic |

Each phase ships with its own `make test` additions and a short note in `docs/DEVIATIONS.md` if anything behaves differently from what this document proposes — the existing project convention for honestly tracking departures from a spec.
