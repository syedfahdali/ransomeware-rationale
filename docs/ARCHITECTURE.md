# Architecture

How the pipeline is wired, what each module owns, and where the interesting code
lives. For commands and workflows see [USAGE.md](USAGE.md); for the study results
see [../outputs/report.md](../outputs/report.md); for the optional decision layer
that sits *beside* the verifier and cannot change a score, see
[LAYA_REVIEW.md](LAYA_REVIEW.md).

## 0. The three ways in

The repository is one set of components with three entry points. They share the
model, the evidence builder and the verifier, so anything you learn from one applies
to the others — and no entry point can produce a number the others would dispute.

| entry point | audience | reads | writes | starts at |
|---|---|---|---|---|
| **the study** — `make all` | researchers | `data/raw`, `prompts/`, `schemas/` | every `outputs/` artifact incl. `report.md` | [src/generation/generate.py](../src/generation/generate.py) · [src/evaluation/run_arms.py](../src/evaluation/run_arms.py) |
| **everyday triage** — `python -m src.cli …` | analysts, tools, agents | the *trained* artifacts (`models/`, `data/processed/`, `outputs/evidence/`) | `outputs/triage_batch.csv`, a JSON record, exit codes | [src/cli.py](../src/cli.py) |
| **teaching / exploration** — `notebooks/` | students, reviewers | the same trained artifacts | `outputs/figs/notebook0*.png` | [notebooks/](../notebooks) |

Each entry point maps to concrete use cases (triaging a folder of dumps, gating CI
on explanation quality, teaching a forensics class, reproducing the study) — see
[Who this is for](../README.md#who-this-is-for-use-cases) and
[Usage examples](../README.md#usage-examples) in the README.

```text
                    ┌──────────────────────────────────────────────┐
   the study ──────►│ download → preprocess → train → evidence →   │
   (make all)       │ generate → verify → validity → eval → report │
                    └───────────────────┬──────────────────────────┘
                                        │ trained artifacts
                    ┌───────────────────┴──────────────────┐
   everyday triage ─┤ src/cli.py: check | batch | template ├─ code / agents / CI
   (one sample)     └───────────────────┬──────────────────┘
                                        │
   notebooks ─────── same calls, one cell at a time, figures written to outputs/figs/
```

```
┌──────────────┐   ┌────────────────┐   ┌───────────────┐   ┌──────────────────┐
│ data/raw     │──►│ preprocess     │──►│ train         │──►│ evidence         │
│ *.csv        │   │ dedupe+split   │   │ XGBoost +cal  │   │ TreeSHAP top-6   │
└──────────────┘   └────────────────┘   └───────────────┘   └────────┬─────────┘
                          │                     │                     │
                 data/processed/*.parquet  models/*.joblib   outputs/evidence/*.json
                                                                      │
                        ┌─────────────────────────────────────────────┘
                        ▼
              ┌────────────────────┐        ┌───────────────────────────┐
              │ generate (arms)    │───────►│ outputs/rationales/*.jsonl│
              │ T P0 P0R P2 S2 S2R │        └─────────────┬─────────────┘
              │ + X control        │                      │
              └─────────┬──────────┘                      │
                        │ prompts (queue or API)          │
                        ▼                                 ▼
              ┌────────────────────┐        ┌───────────────────────────┐
              │ src/llm/*          │        │ verify: extract → check   │
              │ cache + backends   │        │ → causal ablation         │
              └────────────────────┘        └─────────────┬─────────────┘
                                                          ▼
                                     outputs/faithfulness/{arm}.jsonl + summaries + stats.json
                                                          │            ← the ONLY source of FS
                        ┌─────────────────────────────────┤
                        ▼                                 ▼
        eval: stats · plots · report.md       (optional, off by default)
        validity: perturbations, gold set,    src/faithfulness/laya_gate.py
        fluency decoupling                    reads the same evidence JSON + rationales,
                        │                     writes outputs/laya/ — advisory only,
                        ▼                     proven unable to move FS
              outputs/report.md, outputs/figs/          (`laya_fs_independence`)

   assets: scripts/make_demo_gif.py ─► docs/demo*.gif      (Pillow, offline)
           scripts/make_brand_assets.py ─► docs/logo.*, docs/banner.*
```

## 1. Stage-by-stage data contracts

| stage | command | reads | writes |
|---|---|---|---|
| download | `make download` | UNB/CIC page, Kaggle, HuggingFace | `data/raw/CIC-MalMem-2022.csv`, `data/raw/download_info.json` |
| preprocess | `make preprocess` | `data/raw/*.csv` | `data/processed/{all,train,val,test}.parquet`, `data/processed/features.json`, `data/processed/splits.json`, `outputs/data_summary.md` |
| train | `make train` | `data/processed/*.parquet` | `models/{clf,calibrator}.joblib`, `models/{best_params,metrics,metrics_rowlevel,train_stats,feature_names}.json`, `outputs/classifier_report.md`, `outputs/figs/{test_roc,test_pr,reliability_test,confusion_test}.png` |
| evidence | `make evidence` | models + `data/eval/eval_ids.json` | `outputs/evidence/{sample_id}.json`, `outputs/evidence/index.json`, `outputs/shap_eval.npz` |
| generate | `make generate` | evidence + `prompts/` + `schemas/rationale.schema.json` | `outputs/rationales/{arm}.jsonl`, `outputs/rationales/llm_metadata.json` (may exit 20) |
| verify | `make verify` | rationales + evidence + SHAP | `outputs/faithfulness/{arm}.jsonl`, `{arm}_summary.json`, `all_arms_summary.{json,csv}` |
| validity | `make validity` | `T`/`S2`/`X`/`P0`/`P2` scores + gold set | `outputs/validity.json`, `outputs/spot_check.md` |
| eval | `make eval` | scores + summaries + validity | `outputs/faithfulness/stats.json`, `outputs/figs/*`, `outputs/report.md` |
| laya_gate | `FORCE=1 make laya_gate` | `outputs/evidence/*.json`, `outputs/faithfulness/*.jsonl` | `outputs/laya/{arm}_gate.jsonl`, `{arm}_claims.jsonl`, `summary.json`, `summary_full_run.json`, `disagreements.md` |
| brand | `make brand` | `outputs/faithfulness/all_arms_summary.json` | `docs/logo.{svg,png}`, `docs/banner.{png,gif}` |
| checks | `make checks` | `outputs/phase_checks.json` | acceptance table on stdout |
| tests | `make test` | `tests/*.py` | exit status only |

All LLM traffic is keyed through `src/llm/cache.py` (`make_key` = sha256 of
backend+model+system+user+temperature+seed+schema) and stored under `cache/llm/`.

### Everyday entry point (not a pipeline stage)

`python -m src.cli` ([src/cli.py](../src/cli.py)) consumes the artifacts instead of
rebuilding them:

```
input vector  : data/processed/<split>.parquet + features.json   (or a CSV/JSON row)
probability   : models/clf.joblib + models/calibrator.joblib
percentiles   : models/train_stats.json + src/evidence/feature_glossary.yaml
evidence      : src/evidence/shap_builder.compute_shap -> build_evidence.build_one
explanation   : src/generation/prompts + src/generation/template_baseline
                (or an LLM backend from src/llm/)
verification  : src/faithfulness/extract_rules + extract_llm + verify.score_rationale
output        : text report, or one JSON object with --json
```

Same evidence JSON and the same verifier as the study, so a sample checked by hand
is scored with exactly the metric reported in `outputs/report.md`.

## 2. Module responsibilities

### `src/data/`
* **download.py** — 4-step chain (UNB page scrape → Kaggle CLI → HuggingFace →
  manual) with a validity gate: `rows ≥ 50 000`, `cols ≥ 50`, `Category`/`Class`
  present. Records the provenance in `download_info.json`.
* **preprocess.py** — cleans `Class`/`Category`, derives `y_binary` (ransomware vs
  rest), `y_3class`, `family`, `malware_class`, `capture_id` (hex hash parsed from
  the capture filename) and `group_id`, dedupes, writes parquet + `data_summary.md`.
* **split.py** — group-aware stratified 70/10/20 (`group_id` never crosses splits),
  then an explicit leak check (group overlap + identical rows across splits).

### `src/models/`
* **train.py** — model factory (`xgboost` | `lightgbm`), param space, budget-aware
  `RandomizedSearchCV`, early stopping, final refit, calibration, metrics, report.
* **calibrate.py** — `CalibratedClassifierCV` + `FrozenEstimator` (sigmoid),
  save/load helpers used by evidence and causal code.

### `src/evidence/`
* **shap_builder.py** — `compute_shap()` uses `shap.TreeExplainer`; if the package
  is unavailable it falls back to XGBoost's exact `pred_contribs`, and exposes
  `raw_margin()` + `additivity_error()`.
* **build_evidence.py** — one JSON per eval sample: verdict, calibrated
  probability, top-6 by |SHAP| with direction, value, train percentile,
  qualitative level, z-score, glossary description, plus
  `shap_mass_covered_by_top_k` and `all_feature_names`; validates each file
  against `schemas/evidence.schema.json` and records the additivity check.
* **feature_glossary.yaml** — 55 features: description + analyst aliases. It is
  both the semantic source shown to the LLM and the alias table used by the
  claim extractor.

### `src/generation/`
* **prompts.py** — `render_evidence_table()` renders the *only* facts the model may
  use (rounded values, level words, percentiles, directions, descriptions) plus
  uncertainty/mixed-evidence notes; `build_user()` assembles few-shots + evidence
  (+ repair feedback); `parse_json_response()` is tolerant of code fences.
* **generate.py** — the six arms:
  `run_template()` (deterministic), `run_llm_arm()` (P0/P2/S2, one record per
  sample), `run_repair_arm()` (S2R), `run_prose_repair_arm()` (P0R). Progress is
  append-only per sample (`done_ids`), so a paused run resumes exactly.
* **repair.py** — turns failed checks into deterministic feedback lines
  (“claim about `handles.nfile` says high but the evidence says typical”).
* **template_baseline.py** — faithful-by-construction rationale used as the sanity
  ceiling and as the base for perturbation tests.

### `src/faithfulness/`
* **extract_rules.py** — glossary-alias mention matching with longest-phrase
  resolution, clause-aware cue attachment (`(658.0, typical)` back-binding;
  “the very high number of X” forward-binding), forward-first number binding and
  sentence-splitting. Emits claim dicts tagged `extractor="rules"`.
* **extract_llm.py** — fallback for unparsed sentences; validates against
  `schemas/extraction.schema.json`, merges only features the rules missed.
* **verify.py** — `verify_claim()` (feature existence, direction, level, value,
  verdict/probability), duplicate/conflict handling, policy checks (remediation
  advice, malware-family speculation) and `score_rationale()` aggregates.
* **metrics.py** — composite FS, weight/α sensitivity, per-arm aggregation.
* **causal.py** — ablation suite: median-imputation and marginal-resampling
  comprehensiveness, sufficiency, random-feature and top-SHAP controls, plus the
  other-class variant; `summarize_causal()` per arm.
* **judge.py** — optional LLM-judge cross-check (off by default; requires a
  second model).
* **laya_gate.py** — optional non-generative decision-layer cross-check
  ([Laya](https://github.com/NandhaKishorM/laya)): a pre-generation verdict gate
  (`gate_verdict`), a claim-level cross-judge (`judge_claims`) and a repair
  reviewer (`review_repair`), plus the `make laya_gate` stage that writes
  `outputs/laya/`. Owns the codebase's only `laya` import; off by default
  (`laya.enabled: false`); **soft-fails** to `{"available": False}` everywhere so
  no stage depends on it; and cannot move `FS` (see
  [docs/LAYA_REVIEW.md](LAYA_REVIEW.md)).  **Does it improve the system? No —**
  measured on the full 7-arm × 50-sample run (`outputs/laya/summary_full_run.json`):
  the gate returns `unclear` on all 350 calls (agreement 1.000 by construction), the
  cross-judge recalls 6% of the corrupted arm's 83 contradictions (LAR 0.387 there vs
  1.000 on the faithful arms), and the median gate
  call is 4,649 ms against the proposal's ≤100 ms target. Worth keeping as an
  advisory disagreement file, not as a scorer — the per-question table is in the
  README's *Does it improve the system?* section.

### `src/evaluation/`
* **select_eval_set.py** — stratified TP/TN/FN/FP sampling from test predictions.
* **run_arms.py** — scoring driver for every arm incl. the `X` control, causal
  calls and acceptance-check records.
* **validity.py** — perturbation recall, corrupted-evidence control, extractor
  gold accuracy, fluency correlations, spot check.
* **stats.py** — bootstrap CIs, paired bootstrap + Wilcoxon, per-stratum tables.
* **plots.py** — matplotlib (Agg) figures.
* **report_md.py** — assembles `outputs/report.md` from every artifact; nothing is
  hard-coded, so the report always matches the run.

### `src/llm/`
`base.py` (backend interface, `LLMResult`, `WaitingForAgent`), `cache.py` (disk
cache), `agent_queue.py`, `openai_compat.py`, `ollama_backend.py`.

### `src/cli.py`
* **`check`** — one sample (dataset id, CSV row or JSON) → calibrated probability,
  top-k SHAP evidence, explanation (`none` | `template` | `llm`), faithfulness score;
  `--repair N` adds the verify-and-repair loop, `--json` emits the tool contract,
  `--decide laya` adds the optional advisory Laya block (`verdict gate + per-claim
  cross-judge`).
* **`batch`** — the same path over a CSV, one line per row, writes a triage table;
  `--decide laya` appends `laya_verdict_agree` / `laya_claim_agreement_rate`
  diagnostic columns.
* **`template` / `info`** — write an editable input CSV; print model/paths/examples.

The CLI deliberately re-uses the exact same three functions as the study —
`build_one()` (evidence), `score_rationale()` (metric) and `template_rationale()` —
so a triage verdict and a study verdict are the same computation, not a reimplementation.
That is what makes the everyday path safe to trust: `--explain template` output is
faithful by construction, and `--explain llm` output goes through the identical
verifier the report is built on.

Exit codes are part of the interface: `0` ok, `1` `--fail-on-unfaithful` and the
explanation failed, `2` bad input/usage, `20` waiting for the agent queue.

### `notebooks/`

Two notebooks executed by [scripts/run_notebooks.py](../scripts/run_notebooks.py),
which strips them of any Jupyter dependency: it walks the JSON, runs each code cell
in order in one shared namespace, and saves figures to `outputs/figs/`. `make
notebooks` is therefore a *test* of the documented API, not just a demo — if a
signature changes, the run fails.

| notebook | cells | reads | demonstrates |
|---|---|---|---|
| [01_train_the_classifier.ipynb](../notebooks/01_train_the_classifier.ipynb) | load → inspect a row → group-aware split → train in ~5 s → ROC/PR/confusion → calibration → SHAP importance | `data/processed/*.parquet` | the classifier half: why the split is grouped, why 0.9273 is the honest number, what `make train` adds |
| [02_check_a_sample_end_to_end.ipynb](../notebooks/02_check_a_sample_end_to_end.ipynb) | one sample → verdict → evidence table → template rationale → faithfulness check → three crafted lies caught → real-LLM hook | `models/`, `data/processed/`, `outputs/evidence/` | the explanation half: what `FS` measures, which checks fire on which lie, how to swap in a generator |

### `scripts/` (offline tooling, all pure-Pillow / stdlib)

| script | output | note |
|---|---|---|
| `install.sh` | `.venv/` | the only installer; pins nothing until `requirements.lock.txt` is written |
| `make_demo_gif.py` | `docs/demo.gif`, `docs/demo_check.gif` | renders a real `make all` run and a real CLI run |
| `make_brand_assets.py` | `docs/logo.{svg,png}`, `docs/banner.{png,gif}` | reads the real arm scores, writes an SVG by hand + PNG/GIF with Pillow |
| `make_laya_finetune_set.py` | `data/laya_finetune/` | builds the optional layer's fine-tuning sets and reports why they are (mostly) circular |
| `check_docs.py` | exit status | resolves every markdown link and backticked path; percent-decodes link targets so a path containing spaces (the Laya proposal at the repo root) still resolves |
| `run_notebooks.py` | `outputs/figs/notebook0*.png` | executes every notebook cell |

## 3. Core code

**0 — the single-sample path** (`src/cli.py`; mirrors what the study stages do)

```python
def analyse(x_row, sample_id, cfg, features, stats, glossary, booster, calibrator, top_k):
    x = np.asarray(x_row, dtype=np.float32).reshape(1, -1)
    p = float(calibrated_proba(calibrator, x)[0])          # same calibrated probability
    pred = int(p >= THRESHOLD)
    sv, base_values, impl = compute_shap(booster, x, features)
    ev = build_one({"sample_id": sample_id, "pred": pred}, x[0], sv[0],
                   float(base_values[0]), p, top_k, features, stats, glossary)
    ev["shap_implementation"] = impl
    return ev                                             # then explain() + score_rationale()
```

**1 — the model factory and search space** (`src/models/train.py`)

```python
def make_model(kind: str, params: dict | None = None, n_estimators: int = 300, seed: int = 42):
    if kind == "xgboost":
        from xgboost import XGBClassifier
        base = dict(tree_method="hist", n_jobs=4, random_state=seed, n_estimators=n_estimators,
                    eval_metric="logloss")
        base.update(params or {})
        return XGBClassifier(**base)
    from lightgbm import LGBMClassifier
    ...
```

**2 — early stopping then refit on the full split**

```python
model = make_model(kind, best, n_estimators=1500, seed=int(cfg.seed))
model.set_params(early_stopping_rounds=50, eval_metric="auc")
model.fit(X_tr[fit_idx], y_tr[fit_idx], eval_set=[(X_tr[es_idx], y_tr[es_idx])], verbose=False)
best_iter = int(getattr(model, "best_iteration", 0) or 0) + 1     # -> 191 trees
booster = make_model(kind, best, n_estimators=best_iter, seed=int(cfg.seed))
booster.fit(X_tr, y_tr)
calibrator = fit_calibrator(booster, X_va, y_va, method="sigmoid")
```

**3 — evidence the LLM is allowed to see** (`src/generation/prompts.py`)

```python
lines.append("RANKED EVIDENCE (most influential first). You may only mention these features:")
for f in ev["top_features"]:
    lines.append(f"{f['rank']}. {f['name']} -- value {fmt_value(f['value'])} | "
                 f"level {f['qualitative_level']} | {f['percentile_vs_train']:.1f}th percentile of "
                 f"training data | direction {f['direction']} | {f['description']}")
```

**4 — deterministic claim verification** (`src/faithfulness/verify.py`)

```python
if name not in evidence_by_name:
    res.status = "out_of_topk" if name in all_lower else "unsupported"
    res.failed_checks = ["feature_exists"]
else:
    if claim_direction and ev_feature["direction"] != claim_direction:
        res.failed_checks.append("direction_ok")
    if claimed_level and normalize_level(claimed_level) != ev_feature["qualitative_level"]:
        res.failed_checks.append("level_ok")
    if value_stated is not None and not within(value_stated, ev_feature["value"], tol):
        res.failed_checks.append("value_ok")
    res.status = "supported" if not res.failed_checks else "contradicted"
```

**5 — composite score** (`src/faithfulness/metrics.py`)

```python
FS = verdict_ok * claim_precision**alpha * ((1 - w) + w * evidence_coverage)   # alpha=0.5, w=0.5
```

**6 — the verify-and-repair loop** (`src/generation/generate.py`)

```python
score = score_rationale(rationale, ev, ...)
fs_before = score["FS"]
rounds_used = 0
while not score["FS_pass"] and rounds_used < rounds:
    feedback = build_feedback(score, ev)              # deterministic text from failed checks
    res = backend.generate(system, build_user(ev, mode, version, fewshot, feedback=feedback), ...)
    rounds_used += 1
    rationale = parse_structured(res.text, schema)
    score = score_rationale(rationale, ev, ...)
```

**7 — the agent-queue protocol** (`src/llm/agent_queue.py`)

```python
rid = key[:16]
if rid in self._load_responses():        # answered -> continue
    return self._result_from_response(rid, ...)
self._ensure_archived(rid, system, user, ...)   # append to queue/requests_archive.jsonl
raise WaitingForAgent(rid, kind=kind)           # caller batches and exits with code 20
```

## 4. Data contracts (JSON Schemas)

| schema | used by | key fields |
|---|---|---|
| [schemas/evidence.schema.json](../schemas/evidence.schema.json) | evidence builder → prompts | `sample_id, verdict, probability_ransomware, threshold, base_value, top_features[] (rank,name,value,shap,direction,percentile_vs_train,train_median,z_score,description,qualitative_level), shap_mass_covered_by_top_k, all_feature_names` |
| [schemas/rationale.schema.json](../schemas/rationale.schema.json) | structured arms (S2/S2R) | `verdict, probability_stated, claims[], summary` |
| [schemas/extraction.schema.json](../schemas/extraction.schema.json) | prose fallback extractor | `claims[]` |

## 5. Extension points

* **Another classifier** — add a branch in `make_model()` and a param space in
  `param_space()`; the rest of the pipeline is model-agnostic (SHAP is computed
  from whatever booster is saved).
* **Another explanation method** — swap `compute_shap()`; keep the
  `dense → (values, base, margin)` contract and the additivity test.
* **Another dataset** — implement a loader like `src/data/ember_loader.py`, emit
  the same parquet + `features.json` contract (a `row_uid` column and
  `y_binary` label are required).
* **Another LLM** — implement the four methods of `LLMBackend` in `src/llm/base.py`
  (`generate`, `estimate`, `unload`, `name`) or reuse `openai_compat`.
* **New arms** — add an entry to `LLM_ARMS` in `src/generation/generate.py`
  (version + mode) and to `ARMS` in `src/evaluation/run_arms.py`.
* **New CLI input source** — add a `sample_from_*` loader in `src/cli.py` that
  returns `(feature_vector, ground_truth_or_None, source_label, missing_features)`;
  everything downstream (evidence, explanation, verification, `--json`) is reused.
* **New notebook** — drop it in `notebooks/`; `make notebooks`
  ([scripts/run_notebooks.py](../scripts/run_notebooks.py)) picks it up and executes
  every code cell in order.
* **Another cross-check / decision layer** — add a module that *reads* the persisted
  artifacts (`outputs/evidence/*.json`, `outputs/faithfulness/*.jsonl`) and writes to
  its own subdirectory, exactly like `src/faithfulness/laya_gate.py`. Two rules make
  such a layer safe: (1) it must return a plain dict contract and soft-fail when its
  dependency is absent, and (2) it must never be an input to `score_rationale()` —
  `fs_independence()` in that module is the template for proving it can't be.
* **Another generated asset** — add a `scripts/make_*.py` that only reads `outputs/`,
  and attach it to the `brand` / `docs` targets so `make docs` keeps it reproducible.
* **A new report section** — every table in `outputs/report.md` is produced by
  `src/evaluation/report_md.py` from artifacts on disk; nothing is hard-coded, so a
  new section only needs a helper that reads what an earlier stage already wrote.
* **Harder faithfulness rules** — extend `verify_claim()` checks; the report,
  repair feedback and perturbations all pick the new check up automatically.
