# SPEC: Faithfulness-Checked LLM Rationale Generation for Ransomware Verdicts

> **Audience:** an autonomous coding agent (Freebuff, OpenCode, or similar) with shell access to the user's machine.
> **Goal:** implement everything in this document end to end: dataset download, classifier, explanation evidence, LLM rationale generation, faithfulness scoring, evaluation on 50–100 held-out samples, and a final report.
> **Read the whole document before writing code. Follow the phases in order. Do not skip acceptance checks.**

---

## 0. Agent operating rules (read first)

1. **Work inside one project directory**: `~/ransomware-rationale/`. Create it if missing. Never write outside it, except `pip` installs into the project venv.
2. **Host machine (confirmed from user's Settings screen):** Ubuntu 26.04 LTS, Dell XPS 13 9370, Intel i7-8550U (4 cores / 8 threads), **8 GB RAM**, no discrete GPU, free disk space unknown.
   - Check free disk first: `df -h ~`. If less than 5 GB free, use only the small dataset (see §3).
   - **Do NOT run a large LLM locally.** The LLM layer must call a remote OpenAI-compatible API by default (see §6). A tiny local fallback is allowed (≤3B params, 4-bit).
   - Keep peak RAM under ~5 GB. Use `float32`, XGBoost `tree_method="hist"`, `n_jobs=4`.
3. **Python 3.11+** in a venv: `python3 -m venv .venv && source .venv/bin/activate`. Pin versions in `requirements.txt`.
4. **Everything must be reproducible with one command per stage** via `Makefile` targets (§14). Seeds fixed (`SEED=42`).
5. **No secrets in code.** API keys live in `.env` (gitignored). If a key is missing, stop the LLM stage and ask the user, or fall back to the `agent_queue` backend (§6.3).
6. **Log everything** to `logs/` and write all intermediate artifacts to `data/`, `models/`, `outputs/`. Cache every LLM call on disk so reruns cost nothing.
7. **After each phase run its acceptance checks** (listed per phase). If a check fails, fix it before moving on. At the end, print a summary table of which checks passed.
8. **If something in this spec is wrong or a URL is dead**, do not guess silently. Try the listed fallbacks, search the web for the current location, and record what you did in `docs/DEVIATIONS.md`.
9. Be honest in the report: if faithfulness numbers are poor, report them as they are.

---

## 1. Problem statement

Given:
- a trained tabular classifier (XGBoost or LightGBM) predicting **ransomware vs. benign** (optionally malware family), and
- per-sample feature evidence (feature values + SHAP attributions),

an LLM generates a **natural-language justification** of the verdict. We then build a **faithfulness metric** that checks whether the rationale's claims actually match the underlying evidence (feature identity, direction of effect, magnitude, verdict consistency, coverage), plus a **causal faithfulness test** (do the features the rationale cites really drive the model's prediction?). Fluency alone is explicitly not the target.

**Research questions**
- RQ1: How often do LLM rationales contain claims unsupported or contradicted by the evidence?
- RQ2: Does a deterministic verify-and-repair loop improve faithfulness without hurting readability?
- RQ3: Do fluency/overlap metrics fail to detect unfaithful rationales (showing why a dedicated metric is needed)?
- RQ4: Are the features cited in rationales causally important to the classifier (ablation test)?

---

## 2. High-level architecture

```
                 ┌──────────────────────┐
  download ───►  │  Raw dataset (CSV)   │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Preprocess + split   │  train / val / test(held-out) 
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │ XGBoost / LightGBM   │  + calibration, metrics
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Evidence builder     │  SHAP top-k, values, percentiles,
                 │ (structured JSON)    │  feature glossary, verdict, prob
                 └──────────┬───────────┘
                            ▼
        ┌───────────────────────────────────────────┐
        │ Rationale generator (LLM backend adapter) │
        │  - openai_compat (remote, default)        │
        │  - ollama (local small / ollama cloud)    │
        │  - agent_queue (coding agent answers)     │
        └──────────────────┬────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────┐
        │ Claim extraction (structured JSON claims) │
        └──────────────────┬────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────┐
        │ Faithfulness scorer                       │
        │  A. Deterministic claim verification      │
        │  B. Coverage / hallucination metrics      │
        │  C. Causal ablation test                  │
        │  D. (optional) LLM-judge cross-check      │
        └──────────────────┬────────────────────────┘
                           ▼
        ┌───────────────────────────────────────────┐
        │ Optional repair loop (feedback → regen)   │
        └──────────────────┬────────────────────────┘
                           ▼
              Evaluation on 50–100 held-out samples
              (baselines, ablations, CIs, report.md)
```

---

## 3. Dataset

### 3.1 Primary: CIC-MalMem-2022 (recommended for this hardware)

- Memory-dump-derived features, ~58k rows (balanced benign/malware), ~55 numeric features with **human-readable names** (e.g. `pslist.nproc`, `handles.nhandles`, `malfind.commitCharge`, `svcscan.nservices`, …). Names are meaningful, which makes rationales checkable.
- Labels: `Category` (`Benign` or `Malware-<Family>`), `Class` (`Benign`/`Malware`). The malware categories include **Ransomware** families (Conti, Maze, Pysa, Ako, Shade/Troldesh). Use the category string to derive:
  - `y_binary`: **1 = Ransomware**, 0 = everything else (benign + non-ransomware malware). Default task: *ransomware vs. rest*.
  - Optional secondary task: 3-class `benign / ransomware / other-malware`.
- Size is small (tens of MB), so it fits easily in 8 GB RAM.

**Automated download order (try each until one works; verify file exists and has >50k rows):**
1. Official UNB CIC page: `https://www.unb.ca/cic/datasets/malmem-2022.html`. Parse the page for the CSV/zip link (the file is typically named like `Obfuscated-MalMem2022.csv`). The link has historically pointed at a UNB-hosted directory; the agent must **discover the current URL by fetching the page**, not rely on memory.
2. Kaggle mirror via `kaggle datasets download` (requires `~/.kaggle/kaggle.json`; search Kaggle for "CIC-MalMem-2022"). Only try if credentials exist.
3. Hugging Face datasets mirror: search `huggingface.co/datasets` for "MalMem2022" or "CIC-MalMem" and use `huggingface_hub.snapshot_download`.
4. **Manual fallback:** print exact instructions asking the user to place the CSV at `data/raw/CIC-MalMem-2022.csv`, then continue automatically when the file appears.

### 3.2 Optional secondary: EMBER (only if ≥ 15 GB free disk and user requests it)

- EMBER 2018 (1.1M PE files, 2381 features) or EMBER2024 (Hugging Face / `thrember`). EMBER feature names are index-based/hashed and much less interpretable; it only has binary malware labels (no ransomware-specific label), so it is **not** the default.
- If implemented, put it behind `--dataset ember` and use the pre-vectorized `.dat`/parquet artifacts, **never** re-extract features from raw binaries (the machine is too weak). Subsample (e.g. 100k train rows) to fit memory. Map the 2381 feature indices to group names (general, header, section, imports, exports, byte histogram, byte-entropy histogram, strings, data directories) and build evidence at the **feature-group level** to keep rationales readable.

> Implement 3.1 fully. Implement 3.2 as a stub (`src/data/ember_loader.py` raising `NotImplementedError` with clear instructions) unless the user asks otherwise.

### 3.3 Splits

- Stratified 70 / 10 / 20 → `train / val / test` on `y_binary` (and family where applicable). Seed 42.
- **Group-aware split caution:** MalMem rows may be near-duplicates of the same capture. Drop exact duplicate rows before splitting, and note this in the report. If a family/sample-id column exists, split by group to avoid leakage.
- The **evaluation set** is drawn from `test`: sample **N=100** (configurable, min 50) **stratified**: 40 true-positive ransomware, 30 true-negative, 15 false-negative, 15 false-positive (use whatever exists if fewer; report the actual counts). Include errors on purpose, since faithful rationales for wrong verdicts are the hardest case.

---

## 4. Classifier

- Primary: **XGBoost** (`XGBClassifier`, `tree_method="hist"`, early stopping on val). Secondary: **LightGBM** behind `--model lightgbm`.
- Hyperparameter search: small `RandomizedSearchCV` or Optuna with ≤ 30 trials, 3-fold CV, metric = PR-AUC. Keep runtime under ~20 min on this CPU.
- Handle imbalance with `scale_pos_weight` if needed.
- **Calibrate** probabilities (`CalibratedClassifierCV` isotonic/sigmoid on val) and save both raw and calibrated models. The rationale uses the calibrated probability; SHAP uses the underlying booster.
- Report on test: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, Brier score, reliability curve plot → `outputs/classifier_report.md` + PNGs.
- Save: `models/clf.joblib`, `models/calibrator.joblib`, `models/feature_names.json`, `models/train_stats.json` (per-feature median, mean, std, quantiles 5/25/50/75/95 computed on **train only**).

**Acceptance (Phase 2):** test ROC-AUC ≥ 0.95 on the MalMem binary task (typical results are far higher; if suspiciously ~1.0, check duplicates/leakage and document it); model files load; `train_stats.json` has every feature.

---

## 5. Evidence builder (the ground truth the rationale must match)

For each sample produce `evidence.json`:

```json
{
  "sample_id": "test_000123",
  "verdict": "ransomware",
  "predicted_label": 1,
  "probability_ransomware": 0.973,
  "threshold": 0.5,
  "base_value": -0.12,
  "top_features": [
    {
      "rank": 1,
      "name": "handles.nhandles",
      "value": 18234.0,
      "shap": 1.84,
      "direction": "toward_ransomware",
      "percentile_vs_train": 97.3,
      "train_median": 5230.0,
      "z_score": 2.9,
      "description": "Total number of open handles across processes"
    }
  ],
  "shap_mass_covered_by_top_k": 0.81,
  "k": 6
}
```

Rules:
- Compute SHAP with `shap.TreeExplainer` (log-odds space). `k = 6` default (configurable).
- `direction` = sign of SHAP (`toward_ransomware` if > 0 else `toward_benign`).
- `percentile_vs_train` from `train_stats.json` quantiles (interpolate against the stored empirical CDF; store a 101-point percentile grid per feature).
- `description`: from `src/evidence/feature_glossary.yaml` (hand-write short descriptions for each feature; for any feature without an entry, use a neutral `"Memory-forensics feature <name>"`. **Never** invent semantics beyond the glossary, because the glossary is the only allowed semantic source for the LLM).
- Also include `qualitative_level`: one of `very_low, low, typical, high, very_high` derived from percentile (<5, 5–25, 25–75, 75–95, >95). The LLM is told to use these words, and the verifier can check them.
- Cache SHAP values for all eval samples (`outputs/shap_eval.npz`).

**Acceptance (Phase 3):** for all eval samples, `sum(shap) + base_value` reproduces the model's raw margin within 1e-3; evidence JSON validates against `schemas/evidence.schema.json` (write the JSON Schema).

---

## 6. LLM layer (generation backends)

### 6.1 Backend adapter interface

```python
# src/llm/base.py
class LLMBackend(Protocol):
    def generate(self, system: str, user: str, *, temperature: float,
                 max_tokens: int, seed: int | None, json_schema: dict | None) -> LLMResult: ...
```

`LLMResult` = `{text, model, backend, latency_s, prompt_tokens, completion_tokens, cache_hit}`.

Implement a **disk cache** keyed by `sha256(backend + model + system + user + temperature + seed + schema)` in `cache/llm/`. Add retry with exponential backoff (max 6 tries), timeout 120 s, and a global rate limiter (`LLM_RPM`, default 20).

### 6.2 Backends

**(a) `openai_compat` (DEFAULT, remote, works with the user's weak PC)**
Any OpenAI-compatible `/v1/chat/completions` endpoint, configured by env:

```
LLM_BACKEND=openai_compat
LLM_BASE_URL=https://openrouter.ai/api/v1      # or Groq / Together / Ollama cloud / your own server
LLM_API_KEY=...                                 # in .env only
LLM_MODEL=<open-weight instruct model id>       # e.g. a Llama-3.x-70B or Mistral/Mixtral instruct id offered by the provider
LLM_RPM=20
```

The agent must **look up currently available model IDs** from the provider's `/models` endpoint rather than hard-coding them (free-tier IDs change often). Use an open-weight model (Llama / Mistral / Qwen family) so the study stays "open-weight LLM". Record the exact model ID and provider in every output file.

If the provider supports JSON-mode / structured outputs, use it; otherwise parse JSON defensively (strip fences, `json.loads`, one repair retry).

**(b) `ollama`**
Local or remote Ollama server (`OLLAMA_HOST`, default `http://localhost:11434`), native `/api/chat` or its OpenAI-compatible `/v1`.
- For this PC only tiny models are viable (≈1–3B, 4-bit, e.g. `llama3.2:3b`, `qwen2.5:3b`); expect slow, lower-quality output. Treat it as a fallback / small-model comparison, not the main run.
- If the user has a stronger machine or Ollama cloud, they only change `OLLAMA_HOST` / model name.

**(c) `llamacpp_server`**
`llama-server` exposes an OpenAI-compatible endpoint, so it reuses backend (a) with `LLM_BASE_URL=http://localhost:8080/v1`. Document, don't build.

### 6.3 `agent_queue` backend (lets the coding agent itself act as the LLM)

For when no API key is available and the local machine can't host a model. Protocol:
1. The pipeline writes pending prompts to `queue/requests.jsonl` as `{"id","system","user","schema"}` and then **exits with code 20** and the message `WAITING_FOR_AGENT`.
2. The coding agent reads each request, writes its own answer to `queue/responses.jsonl` as `{"id","text"}` (one line per request; JSON content must satisfy the schema).
3. The user/agent re-runs the same command; the backend consumes matching ids from `responses.jsonl` (cache-style) and continues.
4. Process in batches (default 25 prompts) so context stays small.
5. **Caveat to record in the report:** when the agent is the generator, the "LLM" is whatever model powers the agent, which may be a closed model. Mark such runs `generator=agent_queue:<agent name>` and **do not present them as open-weight results**.

### 6.4 Connecting Freebuff / OpenCode

The coding agent only needs: shell access in `~/ransomware-rationale/`, the ability to run `make` targets, and read/write files. No special integration is required. Two optional wiring patterns:
- **Agent as orchestrator, remote model as generator (preferred):** the agent runs `make` targets; the pipeline calls the remote API itself.
- **Agent as generator:** use `agent_queue` (§6.3).

If the user wants OpenCode or Freebuff to be configured with a custom OpenAI-compatible provider, tell them to follow *that tool's* current provider docs, and put the same `LLM_BASE_URL/LLM_MODEL` in `.env`. Do not assume exact config syntax.

---

## 7. Rationale generation

### 7.1 Two output modes (run both; they are separate experimental arms)

- **Mode S (structured-then-prose):** LLM returns JSON: `{verdict, confidence_statement, claims: [...], summary}` where each claim is machine-checkable (see §7.3) and `summary` is 2–4 sentences of prose built from those claims.
- **Mode P (free prose):** LLM returns only prose (3–5 sentences). A second LLM call (or the verifier's extractor, §8.2) converts prose to claims. Mode P is the realistic baseline where faithfulness errors are expected.

### 7.2 Prompt (system)

```
You are a malware-analysis assistant explaining the output of a machine-learning
ransomware classifier to a security analyst.

STRICT RULES
1. Use ONLY the evidence provided. Do not mention any feature, value, or behaviour
   that is not in the evidence list.
2. State the classifier's verdict and probability exactly as given.
3. For every feature you mention, state correctly whether it pushed the decision
   TOWARD ransomware or TOWARD benign, exactly as in the evidence ("direction").
4. When describing a feature value, use its qualitative_level word
   (very_low/low/typical/high/very_high) or the exact number given. Never invent numbers.
5. Do not claim causation beyond "contributed to the model's decision".
6. If evidence is mixed or the probability is near the threshold, say the verdict is uncertain.
7. Do not recommend remediation, and do not speculate about malware family unless given.
8. Mention the most influential features first.
```

### 7.3 Claim schema (Mode S output and extractor target)

```json
{
  "verdict": "ransomware | benign",
  "probability_stated": 0.97,
  "claims": [
    {
      "feature": "handles.nhandles",
      "direction": "toward_ransomware | toward_benign",
      "level": "very_low|low|typical|high|very_high|null",
      "value_stated": 18234.0,
      "text": "The sample has a very high number of open handles (18,234), pushing toward ransomware."
    }
  ],
  "summary": "…prose…"
}
```
Define it as a JSON Schema in `schemas/rationale.schema.json` and use it for structured generation and validation.

### 7.4 Generation settings

- `temperature=0.2` main; also run `temperature=0.7` as a robustness arm if budget allows.
- `max_tokens=500`. One sample per prompt, `seed` fixed where supported.
- Few-shot: include **2 hand-written, verified-faithful examples** (from the train split, not eval) in the prompt. Store them in `prompts/fewshot.json`.
- Prompt variants (for ablation): `v0_zero_shot`, `v1_rules`, `v2_rules_fewshot`. Default = `v2`.

---

## 8. Faithfulness scoring method (core contribution)

All checks in A–C are **deterministic** and need no LLM. Implement in `src/faithfulness/`.

### 8.1 Definitions

Let the evidence top-k set be `E = {(f_i, v_i, s_i, d_i, level_i)}` and the rationale's claims be `C = {c_j}`.

**A. Claim-level verification** (each claim gets a status):
| Check | Passes if |
|---|---|
| `feature_exists` | `c.feature ∈ E` (else **hallucinated feature**; also flag if it's a valid model feature but outside top-k → `out_of_topk`) |
| `direction_ok` | `c.direction == E[c.feature].direction` |
| `level_ok` | `c.level == E[c.feature].qualitative_level` (allow adjacent-level tolerance as a separate lenient score) |
| `value_ok` | `|c.value_stated − v| / max(|v|,1) ≤ 0.05` when a number is stated |
| `verdict_ok` | stated verdict and probability match the model output (probability within ±0.02) |

Claim is **supported** iff all applicable checks pass. Claim is **contradicted** if `feature_exists` but `direction_ok`/`level_ok`/`value_ok` fail. Claim is **unsupported** if the feature does not exist in evidence.

**B. Aggregate metrics**
- `claim_precision = supported / total_claims` ← main "faithfulness" number
- `hallucination_rate = unsupported / total_claims`
- `contradiction_rate = contradicted / total_claims`
- `evidence_coverage = Σ|shap_i| for features mentioned / Σ|shap_i| over top-k` (SHAP-mass-weighted recall)
- `top1_mentioned` (bool): was the most influential feature mentioned?
- `verdict_consistency` (bool)
- **Composite Faithfulness Score (FS)** = `verdict_ok × (claim_precision)^0.5 × (0.5 + 0.5·evidence_coverage)`. Document and justify this form, and also report all components separately; do not hide behind the composite. Run a small sensitivity analysis on the weights.
- **Strict pass:** `FS_pass = (contradiction_rate == 0) and (hallucination_rate == 0) and verdict_ok`.

**C. Causal faithfulness (does the cited evidence actually drive the model?)**
For each sample, with `M(x)` = model's calibrated P(ransomware) and `μ` = train medians:
- `Comprehensiveness = M(x) − M(x with cited features set to μ)` (sign-adjusted for predicted class; a faithful rationale ⇒ large drop).
- `Sufficiency = M(x) − M(μ with only cited features restored to x)` (small gap ⇒ cited features suffice).
- **Control:** repeat with `n_cited` random features (100 draws) and with the top-n by |SHAP|. Report `causal_lift = comprehensiveness(cited) − mean(comprehensiveness(random))`. A faithful generator should beat random and approach top-SHAP.
- Note the standard caveat: median-imputation ablation can create off-distribution points; also run a second variant using values resampled from train marginals, and report both.

**D. LLM-judge cross-check (optional, off by default)**
Ask a *different* model whether each claim is entailed by the evidence table; report agreement (Cohen's κ) with the deterministic verifier on 30 samples. Purpose is to validate that the deterministic verifier isn't missing paraphrased claims.

### 8.2 Claim extraction for free prose (Mode P)

Two-stage:
1. **Rule-based extractor** (`src/faithfulness/extract_rules.py`): detect mentions of any model feature name or its glossary aliases (build an alias table: raw name, name with `.`/`_` replaced by spaces, glossary short names). For each mention sentence, detect direction cues (`toward ransomware|indicates malicious|suspicious|raises|increases` vs `toward benign|reassuring|lowers|typical for benign`), level words (`very high|high|elevated|low|very low|typical|normal`), and numbers (regex, handle thousands separators and `k/M` suffixes).
2. **LLM extractor fallback** (same schema as §7.3, temperature 0) for sentences the rules couldn't parse. Always log which extractor produced each claim.

Report **extractor accuracy** on 20 hand-labelled rationales (the agent drafts the labelled set into `data/eval/extractor_gold.jsonl` and flags it for the user to spot-check). State clearly in the report that this gold set was agent-drafted, not independently annotated.

### 8.3 Metric validity tests (must be run; this is what makes the metric credible)

1. **Corrupted-evidence control:** pair each rationale with the evidence of a *different* sample (same verdict). FS must drop sharply; report the drop.
2. **Synthetic perturbations of faithful rationales:** programmatically (a) flip one direction, (b) swap a feature for an out-of-top-k one, (c) change a number by 30%, (d) drop the top feature. Verify the metric detects each with ≥ 95% recall (unit tests in `tests/test_metric_validity.py`).
3. **Fluency-vs-faithfulness decoupling (RQ3):** compute fluency/overlap proxies (e.g. mean length, readability, perplexity under a small LM if cheap, or simply BERTScore/ROUGE-L against a template reference) and show correlation with FS (Spearman). Also show that corrupted rationales score similarly on fluency but much lower on FS.
4. **Human-style spot check:** print 10 randomly chosen rationale/evidence/verifier-output triples in `outputs/spot_check.md` for the user to review.

---

## 9. Repair loop (optional arm, implement it)

`generate → verify → if FS_pass false: send verifier feedback to LLM → regenerate (max 2 rounds)`.
Feedback is deterministic text built from failed checks, e.g.:
`"Claim about 'handles.nhandles' says toward_benign but the evidence says toward_ransomware. Claim about 'foo.bar' is not in the evidence. Rewrite using only the evidence table."`
Report FS before/after, average rounds used, and token cost. Also report whether readability degraded (length change).

---

## 10. Experimental design

**Arms** (all on the same N eval samples, same evidence):
| Arm | Description |
|---|---|
| `T` | Template rationale from evidence (deterministic, faithful by construction; sanity ceiling for faithfulness, floor for fluency) |
| `P0` | LLM free prose, zero-shot prompt (v0) |
| `P2` | LLM free prose, rules + few-shot (v2) |
| `S2` | LLM structured-then-prose (v2) |
| `S2R` | `S2` + repair loop |
| `X` | Negative control: evidence from another sample (§8.3.1) |

Optional extra factor: small local model (≤3B via Ollama) vs. large remote model, to show model-size effect on faithfulness.

**Statistics:** report mean ± 95% bootstrap CI (10,000 resamples) for each metric per arm; paired bootstrap / Wilcoxon signed-rank for `P2` vs `S2` vs `S2R`; stratify by (TP, TN, FN, FP). State N honestly; with N=100, differences under ~0.05 are not reliable.

**Cost control:** estimate calls = N × arms × (1 + repair rounds). Print the estimate and require `--yes` before spending more than 1,000 calls.

---

## 11. Repository layout (create exactly this)

```
ransomware-rationale/
├── README.md
├── Makefile
├── requirements.txt
├── .env.example
├── .gitignore
├── configs/
│   └── default.yaml
├── data/{raw,processed,eval}/
├── models/
├── outputs/
├── cache/llm/
├── queue/                      # agent_queue protocol files
├── logs/
├── docs/DEVIATIONS.md
├── prompts/{system_v0.txt,system_v1.txt,system_v2.txt,fewshot.json}
├── schemas/{evidence.schema.json,rationale.schema.json}
├── src/
│   ├── data/{download.py,preprocess.py,split.py,ember_loader.py}
│   ├── models/{train.py,calibrate.py,report.py}
│   ├── evidence/{shap_builder.py,feature_glossary.yaml,build_evidence.py}
│   ├── llm/{base.py,openai_compat.py,ollama_backend.py,agent_queue.py,cache.py}
│   ├── generation/{prompts.py,generate.py,template_baseline.py,repair.py}
│   ├── faithfulness/{extract_rules.py,extract_llm.py,verify.py,metrics.py,causal.py,judge.py}
│   ├── evaluation/{select_eval_set.py,run_arms.py,stats.py,plots.py,report_md.py}
│   └── utils/{config.py,logging.py,seed.py}
└── tests/
    ├── test_download.py
    ├── test_evidence_additivity.py
    ├── test_verify.py
    ├── test_metric_validity.py
    └── test_llm_cache.py
```

---

## 12. Config (`configs/default.yaml`)

```yaml
seed: 42
dataset: cic_malmem_2022          # or ember (stub)
task: ransomware_vs_rest
split: {train: 0.70, val: 0.10, test: 0.20}
model: xgboost                    # or lightgbm
tuning: {n_trials: 30, cv: 3, metric: average_precision}
evidence: {top_k: 6}
eval:
  n_samples: 100
  strata: {tp: 40, tn: 30, fn: 15, fp: 15}
llm:
  backend: openai_compat          # openai_compat | ollama | agent_queue
  temperature: 0.2
  max_tokens: 500
  prompt_version: v2
  repair_rounds: 2
faithfulness:
  value_tolerance: 0.05
  prob_tolerance: 0.02
  causal_random_draws: 100
  judge_enabled: false
```

`.env.example`:
```
LLM_BACKEND=openai_compat
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=
LLM_MODEL=
LLM_RPM=20
OLLAMA_HOST=http://localhost:11434
KAGGLE_USERNAME=
KAGGLE_KEY=
```

---

## 13. Core code sketches (implement and extend; these show intended logic)

### 13.1 Verifier

```python
# src/faithfulness/verify.py
from dataclasses import dataclass

LEVELS = ["very_low", "low", "typical", "high", "very_high"]

@dataclass
class ClaimResult:
    feature: str
    status: str            # supported | contradicted | unsupported | out_of_topk
    failed_checks: list[str]

def verify_claim(claim: dict, evidence: dict, tol=0.05) -> ClaimResult:
    topk = {f["name"]: f for f in evidence["top_features"]}
    name = claim["feature"]
    if name not in topk:
        status = "out_of_topk" if name in evidence["all_feature_names"] else "unsupported"
        return ClaimResult(name, status, ["feature_exists"])
    e, failed = topk[name], []
    if claim.get("direction") and claim["direction"] != e["direction"]:
        failed.append("direction_ok")
    if claim.get("level") and claim["level"] != e["qualitative_level"]:
        failed.append("level_ok")
    v = claim.get("value_stated")
    if v is not None and abs(v - e["value"]) / max(abs(e["value"]), 1.0) > tol:
        failed.append("value_ok")
    return ClaimResult(name, "supported" if not failed else "contradicted", failed)

def score_rationale(rationale: dict, evidence: dict, cfg) -> dict:
    results = [verify_claim(c, evidence, cfg.value_tolerance) for c in rationale["claims"]]
    n = max(len(results), 1)
    sup = sum(r.status == "supported" for r in results)
    unsup = sum(r.status in ("unsupported", "out_of_topk") for r in results)
    contra = sum(r.status == "contradicted" for r in results)
    mentioned = {c["feature"] for c in rationale["claims"]}
    mass_total = sum(abs(f["shap"]) for f in evidence["top_features"]) or 1.0
    mass_cov = sum(abs(f["shap"]) for f in evidence["top_features"] if f["name"] in mentioned) / mass_total
    verdict_ok = (rationale["verdict"] == evidence["verdict"]
                  and abs(rationale.get("probability_stated", evidence["probability_ransomware"])
                          - evidence["probability_ransomware"]) <= cfg.prob_tolerance)
    precision = sup / n
    fs = float(verdict_ok) * (precision ** 0.5) * (0.5 + 0.5 * mass_cov)
    return dict(claim_precision=precision, hallucination_rate=unsup / n,
                contradiction_rate=contra / n, evidence_coverage=mass_cov,
                top1_mentioned=evidence["top_features"][0]["name"] in mentioned,
                verdict_ok=verdict_ok, FS=fs,
                FS_pass=(unsup == 0 and contra == 0 and verdict_ok and len(results) > 0),
                per_claim=[r.__dict__ for r in results])
```

Edge cases to handle: zero claims (FS = 0, flag `no_claims`), duplicate claims about one feature (dedupe, but a conflicting duplicate counts as contradiction), `value_stated` with units or `k` suffix.

### 13.2 Causal ablation

```python
# src/faithfulness/causal.py
def ablate(x, features, medians):           # set cited features to train median
    x2 = x.copy()
    for f in features: x2[f] = medians[f]
    return x2

def comprehensiveness(model, x, cited, medians, pred_class):
    p  = model.predict_proba(x)[pred_class]
    p2 = model.predict_proba(ablate(x, cited, medians))[pred_class]
    return p - p2
```
Include the random-feature control, the top-n-SHAP control, and the marginal-resampling variant.

### 13.3 Template baseline (arm T)

Deterministic: for each top-k feature emit
`"{name} is {level} ({value}, {percentile:.0f}th percentile of training data), which pushed the decision {direction_phrase}."` joined into a paragraph, preceded by `"The classifier predicts {verdict} with probability {p:.2f}."` This must score FS ≈ 1.0; if it doesn't, the verifier has a bug, so use it as a unit test.

---

## 14. Makefile targets

```
make setup        # venv + pip install -r requirements.txt
make download     # §3.1 download chain with fallbacks
make preprocess   # clean, dedupe, label, split
make train        # tune + train + calibrate + classifier_report
make evidence     # SHAP + evidence JSON for the eval set
make generate     # all arms (respects cache; may exit 20 in agent_queue mode)
make verify       # extraction + faithfulness metrics + causal tests
make validity     # §8.3 validity tests
make eval         # stats, plots, outputs/report.md
make test         # pytest
make all          # setup → ... → eval
```
Each target must be idempotent and skip finished work unless `FORCE=1`.

### requirements.txt (pin after first successful install)
```
numpy pandas scikit-learn xgboost lightgbm shap pyyaml jsonschema
requests httpx tenacity tqdm joblib matplotlib scipy
pytest python-dotenv huggingface_hub kaggle beautifulsoup4
```

---

## 15. Phases and acceptance checks

| Phase | Deliverable | Acceptance |
|---|---|---|
| 0 | Project skeleton, env, config | `make setup` OK; `df -h` result logged |
| 1 | Dataset downloaded | file present, >50k rows, label columns found, row/class counts in `outputs/data_summary.md` |
| 2 | Classifier | ROC-AUC ≥ 0.95, report written, models load |
| 3 | Evidence builder | additivity test passes; schema validation passes |
| 4 | LLM adapter + cache | live ping works (or agent_queue round trip works); cache hit verified in `test_llm_cache.py` |
| 5 | Generation (arms T, P0, P2, S2) | N×arms rationales saved in `outputs/rationales/*.jsonl` with model id, prompt version, latency |
| 6 | Faithfulness scorer | `test_verify.py` passes; template arm FS ≥ 0.99 |
| 7 | Metric validity | perturbation recall ≥ 95%; corrupted-evidence FS drop reported |
| 8 | Repair loop (S2R) | before/after table produced |
| 9 | Full evaluation | `outputs/report.md` with tables, CIs, plots, limitations |

---

## 16. Final report (`outputs/report.md`) must contain

1. Setup: dataset version/URL actually used, split sizes, eval-set composition (actual counts), classifier metrics.
2. LLM details: provider, exact model ID, backend, temperature, prompt version; flag if any arm used `agent_queue`.
3. Table: per-arm means with 95% CI for `claim_precision`, `hallucination_rate`, `contradiction_rate`, `evidence_coverage`, `FS`, `FS_pass rate`, causal lift.
4. Breakdown by stratum (TP/TN/FN/FP).
5. Validity results (§8.3) and extractor accuracy (with the agent-drafted-gold caveat).
6. Fluency-vs-faithfulness correlation plot.
7. Qualitative examples: 3 faithful, 3 unfaithful rationales with verifier output.
8. **Limitations:** SHAP ≠ ground-truth causality; median-ablation off-distribution risk; MalMem is a lab-generated dataset and may not generalize to real-world ransomware; small N; glossary descriptions are hand-written; single LLM/provider; the metric checks agreement with the model's evidence, not whether the model itself is right about ransomware.
9. Reproduction commands and random seeds.

---

## 17. Common pitfalls (avoid)

- Letting the LLM see raw SHAP floats it might misquote: pass rounded values and the qualitative words.
- Data leakage via duplicate rows, which inflates AUC; dedupe before splitting.
- Computing percentiles/medians on the full data instead of train only.
- Treating the template arm's perfect score as a result: it is only a sanity check.
- Reporting FS only; always show components.
- Spending API budget before the cache and cost estimator are in place.
- Claiming open-weight results for runs generated by the coding agent itself.

---

## 18. Definition of done

`make all` runs from a clean clone (with a valid `.env`, or via `agent_queue`) and produces `outputs/report.md`, all tests pass, `docs/DEVIATIONS.md` lists any departures from this spec, and the final message to the user summarizes: classifier quality, per-arm faithfulness numbers, the main finding for each of RQ1–RQ4, and the known limitations.
