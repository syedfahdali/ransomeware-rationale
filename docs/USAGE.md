# Usage guide

Everything you can run, what it produces, and how to work with the agent-queue
LLM backend. For the design see [ARCHITECTURE.md](ARCHITECTURE.md); for the
results see [../outputs/report.md](../outputs/report.md).

---

## 1. Prerequisites

| requirement | notes |
|---|---|
| Python 3.11+ | the venv on this machine is 3.14.4 (see [DEVIATIONS.md](DEVIATIONS.md) #2) |
| `make` | every stage has a target; modules work without make |
| ~1.5 GB disk | dataset (19 MB) + parquet + models + cache |
| Kaggle credentials *(optional)* | `~/.kaggle/kaggle.json` for `make download`; the downloader also tries the UNB page, HuggingFace and a manual drop |
| LLM endpoint *(optional)* | only if you switch off the default `agent_queue` backend |

## 2. Install

```bash
cp .env.example .env        # safe defaults; no secrets required for agent_queue
make setup                  # creates .venv, installs deps, writes requirements.lock.txt
```

`make setup` runs [scripts/install.sh](../scripts/install.sh) and is safe to
re-run. It installs core deps first, then extras (`shap`, `lightgbm`, `kaggle`,
`beautifulsoup4`, `huggingface_hub`), then freezes exact versions.

## 3. One-command run

```bash
make all
```

Runs, in order: `setup → download → preprocess → train → evidence → cli →
generate → verify → validity → eval → notebooks → test → checks`, and ends with:

```
**11/12 checks passed.**
make all finished
```

> `make all` reproduces the **research study**. You do not need it to *use* the
> project — for everyday triage go straight to
> [section 12](#12-everyday-triage-check-one-sample-explain-it-verify-the-explanation)
> (`make check` / `make triage` / `python -m src.cli`), which only needs
> `make setup download preprocess train`.

Typical artifacts after a run:

```
outputs/report.md            the study (start here)
outputs/classifier_report.md model card
outputs/evidence/            50 SHAP evidence JSONs
outputs/rationales/          one JSONL per arm
outputs/faithfulness/        scores, per-arm summaries, stats.json
outputs/validity.json        metric-validity results
outputs/phase_checks.json    acceptance-check table
```

## 4. Stage by stage

Each stage is idempotent: it detects finished work and skips (`logs/<stage>.log`
records what happened). Force a redo with `FORCE=1 make <target>`.

### `make download`
Fetches the dataset to `data/raw/CIC-MalMem-2022.csv` and validates it
(≥ 50 000 rows, ≥ 50 columns, label columns). Provenance lands in
`data/raw/download_info.json`:

```json
{ "path": ".../data/raw/CIC-MalMem-2022.csv", "source": "existing",
  "info": { "rows": 58596, "cols": 57, "has_labels": true } }
```

If every online source fails, drop `Obfuscated-MalMem2022.csv` into `data/raw/`
by hand and re-run — the stage accepts any file that passes the validity gate.

### `make preprocess`
Cleans, dedupes (58 596 → 58 023 rows), derives labels and the capture `group_id`,
and writes the group-aware 70/10/20 split:

```
train 40540 · val 5813 · test 11670 rows
leak check OK: no group overlap, no identical rows across splits
```

Outputs: `data/processed/{all,train,val,test}.parquet`, `features.json`,
`splits.json`, `outputs/data_summary.md`.

### `make train`
Tunes with `RandomizedSearchCV` inside a wall-clock budget, early-stops, refits,
calibrates:

```
probe fit 1.4s -> n_iter=30 (budget 1200.0s)
best params: {'subsample': 0.7, 'reg_lambda': 1.0, 'reg_alpha': 0.0, 'min_child_weight': 1,
              'max_depth': 8, 'learning_rate': 0.2, 'colsample_bytree': 0.8}
              (cv average_precision=0.68222)
early stopping -> 191 trees
```

Outputs: `models/clf.joblib`, `models/calibrator.joblib`, `models/best_params.json`,
`models/metrics.json`, `models/train_stats.json`, `outputs/classifier_report.md`,
`outputs/figs/{test_roc,test_pr,reliability_test,confusion_test}.png`.

Switch the learner in [configs/default.yaml](../configs/default.yaml):

```yaml
model: lightgbm      # was: xgboost
```

### `make evidence`
Computes exact TreeSHAP for the 50 eval samples and writes one evidence JSON per
sample, validating against the schema and the additivity identity:

```
SHAP additivity max error: 6.696e-06
wrote 50 evidence files (0 schema-invalid)
```

### `make cli`

Smoke-tests the everyday triage entry point so the usage path is part of the
acceptance table: it checks the first evaluation sample (verdict + evidence +
faithfulness of the deterministic explanation), prints the `info` banner, and
triages 5 rows of the raw CSV. It records two `usage` rows in
`outputs/phase_checks.json`. Takes a few seconds.

### `make check` / `make triage`

Thin wrappers around the CLI for everyday use:

```bash
make check SAMPLE=test_031534                       # one sample, verified explanation
make check SAMPLE=test_030109 ARGS="--explain llm --repair 2"
make triage                                          # CSV=data/raw/CIC-MalMem-2022.csv
make triage CSV=dumps.csv STRIDE=1 LIMIT=100         # your own rows
```

See [section 12](#12-everyday-triage-check-one-sample-explain-it-verify-the-explanation)
for the full flag reference.

### `make generate`
Generates the six arms. With the default backend it may pause:

```
========================================================================
WAITING_FOR_AGENT — arm P0
unanswered requests: 25 (active batch: 25)
requests -> .../queue/requests.jsonl
answers  -> .../queue/responses.jsonl   (append one JSON line per request)
Then re-run the exact same command.
========================================================================
```

Exit code **20** is expected, not an error (see §6). Output:
`outputs/rationales/{T,P0,P0R,P2,S2,S2R}.jsonl` + `llm_metadata.json`.

### `make verify`
Extracts claims from every rationale (rules first, LLM fallback for unparsed
sentences), checks them against the evidence, runs the causal ablation and writes
`outputs/faithfulness/{arm}.jsonl`, `{arm}_summary.json`,
`all_arms_summary.{json,csv}`. Console summary:

```
arm P0: n=50 FS=0.772 precision=0.924 halluc=0.045 contra=0.032 coverage=0.607 pass=0.78
arm P0R: n=50 FS=0.891 precision=0.988 halluc=0.012 contra=0.000 coverage=0.790 pass=0.96
```

### `make validity`
Metric-validity tests + `pytest tests/test_metric_validity.py`:

```
perturbation recall: {'flip_direction': 1.0, 'out_of_topk_feature': 1.0,
                      'value_30pct': 0.98, 'drop_top': 1.0}
```

Outputs: `outputs/validity.json`, `outputs/spot_check.md` (10 random
rationale/evidence/verifier triples for manual review).

### `make eval`
Statistics (bootstrap CIs, paired tests, strata), plots and the report:

```
stats: P0: FS 0.772 [0.728, 0.815] pass 0.780 lift 0.155
report_md: wrote outputs/report.md
```

### `make test` / `make checks`

```bash
make test       # 28 passed
make checks     # acceptance table; 11/12 passed (phase2 documented in DEVIATIONS.md)
make notebooks  # executes both notebooks with plain Python (no Jupyter needed)
```

## 5. Running modules directly

Every stage is a module, which is convenient for partial re-runs and debugging:

```bash
python -m src.data.preprocess --force
python -m src.models.train
python -m src.evidence.build_evidence
python -m src.generation.generate --arms T,P0,P2 --yes --force
python -m src.evaluation.run_arms --arms P0,P0R --force
python -m src.evaluation.validity
python -m src.evaluation.stats
python -m src.evaluation.plots
python -m src.evaluation.report_md
python -m src.utils.checks
```

The everyday entry point (single samples, batches, JSON output for tool use):

```bash
python -m src.cli info
python -m src.cli check --sample test_030109 --explain llm --repair 2
python -m src.cli batch --csv data/raw/CIC-MalMem-2022.csv --stride 1500 --limit 30
python -m src.cli template --out my_sample.csv
```

Inspect the extractor on any sentence (small built-in CLI):

```bash
$ python -m src.faithfulness.extract_rules \
  "The model predicts benign with probability 0.03. Very high service count (392) and low mutex handles (186) lean benign."
{"feature": "svcscan.nservices", "direction": "toward_benign", "level": "very_high", "value_stated": 392.0, ...}
{"feature": "handles.nmutant",   "direction": "toward_benign", "level": "low",       "value_stated": 186.0, ...}
```

Score one rationale against one evidence file:

```python
import json
from src.faithfulness.verify import score_rationale
ev = json.load(open("outputs/evidence/test_013221.json"))
rationale = {"verdict": "benign", "probability_stated": 0.029,
             "claims": [{"feature": "handles.nsection", "value_stated": 339.0}]}
print(score_rationale(rationale, ev)["FS"])
```

## 6. The agent-queue workflow

`agent_queue` turns the coding agent into the LLM: the pipeline writes the prompts
it needs, exits, and continues once they are answered. Files live in
[queue/](../queue):

| file | role |
|---|---|
| `requests.jsonl` | the current batch of unanswered prompts (one JSON object per line) |
| `responses.jsonl` | your answers, appended as `{"id": "...", "text": "..."}` or `{"id": "...", "json": {...}}` |
| `requests_archive.jsonl` | every request ever emitted (ids are deterministic sha256 prefixes) |
| `status.json` | `n_unanswered_total`, `n_in_active_batch`, `batch_size`, file paths |
| `HOWTO.md` | the short protocol reminder, rewritten on every batch |

Round trip:

```bash
$ make generate            # exits 20, writes queue/requests.jsonl
$ cat queue/status.json    # {"n_unanswered_total": 25, "n_in_active_batch": 25, ...}
$ cat queue/requests.jsonl | head -1 | python -m json.tool | head -20

# ... answer every pending id into queue/responses.jsonl ...

$ make generate            # same command: consumes answers, queues the next batch or finishes
```

Answering programmatically (adapt the mapping to your model/endpoint):

```python
import json, pathlib
rows = [json.loads(l) for l in open("queue/requests.jsonl") if l.strip()]
with open("queue/responses.jsonl", "a") as out:          # append, never rewrite
    for r in rows:
        answer = my_llm(r["system"], r["user"])           # prose
        out.write(json.dumps({"id": r["id"], "text": answer}) + "\n")
        # or: {"id": r["id"], "json": my_structured_answer}
```

Rules that keep it safe: never edit `queue/requests.jsonl`; append to
`queue/responses.jsonl`; ids are hash-derived, so re-running is always safe and
answers are cached afterwards (a second run costs nothing).

## 7. LLM backends

`.env` (copy of [.env.example](../.env.example)):

```bash
# 1) the agent itself (default; no key needed)
LLM_BACKEND=agent_queue

# 2) any OpenAI-compatible endpoint
LLM_BACKEND=openai_compat
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-...
LLM_MODEL=meta-llama/llama-3.1-8b-instruct

# 3) Ollama
LLM_BACKEND=ollama
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=llama3.1:8b
```

Discover available model ids instead of guessing:

```bash
python -m src.llm.openai_compat --list-models      # queries $LLM_BASE_URL/v1/models
```

After switching, re-run generation and everything downstream:

```bash
FORCE=1 make generate && FORCE=1 make verify && make validity && make eval
```

Raw model notes: on an 8 GB CPU-only laptop only small quantised models are
viable, which is why the default run uses `agent_queue` — and why the report
labels the generator explicitly as *not* open-weight. The disk cache
(`cache/llm/`) is keyed on backend+model+prompt+temperature+seed, so runs with
different models never collide.

## 7b. Optional: the Laya decision layer

A non-generative "System 1" second opinion
([Laya](https://github.com/NandhaKishorM/laya)) that answers typed `choice`
questions — no text generation, so nothing to parse. **Off by default**, and
purely **diagnostic**: it never changes `FS`. Read
[docs/LAYA_REVIEW.md](LAYA_REVIEW.md) first; the short version is that the layer is
cheap and safe, not that the study needs it.

```bash
# 1. install the optional extra (either one)
make setup-laya                       # into .venv; installs CPU torch FIRST (~1.2 GB)
#   or keep the study venv torch-free:
python3 -m venv .venv-laya
.venv-laya/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-laya/bin/pip install -r requirements-laya.txt

# 2. run the stage (writes outputs/laya/)
FORCE=1 make laya_gate                 # the six study arms x 50 samples (~55 min wall on CPU)
.venv/bin/python -m src.faithfulness.laya_gate --force --arms X   # the corrupted control (+20 min)
python -m src.faithfulness.laya_gate --force --arms S2,P0,X --limit 6   # a quick look

# 3. or ask for it inline on one sample / a batch
python -m src.cli check --sample test_031534 --explain template --decide laya
python -m src.cli batch --csv dumps.csv --limit 30 --decide laya
```

`--force` runs it even though `laya.enabled` is false. Set `laya.enabled: true` in
[configs/default.yaml](../configs/default.yaml) to make plain `make laya_gate` (and
the `laya_gate` step of `make all`) pick it up — that is a deliberate opt-in, see the
verdict below. Checkpoints download from Hugging Face on first use into
`HF_HUB_CACHE` (default `~/.cache/huggingface/hub`); set it in `.env` to move them.

With the layer installed and running, `laya_gate_latency` in
`outputs/phase_checks.json` reports the real CPU latency and **FAILS** against the
proposal's 100 ms budget — that is the finding, not a broken check.

What lands in `outputs/laya/`:

| file | contents |
|---|---|
| `{arm}_gate.jsonl` | one row per sample: classifier verdict, Laya's verdict, the gated verdict, `confidence`, `answer_confidence`, `top_option_prob`, token usage, latency |
| `{arm}_claims.jsonl` | one row per sample with a `per_claim` list: verifier status + failed checks, `laya_status`, both confidences, the raw option probabilities, agreement flags, latency |
| `summary.json` | result of the **most recent invocation**: per-arm `gate_agreement_rate`, `LAR` (collated and strict), claim counts, median latencies, plus the checkpoint/device/limit that produced them. `{"status": "skipped", ...}` when the layer is off. |
| `summary_full_run.json` | accumulated whenever an unlimited pass finishes (the six study arms together, the `X` control separately — 7 arms × 50 samples), so a later disabled run or a `--limit N` probe cannot erase the reference numbers (the `laya_full_run_recorded` check re-affirms it) |
| `disagreements.md` | the spot-check file: the cases worth a human look, in the style of `outputs/spot_check.md` |

Reading the results: `LAR` is the fraction of claims where Laya's typed status
matches the verifier's, collated (`unsupported` and `out_of_topk` collapse into
`not_in_evidence`, since Laya cannot distinguish them without the full feature
list). A high LAR is **not** evidence of correctness — it is also what you get when
both checkers are constant. The number that matters is recall on claims the
verifier flags as `contradicted`, cross-tabulated in `disagreements.md`.

The measured verdict on this layer (≈9 % recall on flagged claims, a gate threshold
that can never fire, ~5.7 s per CPU call, and fine-tuning labels that are all
circular) is in the README section *Does it improve the system?* and in
[LAYA_REVIEW.md](LAYA_REVIEW.md).

Building fine-tuning data from this repo's artifacts (see
[LAYA_REVIEW.md §2](LAYA_REVIEW.md) for why the default refuses the circular rows):

```bash
python scripts/make_laya_finetune_set.py                  # independent labels only
python scripts/make_laya_finetune_set.py --allow-circular  # + verifier/classifier-labelled bulk
```

## 8. Output reference

| path | produced by | contents |
|---|---|---|
| `data/raw/CIC-MalMem-2022.csv` | download | the dataset |
| `data/processed/*.parquet` | preprocess | splits with features + labels |
| `models/clf.joblib`, `models/calibrator.joblib` | train | booster + sigmoid calibrator |
| `models/best_params.json`, `models/metrics.json` | train | chosen hyperparameters, metrics |
| `outputs/evidence/{sample_id}.json` | evidence | the facts a rationale may cite |
| `outputs/rationales/{arm}.jsonl` | generate | one record per sample: text/rationale, repair history |
| `outputs/faithfulness/{arm}.jsonl` | verify | per-sample claims, statuses, causal numbers |
| `outputs/faithfulness/all_arms_summary.csv` | verify | one row per arm, all metrics |
| `outputs/faithfulness/stats.json` | eval | bootstrap CIs, comparisons, strata |
| `outputs/validity.json` | validity | perturbations, controls, gold accuracy, fluency |
| `outputs/report.md` | eval | the final study |
| `outputs/phase_checks.json` | all stages | acceptance checks |
| `logs/<stage>.log` | all stages | full run logs |

Quick looks:

```bash
python -m src.utils.checks                       # acceptance table
column -s, -t outputs/faithfulness/all_arms_summary.csv | head
python - <<'PY'
import json; m=json.load(open("outputs/rationales/llm_metadata.json")); print(m)
PY
```

## 9. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `make generate` exits 20 | expected: prompts are waiting for the agent | answer `queue/requests.jsonl` into `queue/responses.jsonl` and re-run the same command |
| stage prints "already complete" but you changed code | stage-level caching | `FORCE=1 make <target>` |
| `download` fails: "no ~/.kaggle/kaggle.json" | Kaggle CLI not configured | create the token file, or drop the CSV in `data/raw/` manually |
| `ValueError: no Class/Category label column found` | wrong file downloaded | delete `data/raw/*.csv` and re-run `make download` |
| SHAP import warning in `logs/build_evidence.log` | `shap` package missing/incompatible | harmless: the builder falls back to XGBoost `pred_contribs` (identical exact values) |
| report shows `agent_queue` and "not open-weight" | default backend | expected; switch `LLM_BACKEND` for open-weight runs |
| `phase2` FAIL in `make checks` | spec threshold assumes a leaky split | expected; see [DEVIATIONS.md](DEVIATIONS.md) #4 |
| `MemoryError` during train | XGBoost search on a small machine | lower `tuning.n_trials` / `time_budget_s` in `configs/default.yaml` |

## 10. Docs tooling

```bash
python scripts/check_docs.py        # verifies every path/link in the docs exists
python scripts/make_demo_gif.py     # regenerates docs/demo.gif + docs/demo_check.gif (offline)
python scripts/run_notebooks.py     # executes every notebook code cell in order
```

## 12. Everyday triage (check one sample, explain it, verify the explanation)

`python -m src.cli` is the single-sample entry point. It reuses the trained model,
the evidence builder and the verifier, so nothing else has to run first.

```
python -m src.cli <command> [options]

commands
  check      check ONE sample: verdict, evidence, explanation, faithfulness
  batch      run many rows of a CSV and write a triage table
  template   write an example input CSV to edit (55 features + Category)
  info       show the loaded model, feature count, paths and example commands
```

### `check` — one sample

| flag | meaning |
|---|---|
| `--sample <id>` | evaluation sample id (`test_003028`, listed in `data/eval/eval_ids.json`) |
| `--csv <path>` `--row <n>` | one row of a CSV holding the 55 features (+ optional `Category`/`Class`) |
| `--json-file <path>` | a JSON object/array with the same fields |
| `--split {test,val,train}` | which split `--sample` is looked up in (default `test`) |
| `--top-k <n>` | evidence features in the table (default `evidence.top_k` = 6) |
| `--allow-missing` | fill missing features with the training median (warns; verdict is weaker) |
| `--explain {none,template,llm}` | `template` (default) = offline deterministic explanation; `llm` = ask the configured backend; `none` = verdict + evidence only |
| `--mode {prose,structured}` | LLM answer style (`structured` = JSON validated against `schemas/rationale.schema.json`) |
| `--prompt-version {v0,v1,v2}` | `v0` zero-shot, `v1` rules, `v2` rules + few-shot (default from the config) |
| `--backend {agent_queue,openai_compat,ollama}` | override `LLM_BACKEND` for this call |
| `--repair <N>` | verify-and-repair rounds: re-ask with the verifier's correction list (≤ N times) |
| `--temperature <t>` | sampling temperature for the LLM |
| `--json` | print one JSON object on stdout (tool/agent use) |
| `--save <path>` | write the full record (evidence + explanation + scores) as JSON |
| `--fail-on-unfaithful` | exit **1** when the explanation fails the faithfulness check |

Exit codes: `0` ok, `1` `--fail-on-unfaithful` failed, `2` bad input/missing model,
`20` waiting for the agent queue (`queue/HOWTO.md`).

### `batch` — many rows

| flag | meaning |
|---|---|
| `--csv <path>` | required; each row is one sample (55 features) |
| `--limit <n>` | only the first N rows (0 = all) |
| `--stride <k>` | take every k-th row (the dataset is ordered by class, so a stride mixes classes) |
| `--out <path>` | output CSV (default `outputs/triage_batch.csv`) |
| `--explain template` | also score the deterministic explanation of every row |
| `--top-k <n>` | evidence features per row (default 6) |

Output columns: `sample_id`, `verdict`, `predicted_label`, `probability_ransomware`,
`dataset_label`, `dataset_malware_class`, `family`, `correct` (when labels exist),
`FS`/`FS_pass`/`n_claims` (with `--explain template`), `top_features`, `missing_features`.

### Reading the result

* `FS` (0-1) is the composite Faithfulness Score; `FS_pass` additionally requires
  zero unsupported/contradicted claims and an agreed verdict. `per_claim[].status`
  is one of `supported`, `contradicted`, `unsupported`, `out_of_topk`, and
  `failed_checks` names the exact rule (`direction_ok`, `level_ok`, `value_ok`,
  `feature_exists`).
* A `benign` verdict means **not ransomware** — Trojan/Spyware captures are
  negatives in this task. The CLI prints that scope line and adds a `WARNING` when
  the dataset label is another malware class.
* `warnings[]` in `--json` carries the same caveats for programs.

More: [Real-world usage](../README.md#real-world-usage--everyday-triage) in the
README, the two [notebooks](../notebooks/) and the agent skill in
[skills/malware-triage/SKILL.md](../skills/malware-triage/SKILL.md).

## 11. Determinism and reproducibility

* Every stage seeds from `seed: 42` in `configs/default.yaml`
  (`src/utils/seed.py`).
* LLM outputs are cached on disk, so re-running a finished study is a no-op and
  the numbers do not move.
* Nothing in `outputs/report.md` is typed by hand: the report generator reads
  `models/*.json`, `data/*.json`, `outputs/validity.json`,
  `outputs/faithfulness/*` and the checks table.
