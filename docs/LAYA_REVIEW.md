# Critical review: the Laya decision-layer proposal

**Subject:** [`Laya Decision-Layer Architecture Overhaul.md`](../Laya%20Decision-Layer%20Architecture%20Overhaul.md)
**Upstream:** [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) · `laya` 0.4.2 on PyPI · checkpoint repo [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
**Verdict up front:** the plumbing is worth having; the *proposal* is not, as written. Three of its
claims are internally contradicted by its own numbers, and the fine-tuning plan it calls
"a required step" is circular in both of its two tasks. I implemented the defensible subset
(off by default, diagnostic only, FS provably untouched) and did **not** implement the parts
that would have made the study's results weaker. Details, with measurements, below.

---

## 1. What the proposal gets right

Credit where it is due — these are real observations that the current architecture does not address:

- **A second opinion that fails differently is genuinely useful.** `extract_rules.py` is regex +
  alias matching with known gaps: on the gold set it scores precision/recall **0.43/0.22** for
  zero-shot prose (P0) versus **0.95/0.95** for rule-following prose (P2). Nothing in the pipeline
  currently tells you *which* claims the extractor missed; an independent checker could.
- **`noul` really is a trap**, and the proposal correctly refuses to use it for anything
  load-bearing. I kept that constraint and asserted it in a test
  (`test_questions_never_use_noul_and_stay_small`).
- **The metric boundary is right in principle.** "LAR is reported per arm next to FS, not folded
  into it" is the correct call, and §8's argument for it (blending would make metric validity
  circular) is sound.
- **Soft-fail is the correct integration discipline.** The proposal's insistence that
  `make all` behave identically with Laya absent is exactly right, and is implemented.
- **§9 step 4 is the sharpest idea in the document**: evaluate Laya *as an extractor* against the
  existing gold set, to get a directly comparable precision/recall. That is the one measurement
  that would actually tell you something. (It is also the one the proposal never runs.)

## 2. Where it is wrong

### 2.1 The framing oversells a diagnostic as a dependency

The summary claims three things "the current architecture cannot do on its own". FS is not one of
them. `score_rationale()` compares stated direction / level / value against the evidence row by
exact match; it is *faithful by construction* and needs no second opinion, as §"Honest limits"
later concedes. Every Laya output ends up in `outputs/laya/` as a diagnostic. A diagnostic is
fine — but it is not a reason to put a 421M-parameter model in the study's critical path, which
is what "default-on" and "`make all` runs it automatically" amount to.

### 2.2 **Fatal: both fine-tuning targets are circular**

This is the part that cannot be fixed by tuning or by a better split. §9 proposes two supervised
sets and calls the volume "larger and comes for free":

| task | proposed label | what training on it teaches |
|---|---|---|
| gate | `(evidence table) → classifier verdict` | to **reproduce the classifier's verdict** |
| claims | `(claim text, evidence row) → verifier status` | to **reproduce the rules verifier's status** |

The gate exists to answer *"does the classifier's verdict match what the evidence says?"* — i.e.
its entire value is the cases where it **disagrees** with the classifier. Training it on the
classifier's verdict optimises agreement, which trains away the only signal it produces. The
cross-judge exists to catch the rule extractor's **blind spots** (§1); training it on the rule
verifier's status teaches it to be wrong in exactly the same places.

The proposal pre-empts the obvious objection — "split by the same capture-group-aware `group_id`"
— but that answers a different question. A group-aware split prevents **train/test leakage**. It
does nothing about **label-provenance circularity**. Those are independent failure modes and only
the first is a splitting problem.

`scripts/make_laya_finetune_set.py` makes both concrete. Running it on this repo's artifacts:

```
gate   50 rows   independent labels:  0   label_provenance: {classifier_verdict: 50}
claims 1614 rows independent labels: 90   label_provenance: {rule_verifier: 1524,
                                                           extractor_gold: 90}
```

The gate set has **zero** non-circular labels, by construction — there is no source of
independent "what verdict does this evidence support?" ground truth in the repo. The claim set's
"~1,800 labelled decisions" resolve to **90 usable labels (5.6%)**, all drawn from the same 20
hand-checked rationales that already exist.

### 2.3 Also unaddressed: every harvestable label is in the held-out test split

The proposal treats `outputs/evidence/*.json` and `outputs/faithfulness/*.jsonl` as free training
data. They are not free: they are **the study's own evaluation set**. Every row the script emits
lands in `split: test` — because that is all the pipeline ever produced:

```
gate   splits: {"test": 50}
claims splits: {"test": 1614}
```

There are **no train or val rows at all**. Fine-tuning on these artifacts means fine-tuning on the
same 50 samples the six arms are reported over. The classifier's `group_id` split protects the
*classifier*; it cannot protect a model trained on post-`verify` outputs of that same test split.

### 2.4 The proposal contradicts itself on its own defaults and budget

| claim | location | contradicted by |
|---|---|---|
| `laya.enabled: true` | §10 config block | §"Rollout plan" phase 0: "`laya.enabled: false` by default". I shipped `false`; the rollout table is the correct instinct. |
| "default-on when the `laya` package is installed" | §4 (CLI) | §1/§9: base checkpoints are "**near chance** zero-shot … 0.362 accuracy against a 0.461 majority-class baseline", and "shipping this without the fine-tuning step would make LAR noisy and the pre-generation gate unreliable". Default-on with a below-baseline model is the wrong default regardless of which section you believe. |
| "adding well under 100 ms total to a call" | §4 | §"Added cost": "193–464 ms on CPU **per call**". The CLI path makes 1 gate call + 1 call per claim (6 claims = 7 calls) → ~1.4–3.3 s per sample even on the proposal's own numbers, not <100 ms. The proposal's own acceptance check `laya_gate_latency` (median ≤ 100 ms **on CPU**) therefore cannot pass by construction. Measured here: **~16 s per sample** (§4.1). |
| "33 ms" | header, §"Mechanics" | A T4 GPU figure. This host has no GPU (`nvidia-smi` is absent) and `device: cpu` in the proposed config. |

The measured number on this host is in §4 below.

### 2.5 "Nothing for it to hallucinate" is a category error

Hallucination risk lives in the *generator*. Laya generates no text, which removes the
parse-and-verify failure mode — a real benefit — but a **mis-scaled or simply wrong typed answer
is still a wrong answer**. The proposal's own §1 (`noul` returning a confident "no" on clearly
positive input) is an example of the model being confidently wrong. "Hallucination-proof" does not
mean "reliable"; it means "fails in a shape that is easier to detect". That distinction matters
because the whole proposal leans on the former phrasing.

### 2.6 The install cost is omitted

§"Added cost is small and bounded" quotes only runtime. Measured, on this host (Linux x86_64,
Python 3.14, no GPU), the default resolution of `pip install laya`:

- `torch 2.14.1` + `transformers 5.19.0` + **19 CUDA-related packages** — 15 `nvidia-*` wheels
  (`cublas`, `cuda-cupti`, `cuda-nvrtc`, `cuda-runtime`, `cudnn-cu13`, `cufft`, `cufile`,
  `curand`, `cusolver`, `cusparse`, `cusparselt-cu13`, `nccl-cu13`, `nvjitlink`,
  `nvshmem-cu13`, `nvtx`) plus `cuda-toolkit`, `cuda-bindings`, `cuda-pathfinder` and `triton`.
  **Several GB of GPU runtime for a machine with no GPU.**
- The study venv currently contains **no torch at all**. Adding it to `.venv` is a large change to
  a frozen environment whose reproducibility is the point.
- Plus the checkpoint: `typed-decisions/model.safetensors` is **842.6 MB**
  (`english` 842.6 MB, `multilingual` 643.8 MB), downloaded from HF on first use.

This is why the implementation ships the layer as a **separate optional environment**
(`requirements-laya.txt` / `make setup-laya`) instead of a hard dependency.
With the CPU wheel index, the isolated venv I actually built is **1.2 GB** total.

### 2.7 Two smaller design criticisms

- **`score_rationale()` should not return LAR.** §8 says the function "now also returns" it. That
  couples a cheap exact-match scorer to a model call, and would make the `verify` stage — currently
  seconds over 6 arms × 50 samples — take hours. LAR belongs in a stage that reads the *persisted*
  artifacts, which is how it is implemented (`outputs/laya/`).
- **Agreement is not accuracy.** Two checkers agreeing (LAR = 1.0) is also what you observe when
  both are wrong in the same direction; §2.2 is precisely how that happens. Without independent
  labels on the disagreement cases, LAR is not falsifiable. The repo does have a way to make it
  falsifiable — the 90 direction labels in `data/eval/extractor_gold.jsonl` — and that is the
  measurement to run first (§5).

## 3. Per-integration-point verdict: is it needed?

| # | integration point | needed? | why |
|---|---|---|---|
| 1 | Pre-generation verdict gate | **No** | Classifier correctness is already measured on a held-out group-aware split (ROC-AUC 0.9273, PR-AUC in `models/metrics.json`). "Is this verdict suspicious?" is answered by the calibrated probability and the near-threshold warning the CLI already prints. A below-baseline zero-shot model adds no information, and there is no independent label set with which to validate it. |
| 2 | Claim-level cross-judge | **Not as proposed; interesting in one narrower form** | As an *agreement metric* it is unfalsifiable (§2.7). As an *extractor* evaluated against the 90 gold direction labels it is genuinely interesting — and cheaper than fine-tuning: run the zero-shot checkpoint as an extractor on the gold rationales and compare its P/R against the rule extractor's 0.43/0.22 (P0) and 0.95/0.95 (P2). If it beats those, it earns a place. If not, the whole layer is settled. |
| 3 | Repair-loop reviewer | **No** | Already strictly dominated. The repair loop re-scores the repaired rationale with the exact-match verifier, so "did the repair fix the flagged claim?" *is* already measured — that is why the report shows S2R FS 1.000 and P0R FS 0.891. "Did it look reworded?" is a strictly weaker question than "does the flagged claim now pass exact match?". |
| 4 | Fine-tuning (the proposal's "required step") | **Blocked, not just unnecessary** | Both targets circular (§2.2); the only non-circular labels all sit in the test split (§2.3). There is no version of §9 that can be run today without either training on the eval set or training the checker to imitate the thing it checks. |

**Bottom line:** for the study's claims, this layer is not needed. It becomes worth revisiting
under exactly one condition — a genuinely independent label source (human-annotated claims and
verdicts, or a held-out corpus never touched by the study) — and even then its first job should be
`extract_rules` comparison, not gating the pipeline.

## 4. What I implemented instead

The safe subset of the proposal, with the three deviations above corrected:

| file | what |
|---|---|
| [src/faithfulness/laya_gate.py](src/faithfulness/laya_gate.py) | the whole layer: `gate_verdict`, `judge_claims`, `review_repair`, temperature loading, and the `make laya_gate` stage. Owns the codebase's **only** `laya` import. |
| [configs/default.yaml](configs/default.yaml) | `laya:` block with **`enabled: false`** (`preload: false`, `device: cpu`, `gate_confidence_threshold: 0.70`). |
| [src/cli.py](src/cli.py) | `--decide laya` on `check` and `batch` — **opt-in**, not default-on; adds `laya_verdict_agree` / `laya_claim_agreement_rate` columns to the batch CSV. |
| [requirements-laya.txt](requirements-laya.txt) | the optional extra, with the CPU-torch recipe and the size warning. |
| [scripts/make_laya_finetune_set.py](scripts/make_laya_finetune_set.py) | the §9 dataset builder, corrected: default `--independent-only`, `--allow-circular` to opt into the circular bulk, and `independent` / `label_provenance` / `circular_for` on every row so a trainer cannot use them by accident. |
| [tests/test_laya_gate.py](tests/test_laya_gate.py) | 18 tests pinning the soft-fail contract, FS independence and the no-clobber rules for a partial run. |
| [Makefile](Makefile) | `make setup-laya` and `make laya_gate`; `laya_gate` is in `make all` but **not** a prerequisite of `validity`/`eval` (the proposal has it as a hard dependency — on CPU that would add hours). |

Deliberate deviations, each with a reason:

1. **`enabled: false` by default**, not `true` — §2.4; below-baseline zero-shot must not be default-on.
2. **Separate stage, not inside `score_rationale()`** — §2.7.
3. **Separate optional environment**, not a hard dependency — §2.6. The study venv stays torch-free.
4. **`laya_gate` not a prerequisite of `validity`/`eval`** — the proposal's dependency graph would
   put a multi-hour CPU stage inside `make all`.
5. **`gate_confidence_threshold` and the temperature file are wired but left at the proposal's
   values until a checkpoint is calibrated on this domain** — fitting them on circular labels (§2.2)
   would bake the circularity in. §4.1 shows the effect of the shipped 0.70 default rather than
   quietly tuning it to fit.
6. **Integration point 3 (repair reviewer) is not wired into the repair loop.** `review_repair()`
   exists and is tested, but the loop already re-scores each repaired rationale with the
   exact-match verifier — a strictly stronger question than "does it still look reworded?"
   (S2R FS 1.000, P0R FS 0.891). Wiring a ~2 s model call into `generate.py` would buy a weaker
   answer at a real cost.

### The contract that makes it safe

Four acceptance checks in `outputs/phase_checks.json` (all **PASS**, `make checks` = 15/16, the one
FAIL being the pre-existing documented ROC-AUC row):

| check | what it proves |
|---|---|
| `laya_available_fallback` | the pipeline behaves identically with the layer off/absent |
| `laya_fs_independence` | FS recomputed from every stored rationale is **byte-identical** with and without Laya fields on the claims — the §8 separation holds *in code*, not just in the formula |
| `laya_gate_latency` | records the true median gate latency against the proposal's 100 ms CPU budget |
| `laya_full_run_recorded` | the full six-arm × 50-sample reference run is preserved in `summary_full_run.json`, so a later partial invocation overwriting `summary.json` cannot erase it |

Measured with the layer absent (the default path):

```text
$ make laya_gate
laya_gate skipped: cfg.laya.enabled=false (opt in with `make setup-laya` ...)
MAKE_EXIT=0
$ md5sum outputs/faithfulness/*.jsonl outputs/validity.json     # before and after
FS ARTIFACTS BYTE-IDENTICAL: yes
$ make laya_gate && .venv/bin/python -m pytest tests/ -q
46 passed
$ make checks
**15/16 checks passed.**        # the one FAIL is the pre-existing documented ROC-AUC row
```

And at full scale (all 7 arms, all 50 eval samples, 350 stored rationales), against the real
`laya_status` field rather than a synthetic one:

```text
fs_independence: PASS | FS byte-identical with and without Laya fields on all 350 stored rationale(s)
stored FS reproduced exactly from rebuilt claims: 350 records, 0 mismatches
```

The rebuilt-claims half of that matters on its own: it says the stage's reconstruction of a
rationale from `outputs/faithfulness/*.jsonl` round-trips exactly, so the cross-judge is judging
the real claim texts and not an approximation.

Measured with the layer live is in §4.1.

### 4.1 Live run (isolated CPU venv) — measured

Setup: `laya` 0.4.2, `torch 2.14.1+cpu`, checkpoint `typed-decisions`, `device: cpu`,
**no GPU** on this host — installed either into the study venv (`make setup-laya`) or into the
isolated `.venv-laya` (1.2 GB). Two runs, both raw on disk:

| run | scope | artifact |
|---|---|---|
| probe | 3 arms × 6 samples (`--force --limit 6`) | [outputs/laya/summary_live_run.json](outputs/laya/summary_live_run.json) |
| **reference** | **6 arms × 50 samples** (`FORCE=1 make laya_gate`) | [outputs/laya/summary_full_run.json](outputs/laya/summary_full_run.json) |

plus [outputs/laya/disagreements.md](outputs/laya/disagreements.md) and the per-arm
`outputs/laya/*_gate.jsonl` / `*_claims.jsonl` for both.

**Latency — the proposal's budget versus this CPU**

| call | proposal says | measured here |
|---|---|---|
| gate (1 per sample) | "33 ms" (T4) / "193–464 ms" (CPU) | median **4,649 ms** (350 calls, all 7 arms × 50 samples) |
| one claim (≈5.5 per sample per arm) | same | median **1,868 ms** (range 1,036–4,807 ms over 350 samples) |

So `python -m src.cli check --decide laya` costs ≈ 4.6 s + 6 × 1.9 s ≈ **16 s per sample**, against
§4's "well under 100 ms". The proposed acceptance check (`median gate call ≤ 100 ms on CPU`) flips
to **FAIL** the moment the layer is enabled; it is recorded as "not applicable" in the shipped
default configuration and will report the real number if you turn it on. The six study arms took
**55 min wall** (03:15→04:10) and the `X` control another 20 min on this host, which is why
`laya_gate` is deliberately not a prerequisite of `validity`/`eval`.

**Full-run agreement — what 350 gate calls and 1,914 claim judgements show**

| arm | gate agreement | LAR (strict) | claims compared |
|---|---|---|---|
| `T` | 1.000 (0 hard disagreements) | 0.997 (0.997) | 300 |
| `P0` | 1.000 | 0.931 (0.903) | 180 |
| `P0R` | 1.000 | 0.990 (0.985) | 234 |
| `P2` | 1.000 | 0.953 (0.953) | 300 |
| `S2` | 1.000 | 1.000 (1.000) | 300 |
| `S2R` | 1.000 | 1.000 (1.000) | 300 |
| `X` (control) | 1.000 | **0.387 (0.133)** | 300 |

The gate column is `unclear` on every one of the 350 calls (see below), so
`gate_agreement_rate` = 1.000 is counted, not earned. The one number that moves is `P0` — the arm
where the verifier has real negatives to find — and it is still Laya agreeing with the verifier
93% of the time while missing 94% of the corrupted arm's contradictions (§2.7).

**The proposed gate threshold cannot fire**

| signal | range over all 350 gate calls | proposal |
|---|---|---|
| `confidence` | 0.008 – 0.198 (median 0.072) | gate on this; `gate_confidence_threshold: 0.70` |
| `top_option_prob` / `answer_confidence` | 0.388 – 0.657 (median 0.520) | — |

All 350 gate calls collapse to `unclear`, so `gate_agreement_rate` = 1.000 **by construction**: the
layer cannot disagree with the classifier because it never commits to a verdict. It is worth
stating what the threshold is hiding: the *ungated* argmax label matches the classifier on only
**266 of 350** calls (77× it says `ransomware` where the classifier says `benign`, 7× the
reverse) — the raw signal disagrees 24% of the time, and the 0.70 gate converts that into a
permanent abstention rather than into a usable warning.

§1's advice to
"gate on `confidence` rather than `action.act_probability`" is the problem — on this checkpoint the
two live on different scales and 0.70 sits on neither (max observed `confidence` is 0.198). Laya warns at load time, unprompted:

```text
RuntimeWarning: laya: this checkpoint ships invalid temperatures or values outside [0.5, 5];
using choice:11+=0.10058280825614929 -> 0.5. Treat confidence from the affected entries as uncalibrated.
```

That is upstream contradicting §"Mechanics" ("ECE drops from 0.466 to 0.081 after temperature
fitting") for the very checkpoint the interim plan depends on.

**The claim cross-judge, on the one arm where it has something to find**

On arm `X` (corrupted-evidence control: a faithful rationale paired with another sample's
evidence), at full scale — 50 samples, 300 claims — the verifier flags **257 of 300** claims:
83 `contradicted`, 174 `out_of_topk`. That is the setting where an independent checker earns
its place:

| verifier \ Laya | supported | not_in_evidence | contradicted |
|---|---|---|---|
| supported (43) | 35 | 8 | 0 |
| contradicted (83) | **53** | 25 | **5** |
| out_of_topk (174) | **98** | 76 | 0 |

- **Recall on `contradicted` claims: 5/83 = 0.06.** Laya called 53 of the 83 corrupted claims
  `supported` and never once said `contradicted` on an `out_of_topk` claim.
- Its one useful behaviour is `not_in_evidence`: of its 109 such answers, 76 land on an
  `out_of_topk` claim (70%) — i.e. it notices the feature is not among the ranked evidence,
  which the verifier already computes exactly and trivially (`feature_exists`).
- **It judged 186 of 300 claims on a fully corrupted arm `supported` (62%).**
- Full-run agreement: `S2`/`S2R`/`T` LAR **1.000** (degenerate — the verifier has zero
  negatives there and Laya answers `supported`), `P0` **0.931**, `P0R` 0.990, `P2` 0.953,
  `X` **0.387** (strict 0.133).

This is §2.7 in its empirical form. LAR 1.000 on the faithful arms and 0.387 on the corrupted arm
measure how often two checkers coincide, not whether either is right — and on the arm where the
question is decidable by construction, the zero-shot cross-judge misses **94%** of the real
failures. A 6%-recall checker cannot deliver the "cheap recall check on the rules extractor" that
§5 promises.

**The one place the idea earns its keep**

On `test_031534` — a ransomware (Maze) sample the classifier **missed** (p = 0.054 → `benign`) —
Laya read the same evidence table as `ransomware` (top option 0.542). That is exactly the
"flag a suspicious verdict before spending LLM tokens on it" case, and it worked. It is the
strongest evidence for the gate *concept* in this whole exercise — and simultaneously the clearest
argument against §9, because training the gate on the classifier's verdict is training away
precisely the disagreement that just caught a real miss.

### 4.2 Reproducing this review

```bash
# default path — no Laya needed, proves the soft-fail contract
make test && make checks && python scripts/check_docs.py
make laya_gate                                  # no-op, exit 0

# live path — isolated CPU venv, ~1.2 GB, plus an 843 MB checkpoint download
python3 -m venv .venv-laya
.venv-laya/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-laya/bin/pip install -r requirements-laya.txt
HF_HUB_CACHE=/tmp/hf_laya .venv-laya/bin/python -m src.faithfulness.laya_gate \
    --force --arms S2,P0,X --limit 6          # ~6 min on CPU

# what fine-tuning data actually exists
python scripts/make_laya_finetune_set.py                  # 0 gate rows, 90 claim rows
python scripts/make_laya_finetune_set.py --allow-circular  # 50 gate, 1614 claim rows
```

## 5. If you want to make this pay for itself

In priority order, and none of it requires the fine-tuning the proposal calls mandatory:

1. **Run Laya zero-shot as an extractor on the 90 gold direction labels** and report P/R next to
   the rule extractor's 0.43/0.22 (P0) and 0.95/0.95 (P2). One number decides whether the layer has
   any future. The harness already exists — `extraction_accuracy()` in
   [src/evaluation/validity.py](../src/evaluation/validity.py).
2. **Collect independent labels** — even 100 hand-annotated `(evidence row, direction/level/value)`
   triples from the *train* split would give the cross-judge a non-circular target and a legitimate
   train/val/test split. Until then `scripts/make_laya_finetune_set.py --allow-circular` is an
   accurate description of what is available, not a dataset.
3. **Only then** consider fine-tuning, and only for the *extractor* role — not for gating the
   pipeline, which is the one place the proposal would let a suggestion change what the study
   reports.

## 6. Limits of this review

- I did not run the fine-tuning (§9) because §2.2/§2.3 show it cannot be run validly; the
  conclusion is from the label provenance, not from a training experiment.
- The circularity argument applies to *training on* these labels. It says nothing about whether
  Laya's zero-shot checkpoint happens to be useful — that is exactly §5's open question, and it is
  testable without any of the above.
- The measured latency/agreement figures in §4.1 are **one host, CPU, `typed-decisions` zero-shot**.
  A GPU and a fine-tuned checkpoint would change them; they would not change §2.2.
- Laya's own published accuracy figures are taken from the checkpoint card and the proposal; I did
  not re-run its typed-decisions benchmark.
