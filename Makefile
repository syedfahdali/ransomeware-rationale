SHELL := /bin/bash
VENV  := .venv
PY    := $(VENV)/bin/python
FORCE ?= 0
FORCEARG := $(if $(filter 1,$(FORCE)),--force,)

# everyday-triage knobs (see `make triage`)
CSV    ?= data/raw/CIC-MalMem-2022.csv
STRIDE ?= 1500
LIMIT  ?= 30
SAMPLE ?= test_003028
ARGS   ?=

.PHONY: setup setup-laya download preprocess train evidence generate verify validity laya_gate
.PHONY: eval test checks docs brand notebooks cli check triage all clean help

help:
	@echo "pipeline : setup download preprocess train evidence generate verify validity eval test checks docs all"
	@echo "optional : setup-laya laya_gate (diagnostic Laya layer, off by default)"
	@echo "assets   : brand (logo + banner), docs (demo GIFs + doc-link check)"
	@echo "everyday : cli (smoke) | check SAMPLE=<id> [ARGS=...] | triage [CSV=.. STRIDE=.. LIMIT=..] | notebooks"
	@echo "FORCE=1 redoes a finished stage"

setup: $(VENV)/.installed

$(VENV)/.installed:
	bash scripts/install.sh
	@touch $(VENV)/.installed

# OPTIONAL. Installs the Laya decision-layer extra into the study venv.
#
# torch is installed FIRST from a CPU-only index: the plain `pip install laya`
# also resolves the CUDA nvidia-* wheels, which is ~19 GPU packages (several GB)
# on a machine that will run this on CPU. On a GPU host, override the index:
#   make setup-laya TORCH_INDEX=https://download.pytorch.org/whl/cu126
# …or install your torch build first and run:
#   make setup-laya TORCH_INDEX=
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu
setup-laya:
ifneq ($(TORCH_INDEX),)
	$(VENV)/bin/pip install torch --index-url $(TORCH_INDEX)
endif
	$(VENV)/bin/pip install -r requirements-laya.txt
	@echo "laya installed — enable it with FORCE=1 make laya_gate, or set"
	@echo "laya.enabled: true in configs/default.yaml to include it in make all."
	@echo "checkpoints download to HF_HUB_CACHE (default ~/.cache/huggingface/hub)."

download: setup
	$(PY) -m src.data.download

preprocess: setup
	$(PY) -m src.data.preprocess $(FORCEARG)

train: setup
	$(PY) -m src.models.train $(FORCEARG)

evidence: setup
	$(PY) -m src.evidence.build_evidence $(FORCEARG)

# Smokes the everyday-triage CLI: one sample end to end, the info banner and a
# 5-row batch. Needs models/ and data/processed/ (i.e. `make train`) and
# data/raw/ for the batch input.
cli: setup
	$(PY) -m src.cli check --sample $$($(PY) -c "import json;print(json.load(open('data/eval/eval_ids.json'))['samples'][0]['sample_id'])") --json > /dev/null
	$(PY) -m src.cli info > /dev/null
	$(PY) -m src.cli batch --csv data/raw/CIC-MalMem-2022.csv --limit 5 --out outputs/triage_smoke.csv > /dev/null
	@echo "cli smoke ok"

# Everyday use: check one sample, and triage a CSV of samples.
check: setup
	$(PY) -m src.cli check --sample $(SAMPLE) $(ARGS)

triage: setup
	$(PY) -m src.cli batch --csv $(CSV) --stride $(STRIDE) --limit $(LIMIT)

notebooks: setup
	$(PY) scripts/run_notebooks.py

# May exit 20 (WAITING_FOR_AGENT) in agent_queue mode: answer queue/requests.jsonl
# into queue/responses.jsonl, then re-run the same command.
generate: setup
	$(PY) -m src.generation.generate --arms T,P0,P0R,P2,S2,S2R --yes $(FORCEARG)

verify: setup
	$(PY) -m src.evaluation.run_arms $(FORCEARG)

validity: setup
	$(PY) -m src.evaluation.validity $(FORCEARG)
	$(PY) -m pytest tests/test_metric_validity.py -q

# Optional Laya decision-layer cross-check (docs/LAYA_REVIEW.md). Reads what
# `verify` wrote and writes outputs/laya/. A no-op unless laya.enabled: true
# (or --force) AND the optional extra is installed, so `make all` can run it
# unconditionally. Deliberately NOT a prerequisite of validity/eval: on CPU the
# layer costs ~200-500 ms per call, which would add hours to a full study run.
laya_gate: setup
	$(PY) -m src.faithfulness.laya_gate $(FORCEARG)

eval: setup
	$(PY) -m src.evaluation.stats
	$(PY) -m src.evaluation.plots
	$(PY) -m src.evaluation.report_md

test: setup
	$(PY) -m pytest -q

checks:
	$(PY) -m src.utils.checks
	$(PY) scripts/check_docs.py

docs: brand
	$(PY) scripts/check_docs.py
	$(PY) scripts/make_demo_gif.py

# Regenerates docs/logo.svg, docs/logo.png, docs/banner.png and docs/banner.gif.
# The banner reads the real per-arm FS from outputs/faithfulness/all_arms_summary.json.
brand:
	$(PY) scripts/make_brand_assets.py

all: setup download preprocess train evidence cli generate verify laya_gate validity eval notebooks test checks
	@echo "make all finished"
