#!/usr/bin/env bash
# Project setup: venv + dependency install. Safe to re-run.
set -x
cd "$(dirname "$0")/.."
if [ ! -d .venv ]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install numpy pandas scikit-learn xgboost pyyaml jsonschema requests httpx tenacity tqdm joblib matplotlib scipy pytest python-dotenv
echo "=== CORE DONE rc=$? ==="
.venv/bin/pip install shap lightgbm kaggle beautifulsoup4 huggingface_hub
echo "=== EXTRAS DONE rc=$? ==="
.venv/bin/pip freeze > requirements.lock.txt
echo "=== ALL DONE ==="
df -h "$HOME" | tail -1
