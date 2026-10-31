"""Clean / dedupe / label CIC-MalMem-2022 (spec sections 3.1, 3.3).

This mirror of the dataset labels every malware row with its source capture
filename, e.g. `Ransomware-Shade-<hash>-7.raw` or `Trojan-Emotet-<hash>-3.raw`.
That gives us:

- `family`      (Shade, Maze, Conti, Pysa, Ako, Emotet, Transponder, ...)
- `malware_class` (Ransomware / Spyware / Trojan / Downloader)
- `capture_id`  (the 32/64-hex capture hash — ~10 rows per capture)
- `group_id`    (capture id for malware; unique per row for benign, because the
                 benign rows carry no capture ids)

Outputs
-------
data/processed/all.parquet, data/processed/features.json, outputs/data_summary.md
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.utils.config import Paths, ensure_dirs, now_iso
from src.utils.logging import setup_logger
from src.utils.seed import set_seed

log = setup_logger("preprocess")

NON_FEATURE = {"category", "class", "unnamed: 0", "index", "row_uid", "y_binary",
               "y_3class", "family", "malware_class", "capture_id", "group_id"}


def find_raw_file() -> str:
    target = Paths.raw / "CIC-MalMem-2022.csv"
    candidates = [target] if target.exists() else []
    candidates += sorted(Paths.raw.rglob("*.csv"))
    for p in candidates:
        try:
            n = sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore")) - 1
        except Exception:  # noqa: BLE001
            continue
        if n >= 50_000:
            return str(p)
    raise FileNotFoundError(
        f"no CSV with >=50k rows under {Paths.raw}; run src/data/download.py first")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    set_seed(42)

    out = Paths.processed / "all.parquet"
    if out.exists() and not args.force:
        log.info("already processed: %s (use --force to redo)", out)
        return 0

    raw_path = find_raw_file()
    log.info("reading %s", raw_path)
    df = pd.read_csv(raw_path, low_memory=False)
    n_raw = len(df)

    df.columns = [str(c).strip() for c in df.columns]
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if drop_cols:
        log.info("dropping index columns: %s", drop_cols)
        df = df.drop(columns=drop_cols)

    cols_lower = {c.lower(): c for c in df.columns}
    cat_col = cols_lower.get("category")
    cls_col = cols_lower.get("class")
    if cls_col is None and cat_col is None:
        raise ValueError("no Class/Category label column found")
    if cat_col is None:
        cat_col = cls_col
        df[cat_col] = df[cls_col]
    if cls_col is None:
        cls_col = cat_col

    features = [c for c in df.columns if c.lower() not in NON_FEATURE]
    log.info("features: %d", len(features))
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    n_nan_rows = int(df[features].isna().any(axis=1).sum())
    if n_nan_rows:
        log.warning("dropping %d rows with NaN features", n_nan_rows)
        df = df[~df[features].isna().any(axis=1)].copy()

    n_before = len(df)
    df = df.drop_duplicates(subset=features, keep="first").reset_index(drop=True)
    n_dupes = n_before - len(df)
    log.info("dropped %d exact duplicate rows (%d -> %d)", n_dupes, n_before, len(df))

    # ---- category parsing -------------------------------------------------- #
    cat = df[cat_col].astype(str).str.strip()
    is_benign = cat.str.lower().str.startswith("benign")
    is_ransomware = cat.str.lower().str.contains("ransomware")
    tokens = cat.str.split("-")
    malware_class = np.where(is_benign, "Benign", tokens.str[0])
    family_tok = np.where(tokens.str.len() > 1, tokens.str[1].fillna("none"), "none")
    hash_m = cat.str.extract(r"-([0-9a-fA-F]{32,64})-\d+\.raw\s*$")[0]
    capture_id = hash_m.str.lower()

    df["y_binary"] = is_ransomware.astype(int)
    df["y_3class"] = np.where(is_benign, "benign",
                              np.where(is_ransomware, "ransomware", "other_malware"))
    df["family"] = np.where(is_benign, "none", family_tok)
    df["malware_class"] = malware_class
    df["capture_id"] = capture_id
    df["row_uid"] = [f"row_{i:06d}" for i in range(len(df))]
    df["group_id"] = np.where(
        df["capture_id"].notna(),
        df["malware_class"].astype(str) + ":" + df["family"].astype(str) + ":" + df["capture_id"].astype(str),
        "singleton:" + df["row_uid"])

    df.to_parquet(out, index=False)
    (Paths.processed / "features.json").write_text(json.dumps(features, indent=2))

    # ---- summary ----------------------------------------------------------- #
    n_hashes = df.loc[df["capture_id"].notna(), "capture_id"].nunique()
    per_cap = df[df["capture_id"].notna()].groupby("capture_id").size()
    mal_no_hash = int(((df["malware_class"] != "Benign") & df["capture_id"].isna()).sum())
    lines = [
        "# Data summary — CIC-MalMem-2022",
        "",
        f"- Raw file: `{raw_path}`",
        f"- Raw rows: {n_raw}",
        f"- Rows dropped for NaN features: {n_nan_rows}",
        f"- Exact duplicate rows dropped: {n_dupes}",
        f"- Final rows: {len(df)}",
        f"- Features: {len(features)}",
        f"- Generated: {now_iso()}",
        "",
        "## Leakage structure (important)",
        "",
        f"- The Category column encodes the source memory capture of every malware row "
        f"(e.g. `Ransomware-Shade-<hash>-7.raw`).",
        f"- Distinct malware captures (hashes): {n_hashes}; rows per capture: "
        f"mean {per_cap.mean():.1f}, median {per_cap.median():.0f}, max {per_cap.max()}.",
        f"- Malware rows without a capture hash: {mal_no_hash}.",
        "- Malware rows from the same capture are kept in ONE split via group-aware "
        "splitting (`group_id`).",
        "- Benign rows carry NO capture ids in this mirror; they are split row-level. "
        "Near-duplicate benign dumps may therefore span splits — documented limitation.",
        "",
        "## `Class` column",
        "", "```", df[cls_col].astype(str).value_counts().to_string(), "```", "",
        "## Binary task (`y_binary`: ransomware vs rest)",
        "", "```",
        df["y_binary"].value_counts().rename({0: "not_ransomware", 1: "ransomware"}).to_string(),
        "```", "",
        "## 3-class task",
        "", "```", df["y_3class"].value_counts().to_string(), "```", "",
        "## Malware classes (non-benign rows)",
        "", "```",
        df.loc[df["malware_class"] != "Benign", "malware_class"].value_counts().to_string(),
        "```", "",
        "## Ransomware families",
        "", "```", df.loc[df["y_binary"] == 1, "family"].value_counts().to_string(),
        "```", "",
        "## Other-malware families",
        "", "```",
        df.loc[df["y_3class"] == "other_malware", "family"].value_counts().head(30).to_string(),
        "```", "",
    ]
    (Paths.outputs / "data_summary.md").write_text("\n".join(lines))
    log.info("wrote %s and %s", out, Paths.outputs / "data_summary.md")
    log.info("binary counts:\n%s", df["y_binary"].value_counts().to_string())
    log.info("ransomware families:\n%s",
             df.loc[df["y_binary"] == 1, "family"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
