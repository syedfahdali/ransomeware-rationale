"""Group-aware stratified 70/10/20 split (spec section 3.3). Seed 42.

Malware capture groups (`group_id`, from the Category filename) never cross
splits (10 rows per capture would otherwise leak). Benign rows have no capture
ids in this mirror and are treated as singleton groups (row-level stratified).

Outputs: data/processed/{train,val,test}.parquet + splits.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.utils.config import Paths, ensure_dirs, load_config
from src.utils.logging import setup_logger
from src.utils.seed import set_seed

log = setup_logger("split")


def split_groups(group_ids: np.ndarray, frac_train: float, frac_val: float,
                 seed: int) -> tuple[set, set, set]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(group_ids))
    n = len(idx)
    n_train = int(round(frac_train * n))
    n_val = int(round(frac_val * n))
    train = set(np.asarray(group_ids)[idx[:n_train]])
    val = set(np.asarray(group_ids)[idx[n_train:n_train + n_val]])
    test = set(np.asarray(group_ids)[idx[n_train + n_val:]])
    return train, val, test


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config()
    set_seed(int(cfg.seed))

    all_p = Paths.processed / "all.parquet"
    train_p, val_p, test_p = (Paths.processed / f"{s}.parquet" for s in ("train", "val", "test"))
    if all(p.exists() for p in (train_p, val_p, test_p)) and not args.force:
        log.info("splits already exist (use --force to redo)")
        return 0
    if not all_p.exists():
        raise FileNotFoundError("run preprocess.py first")

    df = pd.read_parquet(all_p)
    feats = json.loads((Paths.processed / "features.json").read_text())

    groups = df.groupby("group_id")["y_binary"].nunique()
    if (groups > 1).any():
        bad = groups[groups > 1].index.tolist()
        raise ValueError(f"groups with mixed labels: {bad[:5]}")
    gtab = df[["group_id", "y_binary"]].drop_duplicates("group_id")

    tr_g, va_g, te_g = set(), set(), set()
    for y in (0, 1):
        gids = gtab.loc[gtab["y_binary"] == y, "group_id"].to_numpy()
        a, b, c = split_groups(gids, cfg.split.train, cfg.split.val,
                               int(cfg.seed) + 1000 * int(y))
        tr_g |= a
        va_g |= b
        te_g |= c

    parts = {}
    for name, gset in (("train", tr_g), ("val", va_g), ("test", te_g)):
        part = df[df["group_id"].isin(gset)].sort_values("row_uid").reset_index(drop=True)
        part.to_parquet(Paths.processed / f"{name}.parquet", index=False)
        parts[name] = part
        log.info("%s: %d rows, %d groups, %.1f%% ransomware", name, len(part),
                 part["group_id"].nunique(), 100 * part["y_binary"].mean())

    # leakage checks
    overlap = {"train_val": len(tr_g & va_g), "train_test": len(tr_g & te_g),
               "val_test": len(va_g & te_g)}
    dup_overlap = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ka = set(map(tuple, parts[a][feats].round(6).values.tolist()))
        kb = set(map(tuple, parts[b][feats].round(6).values.tolist()))
        dup_overlap[f"{a}_{b}"] = len(ka & kb)
    if any(overlap.values()) or any(dup_overlap.values()):
        log.warning("leakage check: group overlap %s, duplicate rows %s", overlap, dup_overlap)
    else:
        log.info("leakage check OK: no group overlap, no identical rows across splits")

    meta = {
        "seed": int(cfg.seed),
        "features": len(feats),
        "group_aware": True,
        "benign_grouping": "singleton (no capture ids for benign rows in this mirror)",
        "leak_check": {"group_overlap": overlap, "identical_rows": dup_overlap},
        "rows": {k: len(v) for k, v in parts.items()} | {"all": len(df)},
        "groups": {"train": len(tr_g), "val": len(va_g), "test": len(te_g)},
        "ransomware_rate": {k: float(v["y_binary"].mean()) for k, v in parts.items()},
        "families_test": parts["test"].loc[parts["test"]["y_binary"] == 1, "family"]
                             .value_counts().to_dict(),
    }
    (Paths.processed / "splits.json").write_text(json.dumps(meta, indent=2))
    log.info("split metadata: %s", json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
