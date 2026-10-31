"""Plots for the evaluation report (matplotlib, Agg backend)."""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.utils.config import Paths, ensure_dirs
from src.utils.logging import setup_logger  # noqa: E402

log = setup_logger("plots")

ARM_ORDER = ["T", "P0", "P0R", "P2", "S2", "S2R", "X"]
COLORS = ["#4d4d4d", "#c0392b", "#e74c3c", "#e67e22", "#2980b9", "#27ae60", "#8e44ad"]


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(Paths.figs / name, dpi=140)
    plt.close(fig)
    log.info("wrote %s", Paths.figs / name)


def fs_by_arm(stats: dict) -> None:
    arms = [a for a in ARM_ORDER if a in stats["arms"]]
    means = [stats["arms"][a]["FS"]["mean"] for a in arms]
    errs = [[m - stats["arms"][a]["FS"]["ci_low"] for m, a in zip(means, arms)],
            [stats["arms"][a]["FS"]["ci_high"] - m for m, a in zip(means, arms)]]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(arms, means, yerr=errs, capsize=4, color=COLORS[:len(arms)])
    ax.set_ylabel("Composite Faithfulness Score (mean, 95% CI)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Faithfulness by arm")
    _save(fig, "fs_by_arm.png")


def components_by_arm(stats: dict) -> None:
    arms = [a for a in ARM_ORDER if a in stats["arms"]]
    comps = [("claim_precision", "claim precision"), ("hallucination_rate", "hallucination rate"),
             ("contradiction_rate", "contradiction rate"), ("evidence_coverage", "coverage")]
    x = np.arange(len(arms))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for i, (key, label) in enumerate(comps):
        vals = [stats["arms"][a][key]["mean"] for a in arms]
        ax.bar(x + (i - 1.5) * w, vals, w, label=label)
    ax.set_xticks(x, arms)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Faithfulness components (mean)")
    _save(fig, "components_by_arm.png")


def _repair_pairs(arm: str) -> list[tuple[float, float]]:
    p = Paths.rationales / f"{arm}.jsonl"
    if not p.exists():
        return []
    pairs = []
    with p.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("skipped"):
                continue
            pairs.append((float(rec["fs_before"]), float(rec["fs_final"])))
    return pairs


def repair_before_after() -> None:
    s2r = _repair_pairs("S2R")
    p0r = _repair_pairs("P0R")
    if not s2r and not p0r:
        return
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.plot([0, 1.02], [0, 1.02], "k--", lw=1, label="no change")
    for pairs, label, color, marker in ((s2r, "S2R (structured)", "#2980b9", "o"),
                                        (p0r, "P0R (prose)", "#e74c3c", "^")):
        if not pairs:
            continue
        jitter = np.random.RandomState(0).uniform(-0.012, 0.012, len(pairs))
        xs = np.array([b for b, _ in pairs]) + jitter
        ys = np.array([a for _, a in pairs]) + jitter
        ax.scatter(xs, ys, s=18, alpha=0.75, label=label, color=color, marker=marker)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("FS before repair")
    ax.set_ylabel("FS after repair")
    ax.set_title("Verify-and-repair loop effect (RQ2)")
    ax.legend(fontsize=8)
    _save(fig, "repair_before_after.png")


def fluency_vs_fs() -> None:
    p = Paths.outputs / "validity.json"
    if not p.exists():
        return
    v = json.loads(p.read_text())
    fl = v.get("fluency") or {}
    rows = fl.get("rows") or []
    rows_x = fl.get("rows_corrupted") or []
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter([r["n_tokens"] for r in rows], [r["FS"] for r in rows],
               s=16, label="original (S2)", alpha=0.8)
    if rows_x:
        ax.scatter([r["n_tokens"] for r in rows_x], [r["FS"] for r in rows_x],
                   s=16, label="corrupted evidence (X)", alpha=0.8, marker="x")
    ax.set_xlabel("rationale length (tokens)")
    ax.set_ylabel("FS")
    ax.set_title("Fluency proxy vs faithfulness")
    ax.legend(fontsize=8)
    _save(fig, "fluency_vs_fs.png")


def corrupted_vs_original() -> None:
    p = Paths.outputs / "validity.json"
    if not p.exists():
        return
    v = json.loads(p.read_text()).get("corrupted_evidence_control")
    if not v:
        return
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.bar(["original (S2)", "mismatched evidence (X)"],
           [v["FS_original_S2"], v["FS_corrupted_X"]], color=["#2980b9", "#8e44ad"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("FS (mean)")
    ax.set_title("Corrupted-evidence control")
    _save(fig, "corrupted_vs_original.png")


def main() -> int:
    ensure_dirs()
    stats_p = Paths.faithfulness / "stats.json"
    if stats_p.exists():
        stats = json.loads(stats_p.read_text())
        fs_by_arm(stats)
        components_by_arm(stats)
    repair_before_after()
    fluency_vs_fs()
    corrupted_vs_original()
    return 0


if __name__ == "__main__":
    sys.exit(main())
