"""Build Laya fine-tuning sets from this repo's own artifacts — and show,
honestly, why the resulting labels mostly cannot be used.

Proposal section 9 of `Laya Decision-Layer Architecture Overhaul.md` suggests two
typed-decision sets: (1) ``(evidence table) -> classifier verdict`` for the
pre-generation gate, and (2) ``(claim text, evidence row) -> status`` for the
claim cross-judge, harvested from ``outputs/faithfulness/*.jsonl``.

The problem this script makes concrete:

* the **gate** target is the classifier's own verdict. Training Laya to predict
  the classifier's verdict optimises *agreement with the classifier*, which is
  exactly the signal a "gate that flags verdicts the evidence contradicts" is
  supposed to detect. A gate fine-tuned this way cannot flag anything -- it has
  been taught to reproduce the thing it is checking.
* the **claim** target is the deterministic verifier's own status. Training the
  cross-judge on the verifier's labels teaches it to fail wherever the verifier
  fails, destroying the independence that is its only value.

Splitting by the classifier's `group_id` (which this script does, to be fair to
the proposal) prevents *train/test leakage*; it does nothing about *label
provenance circularity*. Those are different problems and only the first is
fixed by a split.

So the default behaviour is `--independent-only`: emit only rows whose label
comes from somewhere other than the component being cross-checked -- i.e. the
20 hand-checked rationales in `data/eval/extractor_gold.jsonl`. Run
`--allow-circular` to also emit the verifier/classifier-labelled bulk set, with
`"independent": false` on every such row so a training script can filter or
weight it explicitly.

Output (default `data/laya_finetune/`):

    gate_{train,val,test}.jsonl     typed-decision rows for task "gate"
    claims_{train,val,test}.jsonl   typed-decision rows for task "claims"
    manifest.json                   counts + provenance + the circularity note
    README.md                       what the numbers mean

Usage:

    python scripts/make_laya_finetune_set.py                    # independent rows only
    python scripts/make_laya_finetune_set.py --allow-circular   # + verifier-labelled bulk
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.faithfulness import laya_gate  # noqa: E402
from src.utils.config import Paths, ensure_dirs, load_config  # noqa: E402
from src.utils.logging import setup_logger  # noqa: E402

log = setup_logger("make_laya_finetune_set")

ARMS = ["T", "P0", "P0R", "P2", "S2", "S2R"]
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# sample -> split (the classifier's own capture-group-aware assignment)
# --------------------------------------------------------------------------- #
def load_split_map() -> dict[str, str]:
    """row_uid -> 'train'|'val'|'test', straight from the classifier's splits.

    Using the classifier's split (which is grouped by capture id) is what keeps
    the *proposal's* leakage concern handled: no capture contributes rows to two
    sides of the fine-tuning boundary, exactly as for the classifier itself.
    """
    out: dict[str, str] = {}
    for split in SPLITS:
        p = Paths.processed / f"{split}.parquet"
        if not p.exists():
            log.warning("missing %s -- `make preprocess` first", p)
            continue
        df = pd.read_parquet(p, columns=["row_uid"])
        for uid in df["row_uid"].astype(str):
            out[uid] = split
    return out


def sample_to_row_uid(sample_id: str) -> str:
    for prefix in ("test_", "val_", "train_"):
        if sample_id.startswith(prefix):
            return "row_" + sample_id[len(prefix):]
    return sample_id


def split_of(sample_id: str, row_uid: str, split_map: dict[str, str]) -> str:
    for key in (row_uid, sample_to_row_uid(sample_id)):
        if key in split_map:
            return split_map[key]
    return "unassigned"


# --------------------------------------------------------------------------- #
# rows
# --------------------------------------------------------------------------- #
def gate_rows(allow_circular: bool, split_map: dict[str, str]) -> list[dict]:
    from src.generation.prompts import render_evidence_table

    rows: list[dict] = []
    for sid, ev in sorted(laya_gate.load_evidence().items()):
        state = render_evidence_table(ev)
        rows.append({
            "task": "gate",
            "sample_id": ev.get("sample_id", sid),
            "state": state,
            "question": laya_gate.gate_question(),
            "label": str(ev["verdict"]),
            # The gate must be able to DISAGREE with the classifier; a label
            # that IS the classifier's verdict cannot teach that.
            "label_provenance": "classifier_verdict",
            "independent": False,
            "circular_for": "gate",
            "split": split_of(str(ev.get("sample_id", sid)),
                              sample_to_row_uid(str(ev.get("sample_id", sid))), split_map),
        })
    if not allow_circular:
        return []
    return rows


def gold_index() -> dict[tuple[str, str], dict[str, str | None]]:
    """(sample_id, arm) -> {feature: gold direction-or-None}.

    The gold set is only 20 rationales (10 P0, 10 P2, 18 samples) and records a
    *direction* per feature, not a level or a value -- so a gold label answers
    "did the rationale point the right way?", nothing more. That scope is
    written onto every row so nobody over-reads it.
    """
    p = Paths.eval / "extractor_gold.jsonl"
    out: dict[tuple[str, str], dict[str, str | None]] = {}
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        key = (item["sample_id"], item.get("arm") or "P0")
        out[key] = {g["feature"]: g.get("direction") for g in item["gold_claims"]}
    return out


def claim_rows(allow_circular: bool, split_map: dict[str, str],
               stats: dict | None = None) -> list[dict]:
    stats = stats if stats is not None else {"gold_covered": 0, "gold_uncovered": 0,
                                           "gold_nondirectional": 0, "circular": 0}
    gold = gold_index()
    evidence = laya_gate.load_evidence()
    rows: list[dict] = []
    for arm in ARMS:
        records = laya_gate.load_jsonl(Paths.faithfulness / f"{arm}.jsonl")
        for sid, rec in sorted(records.items()):
            ev = evidence.get(sid)
            if ev is None:
                continue
            gold_for_this_rationale = gold.get((sid, arm))
            for c in rec["score"].get("per_claim", []):
                feature = str(c.get("feature"))
                verifier_label = laya_gate.collate_status(c.get("status"))
                state = (f"CLAIM about {feature}: {str(c.get('text') or '').strip()}\n\n"
                         f"EVIDENCE ROW:\n{laya_gate.render_claim_row(ev, feature)}")

                label, provenance, independent, scope = (
                    verifier_label, "rule_verifier", False,
                    "the deterministic verifier's own status")
                if gold_for_this_rationale is not None:
                    if feature in gold_for_this_rationale:
                        gdir = gold_for_this_rationale[feature]
                        if gdir is None:
                            # gold deliberately declines to commit on direction
                            stats["gold_nondirectional"] += 1
                        else:
                            label = ("supported" if gdir == c.get("stated_direction")
                                     else "contradicted")
                            provenance = "extractor_gold"
                            independent = True
                            scope = "direction only (gold records no level/value)"
                            stats["gold_covered"] += 1
                    else:
                        # absent from gold -> gold may simply be incomplete, so
                        # this is NOT used as an independent label
                        stats["gold_uncovered"] += 1
                if not independent:
                    stats["circular"] += 1

                rows.append({
                    "task": "claims",
                    "sample_id": sid,
                    "arm": arm,
                    "feature": feature,
                    "state": state,
                    "question": laya_gate.claim_question(),
                    "label": label,
                    "label_provenance": provenance,
                    "label_scope": scope,
                    "independent": bool(independent),
                    "circular_for": None if independent else "claim_cross_judge",
                    "verifier_label": verifier_label,
                    "verifier_status": c.get("status"),
                    "split": split_of(sid, sample_to_row_uid(sid), split_map),
                })
    if not allow_circular:
        rows = [r for r in rows if r["independent"]]
    return rows


# --------------------------------------------------------------------------- #
def write_split(rows: list[dict], out_dir: Path, task: str) -> dict:
    counts: dict[str, int] = {}
    for split in SPLITS + ("unassigned",):
        part = [r for r in rows if r["split"] == split]
        if not part:
            continue
        counts[split] = len(part)
        (out_dir / f"{task}_{split}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in part) + "\n")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(Paths.data / "laya_finetune"))
    ap.add_argument("--allow-circular", action="store_true",
                    help="also emit rows whose label is the component being "
                         "cross-checked (marked \"independent\": false)")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config()   # loads .env + makes dirs, same as every other script

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_map = load_split_map()

    claim_stats: dict = {"gold_covered": 0, "gold_uncovered": 0,
                         "gold_nondirectional": 0, "circular": 0}
    gate = gate_rows(args.allow_circular, split_map)
    claims = claim_rows(args.allow_circular, split_map, claim_stats)

    manifest = {
        "generated_for": "Laya fine-tuning (proposal section 9)",
        "independent_only": not args.allow_circular,
        "classifier_seed": int(cfg.seed),
        "split_rule": "the classifier's own capture-group-aware train/val/test split",
        "gate": {
            "rows": len(gate),
            "splits": write_split(gate, out_dir, "gate"),
            "label_provenance": dict(Counter(r["label_provenance"] for r in gate)),
            "independent_rows": sum(1 for r in gate if r["independent"]),
        },
        "claims": {
            "rows": len(claims),
            "splits": write_split(claims, out_dir, "claims"),
            "label_provenance": dict(Counter(r["label_provenance"] for r in claims)),
            "independent_rows": sum(1 for r in claims if r["independent"]),
            # counts over ALL six arms, before the independent-only filter
            "gold_covered_claims": claim_stats["gold_covered"],
            "gold_uncovered_claims": claim_stats["gold_uncovered"],
            "gold_nondirectional_claims": claim_stats["gold_nondirectional"],
            "circular_claims": claim_stats["circular"],
            "candidate_claims_scanned": (claim_stats["gold_covered"]
                                         + claim_stats["gold_uncovered"]
                                         + claim_stats["gold_nondirectional"]
                                         + claim_stats["circular"]),
        },
        "circularity_warning": (
            "Labels whose provenance is `classifier_verdict` (gate) or "
            "`rule_verifier` (claims) come from the very component the model is "
            "meant to cross-check. A cross-judge trained on them can only "
            "reproduce that component's errors; a gate trained on the "
            "classifier's verdict cannot disagree with the classifier. "
            "Group-aware splitting fixes train/test leakage, not label "
            "provenance circularity."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    (out_dir / "README.md").write_text(f"""# Laya fine-tuning sets

Built by `scripts/make_laya_finetune_set.py` from this repo's own artifacts.

| set | rows | independent labels | split |
|---|---|---|---|
| gate | {len(gate)} | {manifest['gate']['independent_rows']} | {manifest['gate']['splits']} |
| claims | {len(claims)} | {manifest['claims']['independent_rows']} | {manifest['claims']['splits']} |

`independent` = the label does **not** come from the component being
cross-checked. Rows with `"independent": false` carry
`"label_provenance": "classifier_verdict"` (gate) or `"rule_verifier"`
(claims) and `"circular_for"` naming the component they would teach the model
to imitate.

{manifest['circularity_warning']}

Run with `--allow-circular` to include the circular bulk set; the default emits
only the independent rows, which is why the gate set is empty unless you opt in.
""")

    log.info("gate rows: %s (independent %d)", manifest["gate"]["splits"],
             manifest["gate"]["independent_rows"])
    log.info("claim rows: %s (independent %d)", manifest["claims"]["splits"],
             manifest["claims"]["independent_rows"])
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
