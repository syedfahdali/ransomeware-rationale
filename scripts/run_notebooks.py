"""Execute every code cell of every notebook in notebooks/ with plain Python.

Why this exists: the notebooks are part of the deliverable, so they have to actually
run — but installing Jupyter just to check them is heavy. This script runs the code
cells in order, in one shared namespace, exactly as a kernel would (IPython magics
such as `%matplotlib inline` and `!shell` lines are skipped and reported).

    python scripts/run_notebooks.py            # all notebooks
    python scripts/run_notebooks.py notebooks/02_check_a_sample_end_to_end.ipynb

Exit code 0 = every notebook ran to the end. Figures are written to outputs/figs/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cell_code(cell: dict) -> tuple[str, list[str]]:
    """Join a code cell's source, dropping IPython magics / shell escapes."""
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    kept, skipped = [], []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            skipped.append(stripped)
            continue
        kept.append(line)
    return "\n".join(kept), skipped


def run_notebook(path: Path, verbose: bool) -> tuple[int, int, list[str]]:
    nb = json.loads(path.read_text())
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ns: dict = {"__name__": "__main__", "__file__": str(path)}
    skipped: list[str] = []
    for i, cell in enumerate(code_cells, 1):
        code, magic = cell_code(cell)
        skipped += magic
        if not code.strip():
            continue
        t0 = time.time()
        try:
            exec(compile(code, f"{path.name}:code-cell-{i}", "exec"), ns)
        except Exception:
            print(f"\nFAILED in {path.name}, code cell {i} of {len(code_cells)}:")
            print("-" * 70)
            lines = code.splitlines()
            print("\n".join(lines[:40]) + ("\n..." if len(lines) > 40 else ""))
            print("-" * 70)
            traceback.print_exc()
            return len(code_cells), i, skipped
        if verbose:
            print(f"  cell {i}/{len(code_cells)} ok ({time.time() - t0:.1f}s)")
    return len(code_cells), 0, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="*", type=Path,
                    help="notebook paths (default: every notebooks/*.ipynb)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    paths = args.notebooks or sorted((ROOT / "notebooks").glob("*.ipynb"))
    if not paths:
        print("no notebooks found")
        return 1
    failed = 0
    for p in paths:
        p = p if p.is_absolute() else ROOT / p
        print(f"== {p.relative_to(ROOT)} ==")
        n_cells, bad_cell, skipped = run_notebook(p, args.verbose)
        if bad_cell:
            failed += 1
            print(f"   {n_cells} code cells, failed at cell {bad_cell}")
        else:
            print(f"   {n_cells} code cells ran clean"
                  + (f"  ({len(skipped)} magic/shell line(s) skipped: "
                     f"{', '.join(sorted(set(skipped)))})" if skipped else ""))
    if failed:
        print(f"\n{failed} notebook(s) failed")
        return 1
    print("\nall notebooks executed cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
