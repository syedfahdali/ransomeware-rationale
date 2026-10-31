"""Acceptance-check bookkeeping (spec section 0.7).

Every phase records its checks here; `make checks` prints the summary table.
"""
from __future__ import annotations

import json

from src.utils.config import Paths, now_iso

CHECKS_PATH = Paths.outputs / "phase_checks.json"


def load_checks() -> list[dict]:
    if CHECKS_PATH.exists():
        try:
            return json.loads(CHECKS_PATH.read_text())
        except json.JSONDecodeError:
            return []
    return []


def record_check(phase: str, name: str, passed: bool, details: str = "") -> None:
    checks = load_checks()
    checks = [c for c in checks if not (c["phase"] == phase and c["name"] == name)]
    checks.append({"phase": phase, "name": name, "passed": bool(passed),
                   "details": details, "checked_at": now_iso()})
    Paths.outputs.mkdir(parents=True, exist_ok=True)
    CHECKS_PATH.write_text(json.dumps(checks, indent=2))


def main() -> int:
    print(summary_table())
    return 0


def summary_table() -> str:
    checks = load_checks()
    checks.sort(key=lambda c: (c["phase"], c["name"]))
    lines = ["| Phase | Check | Result | Details |", "|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['phase']} | {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | "
                     f"{c['details'].replace('|', '/')} |")
    n_pass = sum(1 for c in checks if c["passed"])
    lines.append("")
    lines.append(f"**{n_pass}/{len(checks)} checks passed.**")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
