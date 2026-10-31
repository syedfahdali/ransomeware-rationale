"""Optional EMBER loader — deliberately a stub (spec section 3.2).

EMBER is not the default: it needs >=15 GB free disk, has index/hashed
feature names (poor for checkable rationales) and has no ransomware label.
"""
from __future__ import annotations


def load_ember(*_args, **_kwargs):
    raise NotImplementedError(
        "EMBER support is not implemented in this study. To add it:\n"
        "  1. ensure >=15 GB free disk (df -h ~),\n"
        "  2. download EMBER2018 vectorized .dat files (or thrember for EMBER2024),\n"
        "  3. subsample ~100k train rows to fit 8 GB RAM,\n"
        "  4. map the 2381 feature indices to group names and build evidence at the\n"
        "     feature-GROUP level,\n"
        "  5. wire `--dataset ember` in configs/default.yaml.\n"
        "Never re-extract features from raw binaries on this machine."
    )
