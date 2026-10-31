"""Logging helpers: every stage logs to console and logs/<name>.log."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.utils.config import Paths


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    Paths.logs.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(Paths.logs / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger
