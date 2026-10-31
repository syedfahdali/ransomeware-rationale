"""Download CIC-MalMem-2022 (spec section 3.1).

Chain: official UNB/CIC page scrape -> Kaggle CLI -> HuggingFace Hub -> manual.
Success target: data/raw/CIC-MalMem-2022.csv with > 50,000 data rows and the
Category/Class label columns.

Environment: the official CIC host now serves a registration form
(insert.php), so the working path on this machine is the Kaggle mirror
(credentials at ~/.kaggle/kaggle.json) with HuggingFace as fallback.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from src.utils.config import Paths, ensure_dirs, now_iso
from src.utils.logging import setup_logger

log = setup_logger("download")

TARGET = Paths.raw / "CIC-MalMem-2022.csv"
UNB_PAGE = "https://www.unb.ca/cic/datasets/malmem-2022.html"
CIC_DIR = "http://cicresearch.ca/CICDataset/CICMalMem2022/"
KAGGLE_SLUGS = [
    "luccagodoy/obfuscated-malware-memory-2022-cic",
    "jlcole/cic-malmem-2022",
    "dhoogla/cicmalmem2022",
]
HF_REPOS = ["bvk/CIC-MalMem-2022"]
MIN_ROWS = 50_000
LABEL_COLS = {"Category", "Class"}


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def csv_info(path: Path) -> dict | None:
    """Return {rows, cols, columns} if `path` looks like the MalMem CSV."""
    try:
        head = pd.read_csv(path, nrows=5)
    except Exception as e:  # noqa: BLE001
        log.warning("cannot read %s: %s", path, e)
        return None
    cols = {str(c).strip() for c in head.columns}
    n_rows = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
    return {"rows": n_rows, "cols": len(cols), "columns": sorted(cols),
            "has_labels": bool(cols & LABEL_COLS)}


def is_dataset(path: Path) -> bool:
    info = csv_info(path)
    if not info:
        return False
    ok = info["rows"] >= MIN_ROWS and info["cols"] >= 50 and info["has_labels"]
    log.info("candidate %s -> rows=%s cols=%s labels=%s ok=%s",
             path, info["rows"], info["cols"], info["has_labels"], ok)
    return ok


def _download(url: str, dest: Path, timeout: int = 900) -> bool:
    log.info("GET %s -> %s", url, dest)
    try:
        with requests.get(url, stream=True, timeout=(20, timeout)) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
            tmp.replace(dest)
            log.info("downloaded %s bytes (content-length=%s)", done, total)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("download failed: %s", e)
        return False


# --------------------------------------------------------------------------- #
# fallback 1: official page scrape
# --------------------------------------------------------------------------- #
def try_official() -> Path | None:
    log.info("--- fallback 1: official UNB/CIC page ---")
    candidates: list[str] = []
    for page in (UNB_PAGE, CIC_DIR):
        try:
            html = requests.get(page, timeout=25).text
        except Exception as e:  # noqa: BLE001
            log.warning("cannot fetch %s: %s", page, e)
            continue
        for href in re.findall(r'href="([^"]+)"', html, flags=re.I):
            if re.search(r"\.(csv|zip)(\?.*)?$", href, flags=re.I) and "insert.php" not in href:
                if href.startswith("//"):
                    href = "https:" + href
                elif href.startswith("/"):
                    href = "http://cicresearch.ca" + href
                candidates.append(href)
    # also probe the historical directory names directly (HEAD only)
    for name in ("Obfuscated-MalMem2022.csv", "CIC-MalMem-2022.csv", "MalMem2022.csv",
                 "Obfuscated-MalMem2022.zip"):
        candidates.append(CIC_DIR + "Dataset/" + name)
    for url in dict.fromkeys(candidates):
        try:
            resp = requests.head(url, timeout=15, allow_redirects=True)
            ctype = resp.headers.get("content-type", "")
            if resp.status_code == 200 and "text/html" not in ctype:
                dest = TARGET.with_suffix(Path(url).suffix)
                if _download(url, dest):
                    csv_path = _extract_zip(dest)
                    if csv_path and is_dataset(csv_path):
                        return csv_path
        except Exception:  # noqa: BLE001
            continue
    log.warning("official scrape found no direct download (site is form-gated)")
    return None


def _extract_zip(path: Path) -> Path | None:
    if path.suffix.lower() != ".zip":
        return path
    import zipfile
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".csv"):
                zf.extract(name, path.parent)
                return path.parent / name
    return None


# --------------------------------------------------------------------------- #
# fallback 2: Kaggle CLI
# --------------------------------------------------------------------------- #
def try_kaggle() -> Path | None:
    log.info("--- fallback 2: Kaggle CLI ---")
    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        log.warning("no ~/.kaggle/kaggle.json; skipping Kaggle")
        return None
    kaggle_bin = Path(sys.executable).parent / "kaggle"
    kaggle_exe = str(kaggle_bin) if kaggle_bin.exists() else "kaggle"
    for slug in KAGGLE_SLUGS:
        cmd = [kaggle_exe, "datasets", "download", "-d", slug, "-p", str(Paths.raw),
               "--unzip", "-o", "--quiet"]
        log.info("running: %s", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except Exception as e:  # noqa: BLE001
            log.warning("kaggle failed to start: %s", e)
            continue
        log.info("kaggle rc=%s stdout=%s stderr=%s", proc.returncode,
                 proc.stdout.strip()[-400:], proc.stderr.strip()[-400:])
        if proc.returncode != 0:
            continue
        for path in sorted(Paths.raw.rglob("*.csv")):
            if is_dataset(path):
                return path
    return None


# --------------------------------------------------------------------------- #
# fallback 3: HuggingFace Hub
# --------------------------------------------------------------------------- #
def try_huggingface() -> Path | None:
    log.info("--- fallback 3: HuggingFace Hub ---")
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # noqa: BLE001
        log.warning("huggingface_hub unavailable: %s", e)
        return None
    for repo in HF_REPOS:
        local = Paths.raw / "hf" / repo.split("/")[-1]
        try:
            snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(local))
        except Exception as e:  # noqa: BLE001
            log.warning("snapshot_download(%s) failed: %s", repo, e)
            continue
        for path in sorted(local.rglob("*.csv"), key=lambda p: -p.stat().st_size):
            if is_dataset(path):
                return path
    # last resort: HF search API for other mirrors
    try:
        r = requests.get(
            "https://huggingface.co/api/datasets",
            params={"search": "MalMem", "limit": 20}, timeout=25)
        for ds in r.json():
            rid = ds.get("id")
            if not rid:
                continue
            log.info("trying HF search hit: %s", rid)
            try:
                local = Paths.raw / "hf" / rid.replace("/", "__")
                snapshot_download(repo_id=rid, repo_type="dataset", local_dir=str(local))
            except Exception:  # noqa: BLE001
                continue
            for path in sorted(local.rglob("*.csv"), key=lambda p: -p.stat().st_size):
                if is_dataset(path):
                    return path
    except Exception as e:  # noqa: BLE001
        log.warning("HF search failed: %s", e)
    return None


# --------------------------------------------------------------------------- #
# fallback 4: manual
# --------------------------------------------------------------------------- #
def manual_wait(wait_minutes: float = 0.0) -> Path | None:
    log.info("--- fallback 4: manual placement ---")
    log.info("Please place the CIC-MalMem-2022 CSV (e.g. Obfuscated-MalMem2022.csv) at:")
    log.info("    %s", TARGET)
    if wait_minutes <= 0:
        return None
    deadline = time.time() + wait_minutes * 60
    while time.time() < deadline:
        if TARGET.exists() and is_dataset(TARGET):
            return TARGET
        time.sleep(10)
    return None


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Download CIC-MalMem-2022")
    ap.add_argument("--wait-minutes", type=float, default=0.0,
                    help="poll for a manually placed file for this long")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()

    if TARGET.exists() and is_dataset(TARGET) and not args.force:
        info = csv_info(TARGET)
        log.info("dataset already present: %s rows, %s cols", info["rows"], info["cols"])
        _write_info(TARGET, "existing", None)
        return 0

    found = try_official() or try_kaggle() or try_huggingface() or manual_wait(args.wait_minutes)
    if not found:
        log.error("FAILED to obtain the dataset. Place the CSV at %s and re-run.", TARGET)
        return 1

    if found.resolve() != TARGET.resolve():
        shutil.copy2(found, TARGET)
    info = csv_info(TARGET)
    assert info and is_dataset(TARGET), "downloaded file failed validation"
    _write_info(TARGET, "download-chain", info)

    # quick label counts for the log
    head = pd.read_csv(TARGET, usecols=lambda c: str(c).strip() in LABEL_COLS)
    if "Category" in head.columns:
        log.info("category counts:\n%s", head["Category"].value_counts().to_string())
    if "Class" in head.columns:
        log.info("class counts:\n%s", head["Class"].value_counts().to_string())
    log.info("DOWNLOAD OK: %s (%s rows)", TARGET, info["rows"])
    return 0


def _write_info(path: Path, source: str, info: dict | None) -> None:
    payload = {"path": str(path), "source": source, "checked_at": now_iso(),
               "info": info or csv_info(path)}
    (Paths.raw / "download_info.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    sys.exit(main())
