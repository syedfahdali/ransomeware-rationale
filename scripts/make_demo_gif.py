"""Generate the offline animated demos in docs/ (Pillow only, no network).

  docs/demo.gif        `make all`: stages appear one by one and pass their checks,
                       then Faithfulness Score by arm, bars growing to their values.
  docs/demo_check.gif  everyday triage: one sample through
                       `python -m src.cli check ... --explain llm --repair 2`,
                       including the verifier catching a bad claim and the repair loop.

The text in the second demo is a real transcript (sample test_031534, 2026-10-11)
with long lines abridged; numbers are copied, not invented.

Run:  python scripts/make_demo_gif.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo.gif"

W, H = 960, 540
CHECK_OUT = ROOT / "docs" / "demo_check.gif"
BG = (13, 17, 23)
FG = (201, 209, 217)
DIM = (110, 118, 129)
PROMPT = (88, 166, 255)
OK = (63, 185, 80)
WARN = (210, 153, 34)
BAR = (56, 139, 253)
BAR_ALT = (219, 109, 74)

STAGES = [
    ("setup", "venv + pinned deps", "ok"),
    ("download", "CIC-MalMem-2022 -> data/raw/ (58,596 x 57)", "ok"),
    ("preprocess", "dedupe + group-aware 70/10/20 split", "ok"),
    ("train", "XGBoost: 30 trials, 191 trees, ROC-AUC 0.9273", "ok"),
    ("evidence", "TreeSHAP top-6, additivity 6.7e-06", "ok"),
    ("generate", "6 arms x 50 rationales (agent-queue round trip)", "ok"),
    ("verify", "claim checks + coverage + causal ablation", "ok"),
    ("validity", "perturbation recall >= 0.98, control -0.903 FS", "ok"),
    ("eval", "stats, plots, outputs/report.md", "ok"),
    ("test", "24 passed", "ok"),
    ("checks", "9/10 passed (phase2 documented)", "warn"),
]

ARMS = [("T", 1.000), ("P0", 0.772), ("P0R", 0.891), ("P2", 1.000),
        ("S2", 1.000), ("S2R", 1.000), ("X", 0.097)]


def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = []
    try:
        import matplotlib
        candidates.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono.ttf")
    except Exception:  # noqa: BLE001
        pass
    candidates += [Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
                   Path("/usr/share/fonts/TTF/DejaVuSansMono.ttf")]
    for c in candidates:
        if c.exists():
            return ImageFont.truetype(str(c), size)
    return ImageFont.load_default()


MONO = _font(19)
MONO_S = _font(16)
TITLE = _font(26)


def base_frame() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # window chrome
    d.rectangle([0, 0, W, 34], fill=(22, 27, 34))
    for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        d.ellipse([16 + i * 20, 12, 26 + i * 20, 22], fill=c)
    d.text((110, 8), "ransomware-rationale — make all", font=MONO_S, fill=DIM)
    return img, d


def terminal_frames() -> list[Image.Image]:
    frames: list[Image.Image] = []
    typed = "make all"
    # typing animation
    for k in range(1, len(typed) + 1):
        img, d = base_frame()
        d.text((24, 52), "$", font=MONO, fill=PROMPT)
        d.text((46, 52), typed[:k], font=MONO, fill=FG)
        frames.append(img)
    frames += [frames[-1]] * 2
    # stage reveal
    for n in range(1, len(STAGES) + 1):
        img, d = base_frame()
        d.text((24, 52), "$", font=MONO, fill=PROMPT)
        d.text((46, 52), typed, font=MONO, fill=FG)
        y = 88
        for i, (name, detail, status) in enumerate(STAGES[:n]):
            mark, colour = ("✓", OK) if status == "ok" else ("!", WARN)
            d.text((26, y), mark, font=MONO, fill=colour)
            d.text((52, y), f"{name:<11}", font=MONO, fill=FG)
            d.text((228, y), detail, font=MONO_S, fill=DIM)
            y += 30
        if n == len(STAGES):
            y += 12
            d.text((26, y), "→  outputs/report.md   (setup, results, validity, examples)",
                   font=MONO_S, fill=OK)
        frames.append(img)
    frames += [frames[-1]] * 3
    return frames


def result_frames(steps: int = 16) -> list[Image.Image]:
    frames = []
    for s in range(1, steps + 1):
        frac = s / steps
        img, d = base_frame()
        d.text((30, 50), "Faithfulness Score by arm (mean, 50 held-out samples)",
               font=TITLE, fill=FG)
        top, bottom = 150, 470
        x0, bw, gap = 70, 88, 30
        # axis
        d.line([x0 - 16, bottom, W - 40, bottom], fill=(48, 54, 61), width=2)
        for val in (0.25, 0.5, 0.75, 1.0):
            yy = bottom - int((bottom - top) * val)
            d.line([x0 - 16, yy, W - 40, yy], fill=(30, 36, 43))
            d.text((16, yy - 9), f"{val:.2f}", font=MONO_S, fill=DIM)
        for i, (name, val) in enumerate(ARMS):
            x = x0 + i * (bw + gap)
            hgt = int((bottom - top) * val * min(frac * 1.15, 1.0))
            colour = BAR_ALT if name == "X" else BAR
            if name in ("P0R", "P0"):
                colour = WARN if name == "P0R" else (188, 140, 255)
            d.rectangle([x, bottom - hgt, x + bw, bottom], fill=colour)
            d.text((x + 16, bottom + 8), name, font=MONO, fill=FG)
            if frac >= 0.99:
                d.text((x + 8, bottom - hgt - 26), f"{val:.3f}", font=MONO_S, fill=FG)
        if frac >= 0.99:
            d.text((30, bottom + 44),
                   "repair loop: P0 0.772 -> 0.891   |   control X 0.097   |   "
                   "fluency correlates |rho| <= 0.09",
                   font=MONO_S, fill=OK)
        frames.append(img)
    frames += [frames[-1]] * 8      # hold on the final result
    return frames


CHECK_CMD_LINES = ["python -m src.cli check --sample test_031534 \\",
                   "    --explain llm --prompt-version v0 --repair 2"]
CHECK_CMD = "".join(CHECK_CMD_LINES)

# (text, colour) — abridged transcript of the 2026-10-11 run on sample test_031534
CHECK_LINES: list[tuple[str, tuple[int, int, int]]] = [
    ("=====================================================================", DIM),
    (" SAMPLE test_031534   (source: data/processed/test.parquet)", FG),
    (" scope: the classifier answers 'is it ransomware?' (not clean-vs-malware)", DIM),
    ("=====================================================================", DIM),
    (" Dataset label      : ransomware  (family: Maze)", FG),
    (" Classifier verdict : benign / not ransomware   p = 0.0541", WARN),
    (" Classifier vs label: WRONG  (binary task)", WARN),
    ("", FG),
    (" Evidence: top 6 features by |SHAP| (shap.TreeExplainer, 43% of SHAP mass)", FG),
    ("   1. handles.nmutant         282.0  typical    47.8th pct  toward_benign", DIM),
    ("   2. malfind.ninjections      78.0  very_high  99.0th pct  toward_benign", DIM),
    ("   6. malfind.commitCharge   9,215   very_high  95.5th pct  toward_ransomware", DIM),
    ("", FG),
    (" Explanation (llm:prose, model agent-queue):", FG),
    ("   The classifier predicts benign with probability 0.054. handles.nmutant is", DIM),
    ("   typical ... malfind.commitCharge is very_high (value 9,215): both pushed", DIM),
    ("   the decision toward ransomware, but not enough to change the verdict.", DIM),
    ("", FG),
    (" Faithfulness check of that explanation:", FG),
    ("   stated verdict matches classifier : yes", FG),
    ("   FS = 1.000  (PASS)   6 claims: 6 supported, 0 unsupported, 0 contradicted", OK),
    ("   [ok] handles.nmutant   [ok] malfind.ninjections   [ok] pslist.avg_threads", OK),
    ("   [ok] handles.nfile     [ok] handles.ntimer       [ok] malfind.commitCharge", OK),
    ("   repair loop: round 0: FS 0.913  ->  round 1: FS 1.000", OK),
    ("", FG),
    (" VERIFIER FEEDBACK that fixed round 0:", WARN),
    ("   - Claim about 'malfind.commitCharge' says toward_benign but the evidence", WARN),
    ("     says toward_ransomware.", WARN),
    ("", FG),
    (" the explanation the model wrote on its own was NOT faithful; one repair round", DIM),
    (" made it faithful. That is the whole point of the project.", OK),
]


def _draw_prompt(d, typed: int) -> None:
    """Draw the (wrapped) command line, `typed` characters of it revealed."""
    for i, line in enumerate(CHECK_CMD_LINES):
        plain = CHECK_CMD[:sum(len(x) for x in CHECK_CMD_LINES[:i])]
        shown = line[:max(0, typed - len(plain))]
        if i == 0:
            d.text((24, 44), "$", font=MONO, fill=PROMPT)
            d.text((46, 44), shown, font=MONO_S, fill=FG)
        else:
            d.text((46, 60), shown, font=MONO_S, fill=FG)
        if shown != line:
            return


def check_frames(reveal: int = 3) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for k in range(1, len(CHECK_CMD) + 1, 3):
        img, d = base_frame()
        d.text((110, 8), "ransomware-rationale — everyday triage", font=MONO_S, fill=DIM)
        _draw_prompt(d, k)
        frames.append(img)
    frames += [frames[-1]] * 2
    steps = list(range(reveal, len(CHECK_LINES) + 1, reveal))
    if steps[-1] != len(CHECK_LINES):        # always end on the complete transcript
        steps.append(len(CHECK_LINES))
    for n in steps:
        img, d = base_frame()
        d.text((110, 8), "ransomware-rationale — everyday triage", font=MONO_S, fill=DIM)
        _draw_prompt(d, len(CHECK_CMD))
        y = 80
        for text, colour in CHECK_LINES[:n]:
            d.text((22, y), text, font=MONO_S, fill=colour)
            y += 15
        frames.append(img)
    frames += [frames[-1]] * 10
    return frames


def _save(frames: list[Image.Image], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=[120] * (len(frames) - 1) + [900], loop=0, optimize=True)
    saved = Image.open(out)
    n_saved = sum(1 for _ in __import__("PIL.ImageSequence", fromlist=["Iterator"]).Iterator(saved))
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out} — {len(frames)} generated, {n_saved} encoded frames, "
          f"{W}x{H}, {size_kb:.0f} KB")


def main() -> int:
    _save(terminal_frames() + result_frames(), OUT)
    _save(check_frames(), CHECK_OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
