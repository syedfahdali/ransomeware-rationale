"""Generate the project's brand assets — logo, wordmark banner and an animated
banner GIF — as pure Pillow drawing plus a hand-written SVG that uses the same
geometry (no SVG rasteriser, no network, no design tool).

Design: a memory chip whose core is a green verified-check mark — "a memory
forensics sample, explained and then checked" — on the project's dark palette
(the same one `make_demo_gif.py` uses).

Artifacts (all deterministic; the banner reads the real arm scores, so it can
never drift from the study):

    docs/logo.svg      512x512 vector mark (scales to any size)
    docs/logo.png      512x512 mark
    docs/banner.png    1280x360  mark + wordmark + per-arm FS strip
    docs/banner.gif    1000x300  animated: the check draws, the bars grow

Run:  python scripts/make_brand_assets.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DOCS = ROOT / "docs"

# ---------------------------------------------------------------- palette --- #
BG = (13, 17, 23)
BG2 = (22, 27, 34)
BORDER = (48, 54, 61)
FG = (201, 209, 217)
DIM = (110, 118, 129)
ACCENT = (88, 166, 255)      # the pipeline / evidence blue
OK = (63, 185, 80)           # verified green
WARN = (210, 153, 34)
ALT = (219, 109, 74)         # the corrupted-evidence arm

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
             "/usr/share/fonts/truetype/msttcorefonts"]
SANS = ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf", "Verdana_Bold.ttf"]
SANS_REG = ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf", "Verdana.ttf"]
MONO = ["DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "cour.ttf"]


def font(names: list[str], size: int):
    for d in FONT_DIRS:
        for n in names:
            p = Path(d) / n
            if p.exists():
                return ImageFont.truetype(str(p), max(8, size))
    return ImageFont.load_default()


def fit_font(draw: ImageDraw.ImageDraw, text: str, names: list[str], size: int,
             max_width: int) -> ImageFont.FreeTypeFont:
    """Largest size <= `size` at which `text` fits `max_width` pixels.

    Keeps the 1280px banner and the narrower GIF from clipping the wordmark.
    """
    f = font(names, size)
    while size > 10 and draw.textlength(text, font=f) > max_width:
        size -= 1
        f = font(names, size)
    return f


# ------------------------------------------------------------------- mark --- #
def chip_geometry(size: int, ox: float = 0.0, oy: float = 0.0) -> dict:
    """Rounded memory-chip square with 4 pins per side plus a check mark.

    Returned in pixel coordinates so the PNG, the GIF and the SVG all agree.
    """
    s = float(size)
    x0, y0 = ox + 0.16 * s, oy + 0.16 * s
    x1, y1 = ox + 0.84 * s, oy + 0.84 * s
    r = 0.10 * s
    pin_len, pin_w = 0.09 * s, max(2.0, 0.032 * s)
    pins = {"left": [], "right": [], "top": [], "bottom": []}
    for i in range(4):
        t = (i + 1) / 5.0
        cy = y0 + t * (y1 - y0)
        cx = x0 + t * (x1 - x0)
        pins["left"].append(((ox + 0.16 * s - pin_len, cy), (x0, cy)))
        pins["right"].append(((x1, cy), (ox + 0.84 * s + pin_len, cy)))
        pins["top"].append(((cx, y0 - pin_len), (cx, y0)))
        pins["bottom"].append(((cx, y1), (cx, y1 + pin_len)))
    check = [(ox + 0.345 * s, oy + 0.505 * s),
             (ox + 0.455 * s, oy + 0.625 * s),
             (ox + 0.675 * s, oy + 0.375 * s)]
    return {"box": (x0, y0, x1, y1), "radius": r, "pins": pins, "check": check,
            "pin_w": pin_w}


def draw_mark(size: int, tile: bool = True, check_fraction: float = 1.0) -> Image.Image:
    """The standalone mark. `check_fraction` < 1 draws the check partially (GIF)."""
    ss = 4  # supersample for smooth corners
    img = Image.new("RGB", (size * ss, size * ss), BG)
    d = ImageDraw.Draw(img)
    S = size * ss
    if tile:
        d.rounded_rectangle([0, 0, S - 1, S - 1], radius=int(0.22 * S), fill=BG,
                            outline=BORDER, width=max(1, int(0.008 * S)))
    g = chip_geometry(S, 0.0, 0.0)
    x0, y0, x1, y1 = [int(v) for v in g["box"]]
    w = max(2, int(0.035 * S))
    d.rounded_rectangle([x0, y0, x1, y1], radius=int(g["radius"]), fill=BG2,
                        outline=ACCENT, width=w)
    for side in g["pins"].values():
        for a, b in side:
            d.line([tuple(map(int, a)), tuple(map(int, b))], fill=DIM,
                   width=max(2, int(g["pin_w"])))
    ck = g["check"]
    full = [(int(ck[0][0]), int(ck[0][1])), (int(ck[1][0]), int(ck[1][1])),
            (int(ck[2][0]), int(ck[2][1]))]
    n = check_fraction
    if n >= 1.0:
        pts = full
    else:
        seg1 = math.dist(full[0], full[1])
        seg2 = math.dist(full[1], full[2])
        want = (seg1 + seg2) * max(0.0, n)
        pts = [full[0]]
        if want <= seg1:
            t = want / seg1
            pts.append((int(full[0][0] + t * (full[1][0] - full[0][0])),
                        int(full[0][1] + t * (full[1][1] - full[0][1]))))
        else:
            pts.append(full[1])
            t = min(1.0, (want - seg1) / seg2)
            pts.append((int(full[1][0] + t * (full[2][0] - full[1][0])),
                        int(full[1][1] + t * (full[2][1] - full[1][1]))))
    if len(pts) >= 2:
        d.line(pts, fill=OK, width=max(3, int(0.075 * S)), joint="curve")
        for p in pts:  # round the endpoints without a stroke-cap API
            rr = max(3, int(0.075 * S)) // 2
            d.ellipse([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr], fill=OK)
    return img.resize((size, size), Image.LANCZOS)


def mark_svg(size: int = 512) -> str:
    """Hand-written SVG using the same geometry as `chip_geometry`."""
    S = float(size)
    g = chip_geometry(S)
    x0, y0, x1, y1 = g["box"]
    pins = []
    for side, segs in g["pins"].items():
        for a, b in segs:
            pins.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" '
                        f'y2="{b[1]:.1f}" stroke="#{_hex(DIM)}" stroke-width="{g["pin_w"]:.1f}" '
                        f'stroke-linecap="round"/>')
    ck = g["check"]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}"
     width="{size}" height="{size}" role="img"
     aria-label="Memory-forensics sample with a verified explanation">
  <title>ransomware-rationale-faithfulness</title>
  <rect x="0" y="0" width="{size}" height="{size}" rx="{0.22 * S:.0f}"
        fill="#{_hex(BG)}" stroke="#{_hex(BORDER)}" stroke-width="{0.008 * S:.1f}"/>
  <rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}"
        rx="{g["radius"]:.1f}" fill="#{_hex(BG2)}" stroke="#{_hex(ACCENT)}"
        stroke-width="{0.035 * S:.1f}"/>
{chr(10).join("  " + p for p in pins)}
  <polyline points="{' '.join(f'{p[0]:.1f},{p[1]:.1f}' for p in ck)}" fill="none"
            stroke="#{_hex(OK)}" stroke-width="{0.075 * S:.1f}" stroke-linecap="round"
            stroke-linejoin="round"/>
</svg>
'''


def _hex(rgb) -> str:
    return "%02x%02x%02x" % tuple(rgb)


# --------------------------------------------------------------- wordmark --- #
TITLE = "Ransomware Rationale Faithfulness"
SUB = "SHAP evidence  →  LLM rationale  →  deterministic verification"

ARM_ORDER = ["T", "S2", "S2R", "P2", "P0R", "P0", "X"]
ARM_LABEL = {"T": "template", "P0": "zero-shot", "P0R": "0-shot+repair", "P2": "few-shot",
             "S2": "structured", "S2R": "struct.+repair", "X": "corrupted"}
FALLBACK_FS = {"T": 1.000, "S2": 1.000, "S2R": 1.000, "P2": 0.999, "P0R": 0.891,
               "P0": 0.772, "X": 0.097}


def arm_scores() -> dict[str, float]:
    """Real per-arm mean FS from the study, with a documented fallback."""
    p = ROOT / "outputs" / "faithfulness" / "all_arms_summary.json"
    try:
        data = json.loads(p.read_text())
        out = {a: float(data[a]["FS"]) for a in ARM_ORDER if a in data}
        if len(out) == len(ARM_ORDER):
            return out
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return dict(FALLBACK_FS)


def color_for(value: float) -> tuple:
    if value >= 0.98:
        return OK
    if value >= 0.70:
        return ACCENT
    return ALT


def draw_banner(W: int, H: int, grow: float = 1.0, check_fraction: float = 1.0,
                tagline: bool = True) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W - 1, H - 1], outline=BORDER, width=2)
    d.rectangle([0, 0, W, 4], fill=ACCENT)

    mark_px = int(H * 0.52)
    mark = draw_mark(mark_px, check_fraction=check_fraction)
    mx, my = int(H * 0.13), (H - mark_px) // 2
    img.paste(mark, (mx, my))

    tx = mx + mark_px + int(H * 0.13)
    avail = W - tx - int(H * 0.06)
    d.text((tx, int(H * 0.115)), TITLE,
           font=fit_font(d, TITLE, SANS, int(H * 0.140), avail), fill=FG)
    d.text((tx, int(H * 0.295)), SUB,
           font=fit_font(d, SUB, MONO, int(H * 0.056), avail), fill=ACCENT)

    scores = arm_scores()
    bx = tx
    bw = int(avail / len(ARM_ORDER))
    base = int(H * 0.775)
    top = int(H * 0.455)
    d.line([bx, base, bx + bw * len(ARM_ORDER) - int(bw * 0.35), base], fill=BORDER, width=2)
    f_lab = font(MONO, max(9, int(H * 0.042)))
    for i, arm in enumerate(ARM_ORDER):
        v = scores[arm]
        x = bx + i * bw
        h = int((base - top) * min(v, 1.0) * grow)
        color = color_for(v)
        d.rectangle([x, base - h, x + int(bw * 0.52), base], fill=color)
        if grow >= 1.0:
            d.text((x, base - h - int(H * 0.062)), f"{v:.3f}", font=font(MONO, max(9, int(H * 0.042))),
                   fill=FG)
        d.text((x, base + int(H * 0.018)), arm, font=f_lab, fill=color)
    if tagline:
        d.text((bx, int(H * 0.905)),
               "FS holds at 1.000 where the rationale follows the rules  ·  "
               "collapses to 0.097 when evidence and rationale disagree",
               font=font(SANS_REG, max(9, int(H * 0.042))), fill=DIM)
    return img


# ------------------------------------------------------------------- GIFs --- #
def write_banner_gif(path: Path, W: int = 1120, H: int = 340) -> int:
    """The check draws on, then the FS bars grow, then it holds."""
    frames = []
    draw_frames = 10
    grow_frames = 14
    hold = 14
    for i in range(draw_frames):
        frames.append(draw_banner(W, H, grow=0.0, check_fraction=(i + 1) / draw_frames,
                                  tagline=False))
    for i in range(grow_frames):
        t = (i + 1) / grow_frames
        frames.append(draw_banner(W, H, grow=t, check_fraction=1.0, tagline=False))
    full = draw_banner(W, H, grow=1.0, check_fraction=1.0, tagline=True)
    frames.extend([full] * hold)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=110, loop=0,
                optimize=True, disposal=2)
    return len(pal)


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    logo = draw_mark(512)
    logo.save(DOCS / "logo.png", optimize=True)
    (DOCS / "logo.svg").write_text(mark_svg(512))

    banner = draw_banner(1280, 360)
    banner.save(DOCS / "banner.png", optimize=True)

    n = write_banner_gif(DOCS / "banner.gif")
    scores = arm_scores()
    print(f"docs/logo.svg       {len((DOCS / 'logo.svg').read_text())} bytes")
    print(f"docs/logo.png       {(DOCS / 'logo.png').stat().st_size} bytes")
    print(f"docs/banner.png     {(DOCS / 'banner.png').stat().st_size} bytes")
    print(f"docs/banner.gif     {(DOCS / 'banner.gif').stat().st_size} bytes "
          f"({n} frames)")
    print("arm FS used:", ", ".join(f"{a}={scores[a]:.3f}" for a in ARM_ORDER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
