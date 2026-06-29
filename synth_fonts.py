"""
CVdoku — Synthetic Printed-Font Digit Generator
=================================================
Renders digits 1-9 in real sans-serif fonts (the actual font family this
app needs to recognise) instead of relying on MNIST handwriting or a
grab-bag of unrelated Chars74K fonts.

Why this beats downloading a dataset for this specific problem:
  1. Full control over which fonts are used — swap in the exact font of
     your target Sudoku source if you can identify/screenshot it.
  2. Unlimited, perfectly-labelled data, generated offline, no license
     friction.
  3. Every synthetic render is passed through the SAME _binarize() +
     _center_digit() pipeline classifier.py uses at inference time, so
     there is no train/inference preprocessing mismatch — the second,
     usually-invisible domain gap that a downloaded dataset can never
     close for you, no matter how clean its source images are.

Run standalone to preview a sample grid before wiring into train.py:
    python synth_fonts.py --preview
"""

import os
import glob
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ── Font pool ────────────────────────────────────────────────────────────────
# Clean sans-serif faces — the family of fonts a digital Sudoku app/PDF is
# actually likely to use. This searches common Windows/macOS/Linux system
# font locations and keeps only the ones that actually exist and load on
# THIS machine — the original hardcoded Linux paths only worked in the
# sandbox they were written in, not on your Windows install.
#
# To add the exact app font your target Sudoku source uses (the single
# highest-leverage addition you can make here): drop a .ttf/.otf file into
# a "fonts" folder next to this script — it gets picked up automatically.

_CANDIDATE_FONT_PATHS = [
    # Windows
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf"),
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arialbd.ttf"),
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "verdana.ttf"),
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "verdanab.ttf"),
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "tahoma.ttf"),
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "tahomabd.ttf"),
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "calibri.ttf"),
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "calibrib.ttf"),
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "segoeui.ttf"),
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "segoeuib.ttf"),
    # macOS
    "/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/Library/Fonts/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]


def _discover_fonts() -> list:
    """Keep only font paths that exist AND actually load on this machine,
    plus any .ttf/.otf the user drops in a local fonts/ folder."""
    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    candidates = list(_CANDIDATE_FONT_PATHS)
    if os.path.isdir(local_dir):
        candidates += glob.glob(os.path.join(local_dir, "*.ttf"))
        candidates += glob.glob(os.path.join(local_dir, "*.otf"))

    usable = []
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            ImageFont.truetype(path, 40)  # confirm PIL can actually open it
            usable.append(path)
        except Exception:
            continue
    return usable


FONT_PATHS = _discover_fonts()

if not FONT_PATHS:
    raise RuntimeError(
        "No usable fonts found on this system in the known Windows/macOS/"
        "Linux font locations. Drop .ttf/.otf files into a 'fonts/' folder "
        "next to synth_fonts.py (create it if it doesn't exist), then "
        "re-run. On Windows, check that C:\\Windows\\Fonts\\arial.ttf "
        "actually exists -- it should be there by default."
    )

print(f"[synth_fonts] Using {len(FONT_PATHS)} font(s): "
      f"{[os.path.basename(p) for p in FONT_PATHS]}")

CANVAS_SIZE = 96     # Matches classifier._binarize()'s internal working size
OUT_SIZE    = 28     # Final size fed to the CNN


def _render_glyph(digit: int, font_path: str, font_size: int,
                   dx: int, dy: int, rotation: float) -> np.ndarray:
    """Render a single digit onto a 96x96 canvas, dark-on-light (matches a
    real photographed/scanned puzzle before classifier._binarize() inverts
    it)."""
    canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
    font   = ImageFont.truetype(font_path, font_size)
    draw   = ImageDraw.Draw(canvas)

    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    glyph = Image.new("L", (w + 4, h + 4), color=255)
    ImageDraw.Draw(glyph).text((2 - bbox[0], 2 - bbox[1]), text, font=font, fill=0)

    if rotation:
        glyph = glyph.rotate(rotation, resample=Image.BICUBIC, fillcolor=255)

    cx = CANVAS_SIZE // 2 - glyph.width  // 2 + dx
    cy = CANVAS_SIZE // 2 - glyph.height // 2 + dy
    canvas.paste(glyph, (cx, cy))

    return np.array(canvas, dtype=np.uint8)


def _to_inference_pipeline(gray_dark_on_light: np.ndarray) -> np.ndarray:
    """
    Mirrors classifier.py's _binarize() + _center_digit() exactly, so the
    model trains on data that's been through the identical pipeline it
    will see at inference. This closes the preprocessing-mismatch gap
    that a pre-made dataset (MNIST, Chars74K, whatever) cannot close,
    since those were captured/processed independently of this app.
    """
    cell = cv2.resize(gray_dark_on_light, (96, 96))
    cell = cv2.GaussianBlur(cell, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        cell, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=5
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 60:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    crop    = binary[y:y + h, x:x + w]
    resized = cv2.resize(crop, (20, 20))
    out     = np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.uint8)
    out[4:24, 4:24] = resized
    return out


def generate_dataset(samples_per_digit: int = 600, seed: int = 42):
    """
    Generate a synthetic printed-font training set.

    Returns:
        x: (N, 28, 28) uint8 array
        y: (N,) int array, values 1-9
    """
    rng = random.Random(seed)
    xs, ys = [], []

    for digit in range(1, 10):
        for _ in range(samples_per_digit):
            font_path = rng.choice(FONT_PATHS)
            font_size = rng.randint(40, 60)
            dx        = rng.randint(-4, 4)
            dy        = rng.randint(-4, 4)
            rotation  = rng.uniform(-8, 8)

            raw  = _render_glyph(digit, font_path, font_size, dx, dy, rotation)
            proc = _to_inference_pipeline(raw)
            if proc is None:
                continue
            xs.append(proc)
            ys.append(digit)

    return np.array(xs, dtype=np.uint8), np.array(ys, dtype=np.int64)


def save_preview(path: str = "synth_preview.png", per_digit: int = 8, seed: int = 7):
    """Render a grid of samples so you can SEE the output before trusting it."""
    rng = random.Random(seed)
    rows = []
    for digit in range(1, 10):
        row_imgs = []
        for _ in range(per_digit):
            font_path = rng.choice(FONT_PATHS)
            font_size = rng.randint(40, 60)
            dx        = rng.randint(-4, 4)
            dy        = rng.randint(-4, 4)
            rotation  = rng.uniform(-8, 8)
            raw  = _render_glyph(digit, font_path, font_size, dx, dy, rotation)
            proc = _to_inference_pipeline(raw)
            if proc is None:
                proc = np.zeros((OUT_SIZE, OUT_SIZE), dtype=np.uint8)
            row_imgs.append(cv2.resize(proc, (56, 56), interpolation=cv2.INTER_NEAREST))
        rows.append(np.hstack(row_imgs))
    grid = np.vstack(rows)
    cv2.imwrite(path, grid)
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    if args.preview:
        out = save_preview()
        print(f"Preview grid saved to {out}")
    else:
        x, y = generate_dataset(samples_per_digit=50)
        print(f"Generated {len(x)} samples, shape {x.shape}, labels {sorted(set(y.tolist()))}")