"""Build a small SYNTHETIC aerial-survey-style dataset to try dsgate on (no real data needed).

Usage: python examples/make_demo.py demo_data
Frames (2560 px) of textured "ground" with simple shapes standing in for three animal classes are
cut into 1280 px tiles named FRAME_<x>_<y>. Frames are captured 2 s apart in flight lines separated
by 10 minutes; whole flight lines go to train / val / test, so the clean dataset passes the gate.
A manifest.csv records frame and capture_time for the time-leakage check.
Then:  dsgate check demo_data --manifest demo_data/manifest.csv
       dsgate bench demo_data --manifest demo_data/manifest.csv --work demo_bench
"""
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

NAMES = ["elephant", "zebra", "buffalo"]
FRAME, TILE = 2560, 1280
LINES = {"train": 6, "val": 2, "test": 2}       # flight lines per split
FRAMES_PER_LINE = 5


def ground(rng: np.random.Generator) -> Image.Image:
    base = rng.normal(0, 1, (FRAME // 16, FRAME // 16, 3))
    img = Image.fromarray(np.uint8(np.clip(120 + 25 * base, 0, 255))).resize((FRAME, FRAME), Image.BICUBIC)
    tint = np.array([1.05, 0.95, 0.75]) if rng.random() < 0.5 else np.array([0.85, 1.0, 0.8])
    return Image.fromarray(np.uint8(np.clip(np.asarray(img) * tint, 0, 255))).filter(ImageFilter.GaussianBlur(1))


def animal(draw: ImageDraw.ImageDraw, cls: int, cx: float, cy: float, s: float) -> tuple:
    if cls == 0:                                    # "elephant": big grey ellipse
        w, h = s * 1.6, s
        draw.ellipse([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], fill=(110, 110, 112))
    elif cls == 1:                                  # "zebra": striped box
        w, h = s * 1.3, s * 0.6
        draw.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], fill=(235, 235, 235))
        for k in range(5):
            x = cx - w / 2 + (k + 0.5) * w / 5
            draw.line([x, cy - h / 2, x, cy + h / 2], fill=(20, 20, 20), width=max(2, int(w / 14)))
    else:                                           # "buffalo": dark blob
        w, h = s * 1.2, s * 0.8
        draw.ellipse([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], fill=(35, 30, 28))
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def main(out: str, seed: int = 0) -> None:
    root = Path(out)
    rng, prng = np.random.default_rng(seed), random.Random(seed)
    t = datetime(2026, 9, 19, 7, 0, 0)
    rows, n = [], 0
    for split, lines in LINES.items():
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for _ in range(lines):
            for _ in range(FRAMES_PER_LINE):
                frame = f"DEMO_{n:04d}"
                n += 1
                img = ground(rng)
                d = ImageDraw.Draw(img)
                boxes = []
                for _ in range(prng.randint(0, 9)):
                    c = prng.randrange(3)
                    boxes.append((c, *animal(d, c, prng.uniform(60, FRAME - 60), prng.uniform(60, FRAME - 60),
                                             prng.uniform(28, 60))))
                for y0 in (0, TILE):
                    for x0 in (0, TILE):
                        stem = f"{frame}_{x0}_{y0}"
                        img.crop((x0, y0, x0 + TILE, y0 + TILE)).save(root / "images" / split / f"{stem}.jpg", quality=90)
                        lines_ = []
                        for c, x1, y1, x2, y2 in boxes:
                            ix1, iy1, ix2, iy2 = max(x1, x0), max(y1, y0), min(x2, x0 + TILE), min(y2, y0 + TILE)
                            if ix2 - ix1 > 4 and iy2 - iy1 > 4 and (ix2 - ix1) * (iy2 - iy1) > 0.5 * (x2 - x1) * (y2 - y1):
                                lines_.append(f"{c} {((ix1 + ix2) / 2 - x0) / TILE:.6f} {((iy1 + iy2) / 2 - y0) / TILE:.6f} "
                                              f"{(ix2 - ix1) / TILE:.6f} {(iy2 - iy1) / TILE:.6f}")
                        (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines_))
                rows.append(f"{frame},{split},{t.isoformat()}")
                t += timedelta(seconds=2)
            t += timedelta(minutes=10)                  # next flight line
    (root / "manifest.csv").write_text("frame,split,capture_time\n" + "\n".join(rows))
    (root / "data.yaml").write_text(f"path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
                                    f"nc: {len(NAMES)}\nnames: {NAMES}\n")
    print(f"wrote {n} frames ({n * 4} tiles) to {root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_data")
