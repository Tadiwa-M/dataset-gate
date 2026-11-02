"""Dataset model and the result type every check returns.

A dataset is the common YOLO layout:
    <root>/data.yaml                      names (list or {id: name}), optional nc
    <root>/images/<split>/*.jpg|png       splits: train / val / test (whichever exist)
    <root>/labels/<split>/<stem>.txt      "cls cx cy w h" normalised, one box per line
Optional manifest CSV (--manifest) adds what leakage checks need:
    tile or frame column, plus any of: frame, capture_time, lat, lon, split
If no manifest, a tile's frame is derived from its file name with --frame-regex
(default strips a trailing _<x>_<y> tile offset, e.g. BCK_0139_1024_4096 -> BCK_0139).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "valid", "test")


@dataclass
class Result:
    """One check's outcome. status: PASS | WARN | FAIL | SKIP."""
    check: str
    status: str
    summary: str
    details: dict = field(default_factory=dict)      # small, JSON-able evidence
    artifacts: list[str] = field(default_factory=list)  # files written (contact sheets, csvs)

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


@dataclass
class Dataset:
    root: Path
    names: list[str]
    tiles: pd.DataFrame          # one row per image: split, stem, image, label, frame
    boxes: pd.DataFrame          # one row per box: split, stem, cls, cx, cy, w, h, line_ok
    manifest: pd.DataFrame | None = None

    @property
    def splits(self) -> list[str]:
        return sorted(self.tiles.split.unique(), key=lambda s: SPLITS.index(s) if s in SPLITS else 99)


def _names(cfg: dict) -> list[str]:
    n = cfg.get("names", [])
    if isinstance(n, dict):
        return [n[k] for k in sorted(n, key=int)]
    return list(n)


def load(root: str | Path, manifest: str | Path | None = None,
         frame_regex: str = r"_\d+_\d+$") -> Dataset:
    root = Path(root)
    cfg = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8")) if (root / "data.yaml").exists() else {}
    names = _names(cfg)
    rx = re.compile(frame_regex) if frame_regex else None
    tiles, boxes = [], []
    for split in SPLITS:
        idir = root / "images" / split
        if not idir.is_dir():
            continue
        ldir = root / "labels" / split
        for img in sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT):
            lbl = ldir / f"{img.stem}.txt"
            frame = rx.sub("", img.stem) if rx else img.stem
            tiles.append({"split": split, "stem": img.stem, "image": str(img),
                          "label": str(lbl) if lbl.exists() else None, "frame": frame})
            if not lbl.exists():
                continue
            for i, line in enumerate(lbl.read_text(encoding="utf-8", errors="replace").splitlines()):
                if not line.strip():
                    continue
                p = line.split()
                try:
                    c, cx, cy, w, h = int(float(p[0])), *map(float, p[1:5])
                    ok = len(p) >= 5
                except (ValueError, IndexError):
                    c, cx, cy, w, h, ok = -1, 0.0, 0.0, 0.0, 0.0, False
                boxes.append({"split": split, "stem": img.stem, "line": i + 1, "cls": c,
                              "cx": cx, "cy": cy, "w": w, "h": h, "parse_ok": ok})
    T = pd.DataFrame(tiles, columns=["split", "stem", "image", "label", "frame"])
    B = pd.DataFrame(boxes, columns=["split", "stem", "line", "cls", "cx", "cy", "w", "h", "parse_ok"])
    M = None
    if manifest:
        M = pd.read_csv(manifest)
        if "capture_time" in M.columns:
            M["capture_time"] = pd.to_datetime(M.capture_time, errors="coerce")
        key = "tile" if "tile" in M.columns else "frame"
        if key == "tile" and "frame" in M.columns:          # manifest maps tiles to frames: trust it
            T = T.drop(columns="frame").merge(M[["tile", "frame"]].rename(columns={"tile": "stem"}),
                                              on="stem", how="left")
    return Dataset(root=root, names=names, tiles=T, boxes=B, manifest=M)
