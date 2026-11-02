"""Fault injection for evaluating the gate: copy a SAMPLE of a clean dataset, plant known faults,
and record exactly what was planted (faults.json) so a benchmark can score what the gate caught.

Faults (each mirrors a real failure on aerial survey data):
  frame_leak    tiles of one TRAIN frame copied into VAL under the same frame name
  exact_dup     a TRAIN image + label copied into TEST under a new frame name
  near_dup      a TRAIN image re-encoded and very slightly brightened, copied into TEST
  time_leak     a VAL frame's capture_time moved to 10 s after a TRAIN frame (manifest)
  bad_boxes     label lines corrupted: class id >= nc, box outside the image, zero size
  class_hole    every instance of one class removed from VAL (its boxes deleted)
  class_swap    two class ids swapped in a share of TRAIN label files (the Kenya bug)
  dropped_labels all boxes removed from some TRAIN tiles (missed animals -> "empty" tiles)
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageEnhance

from .core import load


def sample(src: str | Path, dst: str | Path, n_frames: int = 60, seed: int = 0,
           manifest: str | Path | None = None, frame_regex: str = r"_\d+_\d+$") -> Path:
    """Copy whole frames (all their tiles) per split into dst, keeping the split structure."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    ds = load(src, manifest, frame_regex)
    rng = random.Random(seed)
    keep = []
    fb = ds.boxes.merge(ds.tiles[["split", "stem", "frame"]], on=["split", "stem"])
    for split, g in ds.tiles.groupby("split"):
        frames = sorted(g.frame.unique())
        k = max(3, int(n_frames * len(frames) / ds.tiles.frame.nunique()))
        # stratified: first one random frame per class present in this split (a clean sample must keep
        # every class in every split, or the gate is right to fail it), then fill at random
        chosen = []
        for c, cf in fb[fb.split == split].groupby("cls"):
            options = sorted(set(cf.frame) - set(chosen))
            if options:
                chosen.append(rng.choice(options))
        rest = [f for f in frames if f not in chosen]
        chosen += rng.sample(rest, max(0, min(k - len(chosen), len(rest))))
        keep += chosen
    T = ds.tiles[ds.tiles.frame.isin(keep)]
    for t in T.itertuples():
        for kind, path in (("images", t.image), ("labels", t.label)):
            if path:
                out = dst / kind / t.split / Path(path).name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)
    shutil.copy2(src / "data.yaml", dst / "data.yaml")
    if manifest:
        M = pd.read_csv(manifest)
        M[M.frame.isin(keep)].to_csv(dst / "manifest.csv", index=False)
    return dst


def inject(root: str | Path, seed: int = 1, frame_regex: str = r"_\d+_\d+$") -> list[dict]:
    root = Path(root)
    rng = random.Random(seed)
    man = root / "manifest.csv"
    ds = load(root, man if man.exists() else None, frame_regex)
    nc = len(ds.names)
    faults = []
    tr = ds.tiles[ds.tiles.split == "train"]
    labelled = tr[tr.label.notna() & tr.stem.isin(set(ds.boxes.stem))]

    def copy_tile(t, split, new_stem):
        for kind, path, ext in (("images", t.image, Path(t.image).suffix), ("labels", t.label, ".txt")):
            if path:
                out = root / kind / split / f"{new_stem}{ext}"
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)

    # frame_leak: one train frame's tiles also appear in val (same frame key)
    f = rng.choice(sorted(labelled.frame.unique()))
    for t in tr[tr.frame == f].itertuples():
        copy_tile(t, "val", t.stem)
    faults.append({"fault": "frame_leak", "expect": "group_leakage", "item": f})

    # exact_dup: a train tile copied to test under an unrelated frame name
    t = labelled.sample(1, random_state=seed).iloc[0]
    copy_tile(t, "test", "INJDUP_9999_0_0")
    faults.append({"fault": "exact_dup", "expect": "duplicates", "item": "INJDUP_9999_0_0"})

    # near_dup: re-encoded + 3% brighter copy in test
    t = labelled.sample(1, random_state=seed + 1).iloc[0]
    with Image.open(t.image) as im:
        ImageEnhance.Brightness(im.convert("RGB")).enhance(1.03).save(root / "images" / "test" / "INJNEAR_9998_0_0.jpg",
                                                                       quality=85)
    shutil.copy2(t.label, root / "labels" / "test" / "INJNEAR_9998_0_0.txt")
    faults.append({"fault": "near_dup", "expect": "duplicates", "item": "INJNEAR_9998_0_0"})

    # time_leak: a val frame moved to 10 s after a train frame
    if man.exists():
        M = pd.read_csv(man)
        M["capture_time"] = pd.to_datetime(M.capture_time)
        vf = sorted(set(ds.tiles[ds.tiles.split == "val"].frame) - {f})
        tf = sorted(set(tr.frame) - {f})
        if vf and tf:
            a, b = rng.choice(vf), rng.choice(tf)
            M.loc[M.frame == a, "capture_time"] = M.loc[M.frame == b, "capture_time"].iloc[0] + pd.Timedelta(seconds=10)
            M.to_csv(man, index=False)
            faults.append({"fault": "time_leak", "expect": "time_leakage", "item": f"{a}~{b}"})

    # bad_boxes: three corrupt lines appended to a train label
    t = labelled.sample(1, random_state=seed + 2).iloc[0]
    with open(t.label, "a", encoding="utf-8") as fh:
        fh.write(f"\n{nc + 3} 0.5 0.5 0.1 0.1\n0 1.20 0.5 0.1 0.1\n0 0.5 0.5 0 0.1\n")
    faults.append({"fault": "bad_boxes", "expect": "labels", "item": t.stem})

    # class_hole: remove one class (present in val) from every val label file
    vb = ds.boxes[ds.boxes.split == "val"]
    if not vb.empty:
        c = int(vb.cls.value_counts().index[-1])               # rarest class in val
        for lbl in (root / "labels" / "val").glob("*.txt"):
            lines = [ln for ln in lbl.read_text().splitlines() if ln.strip() and int(float(ln.split()[0])) != c]
            lbl.write_text("\n".join(lines))
        faults.append({"fault": "class_hole", "expect": "class_balance", "item": ds.names[c]})

    # class_swap: swap the two most common classes in 50% of train label files
    top = ds.boxes[ds.boxes.split == "train"].cls.value_counts().index[:2].tolist()
    if len(top) == 2:
        a, b = map(int, top)
        files = sorted({t.label for t in labelled.itertuples()})
        for lbl in rng.sample(files, len(files) // 2):
            out = []
            for ln in Path(lbl).read_text().splitlines():
                p = ln.split()
                if p and int(float(p[0])) in (a, b):
                    p[0] = str(b if int(float(p[0])) == a else a)
                out.append(" ".join(p))
            Path(lbl).write_text("\n".join(out))
        faults.append({"fault": "class_swap", "expect": "class_agreement", "item": f"{ds.names[a]}<->{ds.names[b]}"})

    # dropped_labels: empty the label files of 5 labelled train tiles
    for t in labelled.sample(min(5, len(labelled)), random_state=seed + 3).itertuples():
        Path(t.label).write_text("")
        faults.append({"fault": "dropped_labels", "expect": "missing_labels", "item": t.stem})

    (root / "faults.json").write_text(json.dumps(faults, indent=1))
    return faults
