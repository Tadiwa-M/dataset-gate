"""The checks. Each is read-only: check(ds, opts) -> Result. Registered in CHECKS (run order).

Born from real failures on aerial wildlife data:
  * group/time/GPS leakage — survey frames 2 s apart (and adjacent transects) show the same animals;
  * duplicates — the same image in two splits;
  * class map — Kenya labels trained with the wrong id->name map (giraffe as "zebra", zebra dropped);
  * split balance — a class with 0 val instances silently drops out of mAP;
  * missing labels — "empty" tiles that hold herds the annotator missed;
  * images / sizes / overlap / background share — corrupt files, rotated frames, animals counted twice by
    overlapping tiles, val splits far emptier than train.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .core import Dataset, Result

EDGE_TOL = 1e-3


def _img_size(path: str) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


# ── structure ────────────────────────────────────────────────────────────────
def check_structure(ds: Dataset, opts: dict) -> Result:
    T = ds.tiles
    if T.empty:
        return Result("structure", "FAIL", f"no images found under {ds.root}/images/<split>/")
    no_lbl = T[T.label.isna()]
    stems = set(zip(T.split, T.stem))
    orphan = []
    for split in ds.splits:
        ldir = ds.root / "labels" / split
        if ldir.is_dir():
            orphan += [f"{split}/{p.name}" for p in ldir.glob("*.txt") if (split, p.stem) not in stems]
    status = "FAIL" if orphan or not ds.names else ("WARN" if len(no_lbl) else "PASS")
    has = set(ds.splits)
    if "train" not in has or not ({"val", "valid"} & has):
        status = "FAIL"
    msg = (f"{len(T)} images in splits {ds.splits}; {len(no_lbl)} without a label file "
           f"(YOLO trains those as empty background); {len(orphan)} label files without an image")
    if "test" not in has:
        msg += "; no test split — val will both choose the model and report it (optimistic)"
        status = "WARN" if status == "PASS" else status
    if not ds.names:
        msg += "; data.yaml has no class names"
    return Result("structure", status, msg,
                  {"per_split": T.split.value_counts().to_dict(), "unlabelled_images": no_lbl.stem.head(20).tolist(),
                   "orphan_labels": orphan[:20]})


# ── label validity ───────────────────────────────────────────────────────────
def check_labels(ds: Dataset, opts: dict) -> Result:
    B = ds.boxes
    if B.empty:
        return Result("labels", "FAIL", "no boxes in any label file")
    nc = len(ds.names)
    bad_parse = B[~B.parse_ok]
    bad_cls = B[B.parse_ok & ((B.cls < 0) | (B.cls >= nc))]
    v = B[B.parse_ok]
    out = v[(v.cx - v.w / 2 < -EDGE_TOL) | (v.cx + v.w / 2 > 1 + EDGE_TOL) |
            (v.cy - v.h / 2 < -EDGE_TOL) | (v.cy + v.h / 2 > 1 + EDGE_TOL)]
    nonpos = v[(v.w <= 0) | (v.h <= 0)]
    # tiny boxes need pixel size: read the header of each image that has boxes (cheap)
    sizes = {s: _img_size(p) for s, p in zip(ds.tiles.stem, ds.tiles.image) if s in set(v.stem)}
    px = v.assign(W=v.stem.map(lambda s: sizes[s][0]), H=v.stem.map(lambda s: sizes[s][1]))
    tiny = px[(px.w * px.W < opts.get("min_box_px", 4)) | (px.h * px.H < opts.get("min_box_px", 4))]
    dup = v[v.duplicated(["split", "stem", "cls", "cx", "cy", "w", "h"], keep="first")]
    fails = len(bad_parse) + len(bad_cls) + len(out) + len(nonpos)
    status = "FAIL" if fails else ("WARN" if len(tiny) or len(dup) else "PASS")
    return Result("labels", status,
                  f"{len(B)} boxes: {len(bad_parse)} unparseable, {len(bad_cls)} class id outside 0..{nc - 1}, "
                  f"{len(out)} outside the image, {len(nonpos)} zero-size, {len(tiny)} under "
                  f"{opts.get('min_box_px', 4)} px, {len(dup)} exact duplicates",
                  {"examples": pd.concat([bad_parse, bad_cls, out, nonpos]).head(10)
                   [["split", "stem", "line", "cls"]].to_dict("records")})


# ── class map & balance ──────────────────────────────────────────────────────
def class_table(ds: Dataset) -> pd.DataFrame:
    v = ds.boxes[ds.boxes.parse_ok & (ds.boxes.cls >= 0) & (ds.boxes.cls < len(ds.names))]
    v = v.merge(ds.tiles[["split", "stem", "frame"]], on=["split", "stem"])
    inst = v.pivot_table(index="cls", columns="split", values="line", aggfunc="size", fill_value=0)
    fr = v.pivot_table(index="cls", columns="split", values="frame", aggfunc="nunique", fill_value=0)
    idx = range(len(ds.names))
    inst, fr = inst.reindex(idx, fill_value=0), fr.reindex(idx, fill_value=0)
    t = pd.DataFrame({s: inst.get(s, 0).astype(str) + " / " + fr.get(s, 0).astype(str) for s in ds.splits})
    t.index = [f"{i} {ds.names[i]}" for i in idx]
    return t


def check_class_balance(ds: Dataset, opts: dict) -> Result:
    allow = set(opts.get("allow_missing", []))
    v = ds.boxes[ds.boxes.parse_ok & (ds.boxes.cls >= 0) & (ds.boxes.cls < len(ds.names))]
    cnt = v.groupby(["cls", "split"]).size().unstack(fill_value=0).reindex(range(len(ds.names)), fill_value=0)
    never = [ds.names[c] for c in cnt.index if cnt.loc[c].sum() == 0]
    holes = []
    for c in cnt.index:
        name = ds.names[c]
        if name in allow or cnt.loc[c].sum() == 0:
            continue
        for s in ds.splits:
            if cnt.loc[c].get(s, 0) == 0:
                holes.append(f"{name}: 0 in {s}")
    rare = [f"{ds.names[c]}: {int(cnt.loc[c].get('val', cnt.loc[c].get('valid', 0)))} in val"
            for c in cnt.index if 0 < cnt.loc[c].get("val", cnt.loc[c].get("valid", 0)) < opts.get("min_val", 10)]
    status = "FAIL" if holes else ("WARN" if never or rare else "PASS")
    msg = ("every class appears in every split" if not holes else
           f"{len(holes)} class/split holes — a class with 0 instances in val/test silently drops out of mAP")
    if never:
        msg += f"; named but never labelled: {never}"
    if rare:
        msg += f"; under {opts.get('min_val', 10)} val instances (noisy scores): {rare}"
    return Result("class_balance", status, msg,
                  {"holes": holes, "never_labelled": never, "rare_in_val": rare,
                   "table (instances / frames)": class_table(ds).to_dict()})


def check_class_map(ds: Dataset, opts: dict) -> Result:
    """Can't prove an id->name map is right from labels alone, so: flag statistical oddities and
    write a per-class contact sheet for a person (or the agent's vision tool) to eyeball. This is
    the check that would have caught Kenya giraffes being trained as 'zebra'."""
    out = Path(opts["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    v = ds.boxes[ds.boxes.parse_ok & (ds.boxes.cls >= 0) & (ds.boxes.cls < len(ds.names))]
    if v.empty:
        return Result("class_map", "SKIP", "no valid boxes")
    stats = v.assign(area=v.w * v.h, aspect=v.w / v.h.clip(lower=1e-6)).groupby("cls") \
        .agg(n=("area", "size"), med_area=("area", "median"), med_aspect=("aspect", "median"))
    stats.index = [ds.names[i] for i in stats.index]
    rng = np.random.default_rng(0)
    excl = set(opts.get("exclude_classes") or [])
    hidden = [ds.names[c] for c in sorted(v.cls.unique()) if ds.names[c].lower() in excl]
    present = [c for c in sorted(v.cls.unique()) if ds.names[c].lower() not in excl]
    if not present:
        return Result("class_map", "SKIP", "every class present is marked sensitive — no sheet drawn")
    per, C = 6, 128
    sheet = Image.new("RGB", (per * C, len(present) * (C + 14)), "white")
    d = ImageDraw.Draw(sheet)
    paths = dict(zip(zip(ds.tiles.split, ds.tiles.stem), ds.tiles.image))
    for r, c in enumerate(present):
        g = v[v.cls == c]
        pick = g.iloc[rng.permutation(len(g))[:per]]
        d.text((2, r * (C + 14) + 1), f"{c} {ds.names[c]}  (n={len(g)})", fill="black")
        for k, b in enumerate(pick.itertuples()):
            try:
                with Image.open(paths[(b.split, b.stem)]) as im0:
                    im = im0.convert("RGB")
            except Exception:  # noqa: BLE001 — unreadable images are the `images` check's finding
                continue
            W, H = im.size
            s = max(b.w * W, b.h * H) * 3 + 24
            cx, cy = b.cx * W, b.cy * H
            crop = im.crop((cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)).resize((C, C))
            k_ = C / s
            ImageDraw.Draw(crop).rectangle([(s / 2 - b.w * W / 2) * k_, (s / 2 - b.h * H / 2) * k_,
                                            (s / 2 + b.w * W / 2) * k_, (s / 2 + b.h * H / 2) * k_],
                                           outline=(255, 40, 40), width=2)
            sheet.paste(crop, (k * C, r * (C + 14) + 14))
    path = out / "class_map_sheet.jpg"
    sheet.save(path, quality=88)
    return Result("class_map", "WARN" if opts.get("strict_class_map") else "PASS",
                  "per-class sample sheet written — look at it: does every row show the animal its name says? "
                  "(statistics cannot catch a swapped id->name map)"
                  + (f"; left out as sensitive: {', '.join(hidden)}" if hidden else ""),
                  {"per_class_stats": stats.round(4).to_dict("index"), "sensitive_not_drawn": hidden}, [str(path)])


# ── leakage ──────────────────────────────────────────────────────────────────
def check_group_leakage(ds: Dataset, opts: dict) -> Result:
    g = ds.tiles.groupby("frame").split.nunique()
    leaks = g[g > 1]
    n_frames = ds.tiles.frame.nunique()
    single = (ds.tiles.groupby("frame").size() == 1).mean()
    msg = f"{len(leaks)} of {n_frames} source frames have tiles in more than one split"
    status = "FAIL" if len(leaks) else "PASS"
    if single > 0.95 and len(ds.tiles) > 50:
        msg += " — but almost every frame has exactly one tile: is --frame-regex grouping tiles correctly?"
        status = "WARN" if status == "PASS" else status
    return Result("group_leakage", status, msg, {"leaking_frames": leaks.index[:20].tolist()})


def _frame_table(ds: Dataset) -> pd.DataFrame | None:
    if ds.manifest is None:
        return None
    M = ds.manifest
    fr = ds.tiles.groupby("frame").split.agg(lambda s: s.mode().iloc[0]).rename("split").reset_index()
    if "frame" not in M.columns:
        return None
    cols = [c for c in ("frame", "capture_time", "lat", "lon", "camera") if c in M.columns]
    return fr.merge(M[cols].drop_duplicates("frame"), on="frame", how="left")


def check_time_leakage(ds: Dataset, opts: dict) -> Result:
    F = _frame_table(ds)
    if F is None or "capture_time" not in F.columns or F.capture_time.isna().all():
        return Result("time_leakage", "SKIP", "no capture_time in a manifest (pass --manifest) — "
                      "sequential survey frames can't be checked for near-duplicates in time")
    gap = float(opts.get("time_gap_s", 60))
    F = F.dropna(subset=["capture_time"]).sort_values("capture_time")
    t = (F.capture_time - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()  # resolution-safe
    sp = F.split.to_numpy()
    close = []
    for i in range(len(F)):                      # sorted: only neighbours within the window matter
        j = i + 1
        while j < len(F) and t[j] - t[i] < gap:
            if sp[j] != sp[i]:
                close.append((F.frame.iloc[i], sp[i], F.frame.iloc[j], sp[j], round(t[j] - t[i], 1)))
            j += 1
    return Result("time_leakage", "FAIL" if close else "PASS",
                  f"{len(close)} cross-split frame pairs captured < {gap:.0f} s apart",
                  {"pairs": close[:20], "frames_with_time": int(len(F))})


def check_gps_leakage(ds: Dataset, opts: dict) -> Result:
    F = _frame_table(ds)
    if F is None or not {"lat", "lon"} <= set(F.columns) or F.lat.isna().all():
        return Result("gps_leakage", "SKIP", "no lat/lon in a manifest — can't check whether splits share ground "
                      "(adjacent transects and re-flights leak even when time gaps are large)")
    r = float(opts.get("gps_radius_m", 250))
    F = F.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    lat0 = np.radians(F.lat.mean())
    x = np.radians(F.lon.to_numpy()) * 6371000 * np.cos(lat0)
    y = np.radians(F.lat.to_numpy()) * 6371000
    cell = {}
    for i, (a, b) in enumerate(zip(x // r, y // r)):
        cell.setdefault((a, b), []).append(i)
    sp = F.split.to_numpy()
    close = set()
    for (a, b), ids in cell.items():
        near = [j for da in (-1, 0, 1) for db in (-1, 0, 1) for j in cell.get((a + da, b + db), [])]
        for i in ids:
            for j in near:
                if j > i and sp[i] != sp[j] and np.hypot(x[i] - x[j], y[i] - y[j]) < r:
                    close.add((F.frame[i], sp[i], F.frame[j], sp[j]))
    return Result("gps_leakage", "FAIL" if close else "PASS",
                  f"{len(close)} cross-split frame pairs whose camera positions are < {r:.0f} m apart "
                  "(a proxy for shared ground; footprints are better if you have them)",
                  {"pairs": sorted(close)[:20]})


def _dhash(path: str, size: int = 8) -> int:
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def check_duplicates(ds: Dataset, opts: dict) -> Result:
    md5, dh = {}, []
    for t in ds.tiles.itertuples():
        b = Path(t.image).read_bytes()
        md5.setdefault(hashlib.md5(b).hexdigest(), []).append((t.split, t.stem))
        try:
            dh.append((_dhash(t.image), t.split, t.stem))
        except Exception:  # noqa: BLE001 — unreadable images are the `images` check's finding
            continue
    exact_x = [v for v in md5.values() if len({s for s, _ in v}) > 1]
    exact_in = sum(len(v) - 1 for v in md5.values() if len(v) > 1)
    thr = int(opts.get("dhash_bits", 4))
    hashes = np.array([h for h, _, _ in dh], dtype=np.uint64)
    splits = np.array([s for _, s, _ in dh])
    near = []
    by_split = defaultdict(list)
    for i, s in enumerate(splits):
        by_split[s].append(i)
    names = list(by_split)
    for a in range(len(names)):                  # compare each split only against later ones
        for b in range(a + 1, len(names)):
            ia, ib = np.array(by_split[names[a]]), np.array(by_split[names[b]])
            for i in ia:
                x = np.bitwise_xor(hashes[ib], hashes[i])
                dist = np.array([bin(int(v)).count("1") for v in x])
                for j in ib[dist <= thr]:
                    near.append((dh[i][1], dh[i][2], dh[j][1], dh[j][2]))
    status = "FAIL" if exact_x or near else ("WARN" if exact_in else "PASS")
    return Result("duplicates", status,
                  f"{len(exact_x)} byte-identical images across splits, {len(near)} near-identical cross-split "
                  f"pairs (dHash <= {thr} bits), {exact_in} exact duplicates within splits",
                  {"cross_split_exact": exact_x[:10], "cross_split_near": near[:20]})




# ── images, tiles, background ────────────────────────────────────────────────
def _image_meta(ds: Dataset) -> pd.DataFrame:
    """Decode every image once (cached on the dataset): size, or the error that stopped it."""
    meta = getattr(ds, "_img_meta", None)
    if meta is None:
        rows = []
        for t in ds.tiles.itertuples():
            w = h = None
            err = None
            try:
                if Path(t.image).stat().st_size == 0:
                    raise ValueError("empty file")
                with Image.open(t.image) as im:
                    im.load()                                   # full decode: catches truncated files
                    w, h = im.size
            except Exception as exc:  # noqa: BLE001 — any failure to decode is the finding
                err = f"{type(exc).__name__}: {exc}"[:160]
            rows.append({"split": t.split, "stem": t.stem, "frame": t.frame, "w": w, "h": h, "error": err})
        meta = pd.DataFrame(rows, columns=["split", "stem", "frame", "w", "h", "error"])
        ds._img_meta = meta
    return meta


def check_images(ds: Dataset, opts: dict) -> Result:
    """Every image must open and decode: a corrupt file crashes a dataloader mid-run or trains as noise."""
    m = _image_meta(ds)
    bad = m[m.error.notna()]
    return Result("images", "FAIL" if len(bad) else "PASS",
                  f"{len(bad)} of {len(m)} images could not be opened or decoded",
                  {"unreadable": [(r.split, r.stem, r.error) for r in bad.head(20).itertuples()]})


def check_image_sizes(ds: Dataset, opts: dict) -> Result:
    """Tiles should share a few sizes. Odd sizes point at a tiling bug; width/height swapped points at a frame
    that was rotated (EXIF orientation) while its labels were not."""
    m = _image_meta(ds).dropna(subset=["w", "h"])
    if m.empty:
        return Result("image_sizes", "SKIP", "no readable images")
    sizes = m.groupby(["w", "h"]).size().sort_values(ascending=False)
    (mw, mh), n_main = sizes.index[0], int(sizes.iloc[0])
    rotated = m[(m.w == mh) & (m.h == mw) & (mw != mh)]
    rare = [(f"{int(w)}x{int(h)}", int(n)) for (w, h), n in sizes.items()
            if n < max(2, 0.02 * len(m)) and not (w == mh and h == mw)]
    status = "WARN" if len(rotated) or rare else "PASS"
    return Result("image_sizes", status,
                  f"most common size {int(mw)}x{int(mh)} ({n_main} of {len(m)}); {len(sizes)} distinct sizes; "
                  f"{len(rotated)} images with width and height swapped (rotated?); {len(rare)} rare sizes",
                  {"sizes": {f"{int(w)}x{int(h)}": int(n) for (w, h), n in sizes.head(10).items()},
                   "rotated": rotated.stem.head(20).tolist(), "rare_sizes": rare[:20]})


def check_tile_overlap(ds: Dataset, opts: dict) -> Result:
    """Tiles cut with overlap put one animal in two tiles. Across splits that is group_leakage; inside a split
    it double-counts animals (and their weight in training). Needs tile names ending in _<x>_<y> pixel offsets."""
    m = _image_meta(ds).dropna(subset=["w", "h"])
    xy = m.stem.str.extract(r"_(\d+)_(\d+)$").astype(float)
    m = m.assign(x=xy[0], y=xy[1]).dropna(subset=["x", "y"])
    if m.empty:
        return Result("tile_overlap", "SKIP", "tile names carry no _<x>_<y> offsets — can't place tiles in their frame")
    tiles = m[["split", "stem", "x", "y", "w", "h"]].rename(columns={"w": "W", "h": "H"})
    bx = ds.boxes[ds.boxes.parse_ok].merge(tiles, on=["split", "stem"])
    bx = bx.assign(fx=bx.x + bx.cx * bx.W, fy=bx.y + bx.cy * bx.H)          # box centre in frame pixels
    by_tile = {k: g for k, g in bx.groupby(["split", "stem"])}
    empty = bx.iloc[:0]
    pairs, dup = 0, set()
    for (split, frame), g in m.groupby(["split", "frame"]):
        t = g.to_dict("records")
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                a, b = t[i], t[j]
                ox = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
                oy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
                if ox > 0 and oy > 0:
                    pairs += 1
                    x0, y0 = max(a["x"], b["x"]), max(a["y"], b["y"])
                    g2 = by_tile.get((split, a["stem"]), empty)
                    inb = g2[(g2.fx >= x0) & (g2.fx < x0 + ox) & (g2.fy >= y0) & (g2.fy < y0 + oy)]
                    dup.update(zip(inb.split, inb.stem, inb.line))
    dup_boxes = len(dup)
    status = "WARN" if dup_boxes else "PASS"
    return Result("tile_overlap", status,
                  f"{pairs} overlapping tile pairs within splits; {dup_boxes} labelled boxes sit in an overlap, so the "
                  "same animal is also in a neighbouring tile (counted twice in training and in the scores) — fine if the overlap is deliberate",
                  {"overlapping_pairs": pairs, "boxes_in_overlaps": dup_boxes})


def check_empty_ratio(ds: Dataset, opts: dict) -> Result:
    """Share of background (label-less) tiles per split. A val/test split much emptier or fuller than train
    makes precision and false-alarm numbers incomparable with training; almost-all-empty splits score nothing."""
    labelled = set(zip(ds.boxes.split, ds.boxes.stem))
    t = ds.tiles.assign(empty=[(a, b) not in labelled for a, b in zip(ds.tiles.split, ds.tiles.stem)])
    r = t.groupby("split").empty.mean().round(3).to_dict()
    gap = float(opts.get("empty_ratio_gap", 0.2))
    base = r.get("train")
    off = {k: v for k, v in r.items() if base is not None and k != "train" and abs(v - base) > gap}
    full = {k: v for k, v in r.items() if v > 0.95}
    status = "WARN" if off or full else "PASS"
    msg = "background share per split: " + ", ".join(f"{k} {v:.0%}" for k, v in r.items())
    if off:
        msg += f"; differs from train by more than {gap:.0%}: {list(off)}"
    if full:
        msg += f"; almost all background: {list(full)}"
    return Result("empty_ratio", status, msg, {"empty_share": r})


def check_negatives_source(ds: Dataset, opts: dict) -> Result:
    """Empty tiles must come from frames someone actually searched (e.g. the survey task, not a test task whose
    frames were never fully annotated). Needs a negatives task in your rules and a manifest with a task column."""
    task, col = opts.get("negatives_task"), opts.get("task_column", "task")
    if not task:
        return Result("negatives_source", "SKIP", "no negatives task set in your rules")
    M = ds.manifest
    if M is None or col not in M.columns or "frame" not in M.columns:
        return Result("negatives_source", "SKIP", f"needs a manifest with frame and '{col}' columns")
    labelled = set(zip(ds.boxes.split, ds.boxes.stem))
    empty = ds.tiles[[(a, b) not in labelled for a, b in zip(ds.tiles.split, ds.tiles.stem)]]
    if empty.empty:
        return Result("negatives_source", "PASS", "no empty tiles")
    tasks = M.groupby("frame")[col].agg(lambda s: set(s.dropna().astype(str)))
    bad = [r for r in empty.itertuples() if task not in tasks.get(r.frame, set())]
    return Result("negatives_source", "FAIL" if bad else "PASS",
                  f"{len(bad)} of {len(empty)} empty tiles come from frames not in task '{task}' "
                  "(they may hold animals nobody marked)",
                  {"examples": [(r.split, r.stem, r.frame) for r in bad[:20]]})


def run(ds: Dataset, opts: dict, names: list[str] | None = None, rules: dict | None = None,
        on_result=None) -> tuple[list[Result], list[str]]:
    """Run checks with your rules merged in; returns (results, notes about rules applied)."""
    from . import rules as rules_mod
    rules = rules_mod.load() if rules is None else rules
    ro = rules_mod.check_opts(rules)
    o = {**opts, **ro}
    o["allow_missing"] = sorted(set(opts.get("allow_missing") or []) | set(ro.get("allow_missing") or []))
    if "gps_radius_m" in opts:                                    # caller's radius, never below the rules'
        o["gps_radius_m"] = max(float(opts["gps_radius_m"]), float(o.get("gps_radius_m", 0)))
    res = []
    for name in names or list(CHECKS):
        try:
            r = CHECKS[name](ds, o)
        except Exception as exc:  # noqa: BLE001
            r = Result(name, "FAIL", f"check crashed: {type(exc).__name__}: {exc}")
        res.append(r)
        if on_result:
            on_result(r)
    notes = rules_mod.apply(res, rules)
    return res, notes


from .model_checks import check_class_agreement, check_missing_labels  # noqa: E402  (optional detector)

CHECKS = {
    "structure": check_structure,
    "images": check_images,
    "image_sizes": check_image_sizes,
    "labels": check_labels,
    "class_balance": check_class_balance,
    "empty_ratio": check_empty_ratio,
    "class_map": check_class_map,
    "group_leakage": check_group_leakage,
    "tile_overlap": check_tile_overlap,
    "time_leakage": check_time_leakage,
    "gps_leakage": check_gps_leakage,
    "duplicates": check_duplicates,
    "negatives_source": check_negatives_source,
    "missing_labels": check_missing_labels,
    "class_agreement": check_class_agreement,
}
