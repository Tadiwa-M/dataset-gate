"""Checks that need a trusted detector (optional: pip install dsgate[detector], pass --model).

The model is a second opinion, never the judge: it flags things for a person to look at.
  missing_labels   label-less tiles where the model sees a herd / a very confident animal, and
                   labelled tiles with confident detections no label covers (missed animals);
  class_agreement  for every labelled box, what class does the model call it? A class whose boxes the
                   model mostly calls one OTHER class points at a swapped id->name map (the Kenya bug).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .core import Dataset, Result


def _predict(ds: Dataset, opts: dict):
    """Run the detector once per image and cache {(split, stem): (xyxy_norm, conf, cls)} in opts."""
    if "_preds" in opts:
        return opts["_preds"]
    from ultralytics import YOLO
    model = YOLO(opts["model"])
    preds = {}
    for t in ds.tiles.itertuples():
        r = model.predict(t.image, conf=float(opts.get("pred_conf", 0.25)), imgsz=int(opts.get("imgsz", 1280)),
                          verbose=False, device=opts.get("device", "cpu"))[0]
        preds[(t.split, t.stem)] = (r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                    r.boxes.cls.cpu().numpy().astype(int))
    opts["_preds"] = preds
    opts["_model_names"] = model.names
    return preds


def _ready(name: str, opts: dict) -> Result | None:
    if not opts.get("model"):
        return Result(name, "SKIP", "no --model given — needs a trusted detector")
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return Result(name, "SKIP", "ultralytics not installed (pip install dsgate[detector])")
    return None


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (n,4) and b (m,4), xyxy."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def _labels_xyxy(ds: Dataset):
    v = ds.boxes[ds.boxes.parse_ok & (ds.boxes.cls >= 0) & (ds.boxes.cls < len(ds.names))]
    out = {}
    for (s, t), g in v.groupby(["split", "stem"]):
        out[(s, t)] = (np.stack([(g.cx - g.w / 2).to_numpy(), (g.cy - g.h / 2).to_numpy(),
                                 (g.cx + g.w / 2).to_numpy(), (g.cy + g.h / 2).to_numpy()], 1), g.cls.to_numpy())
    return out


def check_missing_labels(ds: Dataset, opts: dict) -> Result:
    if (r := _ready("missing_labels", opts)) is not None:
        return r
    preds, labels = _predict(ds, opts), _labels_xyxy(ds)
    conf, herd, hi = float(opts.get("conf", 0.4)), int(opts.get("herd", 3)), float(opts.get("hi_conf", 0.7))
    flagged = []
    for t in ds.tiles.itertuples():
        xy, cf, _ = preds[(t.split, t.stem)]
        keep = cf >= conf
        xy, cf = xy[keep], cf[keep]
        if (t.split, t.stem) in labels:                       # labelled tile: confident boxes no label covers
            lxy = labels[(t.split, t.stem)][0]
            if len(xy):
                unc = (_iou(xy, lxy).max(1) < 0.1) if len(lxy) else np.ones(len(xy), bool)
                if (cf[unc] >= hi).sum() >= 2:
                    flagged.append({"split": t.split, "stem": t.stem, "kind": "uncovered", "n": int(unc.sum()),
                                    "max_conf": round(float(cf[unc].max()), 3), "image": t.image, "xy": xy[unc]})
        elif len(cf) >= herd or (len(cf) and cf.max() >= hi):    # empty tile with animal-like detections
            flagged.append({"split": t.split, "stem": t.stem, "kind": "empty", "n": int(len(cf)),
                            "max_conf": round(float(cf.max()), 3), "image": t.image, "xy": xy})
    arts = []
    if flagged:
        C = 200
        top = sorted(flagged, key=lambda f: (-f["n"], -f["max_conf"]))[:24]
        sheet = Image.new("RGB", (6 * C, ((len(top) + 5) // 6) * (C + 14)), "white")
        d = ImageDraw.Draw(sheet)
        for i, f in enumerate(top):
            with Image.open(f["image"]) as im0:
                im = im0.convert("RGB")
            W, H = im.size
            dr = ImageDraw.Draw(im)
            for x1, y1, x2, y2 in f["xy"]:
                dr.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=(255, 40, 40), width=4)
            r_, c_ = divmod(i, 6)
            sheet.paste(im.resize((C, C)), (c_ * C, r_ * (C + 14) + 14))
            d.text((c_ * C + 2, r_ * (C + 14) + 1), f"{f['kind']} {f['split']} n={f['n']} {f['max_conf']}", fill="black")
        p = Path(opts["out_dir"]) / "missing_labels_sheet.jpg"
        sheet.save(p, quality=88)
        arts.append(str(p))
    n_empty = sum(f["kind"] == "empty" for f in flagged)
    return Result("missing_labels", "WARN" if flagged else "PASS",
                  f"{n_empty} label-less tiles look like they hold animals, {len(flagged) - n_empty} labelled tiles "
                  f"have >=2 confident detections no label covers — likely missed animals (a model trained on them "
                  f"learns 'animal = background')",
                  {"flagged": [{k: f[k] for k in ("split", "stem", "kind", "n", "max_conf")} for f in flagged[:60]]}, arts)


def check_class_agreement(ds: Dataset, opts: dict) -> Result:
    if (r := _ready("class_agreement", opts)) is not None:
        return r
    preds, labels = _predict(ds, opts), _labels_xyxy(ds)
    mnames = opts["_model_names"]
    # map model class ids to dataset class ids by name (case-insensitive); unmatched -> -1
    lookup = {n.lower(): i for i, n in enumerate(ds.names)}
    m2d = {k: lookup.get(str(v).lower(), -1) for k, v in mnames.items()}
    conf = np.zeros((len(ds.names), len(ds.names)), int)
    for key, (lxy, lcls) in labels.items():
        xy, cf, pc = preds[key]
        if not len(xy) or not len(lxy):
            continue
        iou = _iou(lxy, xy)
        for i in range(len(lxy)):
            j = int(iou[i].argmax())
            if iou[i, j] >= 0.3 and m2d.get(int(pc[j]), -1) >= 0:
                conf[int(lcls[i]), m2d[int(pc[j])]] += 1
    issues = []
    for c in range(len(ds.names)):
        row, n = conf[c], conf[c].sum()
        if n < int(opts.get("min_matched", 10)):
            continue
        other = row.copy()
        other[c] = 0
        j = int(other.argmax())
        if other[j] / n >= float(opts.get("swap_share", 0.4)) and other[j] > row[c]:
            issues.append(f"'{ds.names[c]}' labels: the model calls {other[j]}/{n} of them '{ds.names[j]}' "
                          f"(and only {row[c]} '{ds.names[c]}')")
    agree = np.trace(conf) / max(conf.sum(), 1)
    return Result("class_agreement", "FAIL" if issues else "PASS",
                  (f"{len(issues)} classes whose labels the model mostly names as another class — check the "
                   f"id->name map: " + "; ".join(issues)) if issues else
                  f"labels and model agree on class for {agree:.0%} of matched boxes; no class looks swapped",
                  {"confusion (rows=label, cols=model)": {ds.names[i]: {ds.names[j]: int(conf[i, j])
                   for j in range(len(ds.names)) if conf[i, j]} for i in range(len(ds.names)) if conf[i].sum()},
                   "issues": issues})
