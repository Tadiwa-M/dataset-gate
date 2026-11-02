"""Benchmark the gate: sample a clean dataset, run the gate (false alarms?), plant known faults,
run it again (what did it catch?), and write bench.json + a printed scorecard.

A fault counts as CAUGHT when the check it should trip turned WARN/FAIL and, where that check lists
items, the planted item appears in its evidence. A check that fails on the CLEAN sample is a false alarm.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import checks, report
from .core import load
from .inject import inject, sample


def run_gate(root: Path, opts: dict, manifest: Path | None) -> list:
    ds = load(root, manifest if manifest and manifest.exists() else None)
    opts = {**opts, "out_dir": str(root / "_dsgate")}
    Path(opts["out_dir"]).mkdir(parents=True, exist_ok=True)
    res = []
    for name, fn in checks.CHECKS.items():
        try:
            res.append(fn(ds, opts))
        except Exception as exc:  # noqa: BLE001
            res.append(checks.Result(name, "FAIL", f"check crashed: {type(exc).__name__}: {exc}"))
    report.write(res, Path(opts["out_dir"]), str(root), report.verdict(res))
    return res


def bench(src: str, work: str, manifest: str | None = None, model: str | None = None, n_frames: int = 60,
          device: str = "cpu", seed: int = 0) -> dict:
    work = Path(work)
    clean = sample(src, work / "clean", n_frames, seed, manifest)
    faulty = sample(src, work / "faulty", n_frames, seed, manifest)
    opts = {"model": model, "device": device}
    print("gate on CLEAN sample …", flush=True)
    r_clean = {r.check: r for r in run_gate(clean, opts, clean / "manifest.csv")}
    faults = inject(faulty, seed + 1)
    print(f"planted {len(faults)} faults; gate on FAULTY sample …", flush=True)
    r_bad = {r.check: r for r in run_gate(faulty, opts, faulty / "manifest.csv")}

    scored = []
    for f in faults:
        r = r_bad.get(f["expect"])
        tripped = r is not None and r.status in ("WARN", "FAIL")
        listed = r is not None and f["item"].split("~")[0].split("<->")[0] in json.dumps(r.details, default=str)
        lists_items = r is not None and any(isinstance(v, list) and v for v in r.details.values())
        caught = tripped and (listed or not lists_items or f["fault"] in ("class_hole", "class_swap", "time_leak"))
        skipped = r is not None and r.status == "SKIP"
        scored.append({**f, "status": r.status if r else "missing", "caught": bool(caught), "skipped": skipped})
    false_alarms = [{"check": k, "status": v.status, "summary": v.summary} for k, v in r_clean.items()
                    if v.status == "FAIL"]
    testable = [s for s in scored if not s["skipped"]]
    by_type = {}
    for s in scored:
        by_type.setdefault(s["fault"], []).append(s["caught"])
    out = {"faults_planted": len(scored), "testable": len(testable),
           "caught": sum(s["caught"] for s in testable),
           "fault_recall": round(sum(s["caught"] for s in testable) / max(len(testable), 1), 3),
           "by_fault": {k: f"{sum(v)}/{len(v)}" for k, v in by_type.items()},
           "skipped_faults": [s["fault"] for s in scored if s["skipped"]],
           "false_alarms_on_clean": false_alarms, "faults": scored,
           "clean_statuses": {k: v.status for k, v in r_clean.items()},
           "faulty_statuses": {k: v.status for k, v in r_bad.items()}}
    (work / "bench.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"\nfault recall {out['caught']}/{out['testable']} testable  |  by fault: {out['by_fault']}")
    print(f"skipped (needs --model / manifest): {out['skipped_faults']}")
    print(f"false alarms on the clean sample (FAIL): {len(false_alarms)}"
          + "".join(f"\n   {a['check']}: {a['summary']}" for a in false_alarms))
    return out
