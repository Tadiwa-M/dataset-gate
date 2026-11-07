"""dsgate check <dataset> — run every check, write the report, exit 1 if the dataset is BLOCKED.

Put it in front of any training launch:   dsgate check data/my_set && python train.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import checks, report
from .core import load


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dsgate", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="audit a YOLO-format dataset")
    c.add_argument("dataset")
    c.add_argument("--manifest", help="CSV with frame (+tile), capture_time, lat, lon")
    c.add_argument("--frame-regex", default=r"_\d+_\d+$",
                   help="regex removed from a tile name to get its source frame ('' = one frame per image)")
    c.add_argument("--model", help="detector .pt for the missing-label check (needs ultralytics)")
    c.add_argument("--device", default="cpu")
    c.add_argument("--time-gap", type=float, default=60, help="min seconds between frames of different splits")
    c.add_argument("--gps-radius", type=float, default=250, help="min metres between frames of different splits")
    c.add_argument("--allow-missing", nargs="*", default=[], help="class names allowed to be absent from a split")
    c.add_argument("--only", nargs="*", help=f"run only these checks: {list(checks.CHECKS)}")
    c.add_argument("--fail-on-warn", action="store_true")
    c.add_argument("--out", help="report folder (default <dataset>/_dsgate)")
    bp = sub.add_parser("bench", help="plant known faults in a sample and score what the gate catches")
    bp.add_argument("dataset")
    bp.add_argument("--work", required=True, help="scratch folder for the clean/faulty samples")
    bp.add_argument("--manifest")
    bp.add_argument("--model")
    bp.add_argument("--device", default="cpu")
    bp.add_argument("--frames", type=int, default=60)
    ag = sub.add_parser("agent", help="let a local LLM (Ollama) audit an unfamiliar dataset and explain it")
    ag.add_argument("dataset")
    ag.add_argument("--llm", default="qwen2.5:3b", help="Ollama model name")
    ag.add_argument("--model", help="detector .pt for the detector-backed checks")
    ag.add_argument("--device", default="cpu")
    ag.add_argument("--out")
    ag.add_argument("--manifest", help="manifest CSV the agent must use")
    ev = sub.add_parser("eval", help="score the agent on one-fault scenarios (vs the plain gate)")
    ev.add_argument("--dataset", action="append", required=True, help="clean dataset (repeatable)")
    ev.add_argument("--manifest", action="append", default=[], help="DATASET=CSV manifest for a dataset")
    ev.add_argument("--work", required=True)
    ev.add_argument("--seeds", type=int, default=2)
    ev.add_argument("--frames", type=int, default=40)
    ev.add_argument("--llm", default="qwen2.5:3b")
    ev.add_argument("--model", help="detector .pt: adds the class-swap and dropped-label scenarios")
    ui = sub.add_parser("ui", help="local web page: run the gate / agent and watch it live")
    ui.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if a.cmd == "ui":
        from .ui import serve
        serve(a.port)
        return 0
    if a.cmd == "agent":
        from .agent import run_agent
        r = run_agent(a.dataset, a.llm, a.model, a.device, a.out, manifest=a.manifest)
        return 1 if r["verdict"] == "BLOCKED" else 0
    if a.cmd == "eval":
        from .evaluate import evaluate
        mans = dict(m.split("=", 1) for m in a.manifest)
        r = evaluate(a.dataset, a.work, a.seeds, a.llm, a.model, a.frames, mans)
        for k, v in r["summary"].items():
            print(f"{k:34} {v}")
        return 0
    if a.cmd == "bench":
        from .bench import bench
        r = bench(a.dataset, a.work, a.manifest, a.model, a.frames, a.device)
        return 0 if r["fault_recall"] == 1.0 and not r["false_alarms_on_clean"] else 1

    out = Path(a.out) if a.out else Path(a.dataset) / "_dsgate"
    out.mkdir(parents=True, exist_ok=True)
    opts = {"out_dir": str(out), "model": a.model, "device": a.device, "time_gap_s": a.time_gap,
            "gps_radius_m": a.gps_radius, "allow_missing": a.allow_missing}
    t0 = time.time()
    ds = load(a.dataset, a.manifest, a.frame_regex)
    # a crashing check comes back as FAIL (blocks), never as a silent pass; your rules file is applied here
    results, notes = checks.run(ds, opts, [n for n in checks.CHECKS if not a.only or n in a.only],
                                on_result=lambda r: print(f"  {r.status:4}  {r.check:16} {r.summary}", flush=True))
    for n in notes:
        print(f"  rule  {n}")
    v = report.verdict(results, a.fail_on_warn)
    jp, hp = report.write(results, out, str(Path(a.dataset).resolve()), v, notes)
    print(f"\n{v} in {time.time() - t0:.0f} s — report: {hp}")
    return 1 if v == "BLOCKED" else 0


if __name__ == "__main__":
    sys.exit(main())
