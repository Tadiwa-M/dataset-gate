"""Evaluate the dataset agent, not just the checks: one planted fault per scenario, scored on what matters.

    dsgate eval --dataset D [--dataset D2 ...] --work W [--seeds 2] [--llm qwen2.5:3b]

For each dataset and seed: a clean stratified sample, then one copy per fault with only that fault planted
(inject(only=...)). The agent audits every copy; the plain gate (no LLM) runs too, as the baseline.
Per run we record:
  correct          verdict right (clean -> PASSED; faulty -> BLOCKED with the expected check failing)
  caught_by        "model" (the model chose the check that caught it) / "code" (only the coverage guarantee
                   caught it) / "missed"
  model coverage   how many of the checks the model ran on its own; invalid tool calls
  explanation      kept / dropped (repetitive or off-topic); mentions the expected check; unverified numbers
  stalled, fallback, seconds
Your rules file is switched off during the run (AGENT_RULES points at nothing) so results don't depend on it.
Writes eval.json and eval.html into W. Everything runs locally.
"""
from __future__ import annotations

import html
import json
import os
import shutil
import statistics
import time
from pathlib import Path

from . import checks, report
from .core import load
from .inject import inject, sample

FAULTS = ["frame_leak", "exact_dup", "near_dup", "time_leak", "bad_boxes", "class_hole"]
DETECTOR_FAULTS = ["class_swap", "dropped_labels"]


def _gate(root: Path, manifest: Path | None, model: str | None) -> tuple[str, dict]:
    ds = load(root, manifest)
    res, _ = checks.run(ds, {"out_dir": str(root / "_plain"), "model": model, "device": "cpu"}, rules={})
    return report.verdict(res), {r.check: r.status for r in res}


def evaluate(datasets: list[str], work: str, seeds: int = 2, llm: str = "qwen2.5:3b", model: str | None = None,
             n_frames: int = 40, manifests: dict | None = None, log=print) -> dict:
    from .agent import run_agent
    os.environ["AGENT_RULES"] = str(Path(work) / "_no_rules.yaml")       # rules off: measure the agent itself
    work_p = Path(work)
    work_p.mkdir(parents=True, exist_ok=True)
    faults = FAULTS + (DETECTOR_FAULTS if model else [])
    rows = []
    for di, src in enumerate(datasets):
        tag = f"dataset{di + 1}"
        man_src = (manifests or {}).get(src)
        for seed in range(seeds):
            clean = work_p / f"{tag}_s{seed}_clean"
            sample(src, clean, n_frames, seed, man_src)
            for sc in ["clean"] + faults:
                root = clean if sc == "clean" else work_p / f"{tag}_s{seed}_{sc}"
                planted = []
                if sc != "clean":
                    if root.exists():
                        shutil.rmtree(root)
                    shutil.copytree(clean, root, ignore=shutil.ignore_patterns("_dsgate", "_plain", "_agent"))
                    planted = inject(root, seed + 1, only={sc})
                    if not planted:
                        log(f"  {tag} s{seed} {sc}: not applicable (needs a manifest or a class in val)")
                        shutil.rmtree(root)
                        continue
                expect = planted[0]["expect"] if planted else None
                man = root / "manifest.csv" if (root / "manifest.csv").exists() else None
                pv, pst = _gate(root, man, model)
                calls = []

                def ev(e):
                    if e["kind"] == "tool":
                        calls.append((e.get("name"), e.get("args") or {}))
                t0 = time.time()
                r = run_agent(str(root), llm, model, "cpu", str(root / "_agent"), verbose=False, on_event=ev,
                              manifest=str(man) if man else None)
                rep = json.loads((root / "_agent" / "report.json").read_text(encoding="utf-8"))
                st = {x["check"]: x["status"] for x in rep["results"]}
                want = "PASSED" if sc == "clean" else "BLOCKED"
                correct = r["verdict"] == want and (expect is None or st.get(expect) == "FAIL")
                plain_ok = pv == want and (expect is None or pst.get(expect) == "FAIL")
                if expect is None:
                    caught = "n/a"
                elif st.get(expect) != "FAIL":
                    caught = "missed"
                else:
                    caught = "code" if expect in r["skipped_by_model"] else "model"
                log_ = json.loads((root / "_agent" / "agent_log.json").read_text(encoding="utf-8"))
                invalid = sum(1 for e in log_ if isinstance(e.get("result"), dict) and
                              ("error" in e["result"] or "_rejected" in e["result"]))
                ran_by_model = len(checks.CHECKS) - len(r["skipped_by_model"])
                row = {"dataset": tag, "seed": seed, "scenario": sc, "expect": expect, "verdict": r["verdict"],
                       "correct": correct, "plain_correct": plain_ok, "caught_by": caught,
                       "tool_calls": r["steps"], "invalid_calls": invalid, "model_ran": ran_by_model,
                       "n_checks": len(checks.CHECKS), "explanation_kept": bool(r["explanation"]),
                       "explanation_dropped": r["explanation_dropped"],
                       "mentions_expected": bool(expect and expect in (r["explanation"] or "")),
                       "unverified_numbers": len(r["unverified_numbers"]), "stalled": bool(r["stalled"]),
                       "fallback": r["fallback"], "seconds": round(time.time() - t0, 1)}
                rows.append(row)
                log(f"  {tag} s{seed} {sc:14} agent {'OK ' if correct else 'BAD'} ({r['verdict']}, caught by "
                    f"{caught}) · plain {'OK' if plain_ok else 'BAD'} · model ran {ran_by_model}/{len(checks.CHECKS)}"
                    f" · {row['seconds']} s")
    summ = summarise(rows, llm)
    (work_p / "eval.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1), encoding="utf-8")
    (work_p / "eval.html").write_text(to_html(summ, rows), encoding="utf-8")
    return {"summary": summ, "rows": rows}


def summarise(rows: list[dict], llm: str) -> dict:
    f = [r for r in rows if r["scenario"] != "clean"]
    c = [r for r in rows if r["scenario"] == "clean"]
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None  # noqa: E731
    by = {}
    for r in f:
        by.setdefault(r["scenario"], []).append(r)
    return {
        "llm": llm, "runs": len(rows), "fault_runs": len(f), "clean_runs": len(c),
        "agent_accuracy_pct": pct([r["correct"] for r in rows]),
        "plain_gate_accuracy_pct": pct([r["plain_correct"] for r in rows]),
        "faults_caught_pct": pct([r["caught_by"] != "missed" for r in f]),
        "caught_by_model_pct": pct([r["caught_by"] == "model" for r in f]),
        "caught_only_by_code_pct": pct([r["caught_by"] == "code" for r in f]),
        "false_alarms_on_clean": sum(1 for r in c if r["verdict"] != "PASSED"),
        "model_coverage_pct": pct([r["model_ran"] / r["n_checks"] for r in rows]),
        "invalid_tool_calls_per_run": round(statistics.mean(r["invalid_calls"] for r in rows), 2) if rows else None,
        "explanations_kept_pct": pct([r["explanation_kept"] for r in rows]),
        "explanations_name_the_fault_pct": pct([r["mentions_expected"] for r in f]),
        "runs_with_unverified_numbers_pct": pct([r["unverified_numbers"] > 0 for r in rows]),
        "stalls": sum(r["stalled"] for r in rows), "fallbacks": sum(r["fallback"] for r in rows),
        "median_seconds": statistics.median(r["seconds"] for r in rows) if rows else None,
        "per_fault": {k: {"runs": len(v), "caught_pct": pct([x["caught_by"] != "missed" for x in v]),
                          "by_model_pct": pct([x["caught_by"] == "model" for x in v])} for k, v in by.items()},
    }


def to_html(s: dict, rows: list[dict]) -> str:
    def tr(cells, tag="td"):
        return "<tr>" + "".join(f"<{tag}>{html.escape(str(x))}</{tag}>" for x in cells) + "</tr>"
    head = ["dataset", "seed", "scenario", "verdict", "agent ok", "plain ok", "caught by", "model ran", "invalid calls",
            "explanation", "s"]
    body = "".join(tr([r["dataset"], r["seed"], r["scenario"], r["verdict"], "yes" if r["correct"] else "NO",
                       "yes" if r["plain_correct"] else "NO", r["caught_by"], f"{r['model_ran']}/{r['n_checks']}",
                       r["invalid_calls"], "kept" if r["explanation_kept"] else f"dropped ({r['explanation_dropped']})"
                       if r["explanation_dropped"] else "none", r["seconds"]]) for r in rows)
    kv = "".join(tr([k.replace("_", " "), v]) for k, v in s.items() if k != "per_fault")
    pf = "".join(tr([k, v["runs"], f"{v['caught_pct']}%", f"{v['by_model_pct']}%"]) for k, v in s["per_fault"].items())
    return f"""<!doctype html><meta charset="utf-8"><title>Agent evaluation</title>
<style>body{{font:14px/1.5 system-ui;max-width:1100px;margin:0 auto;padding:20px}} table{{border-collapse:collapse;width:100%;margin:10px 0 24px}}
td,th{{border-bottom:1px solid #ddd;padding:5px 8px;text-align:left}} th{{background:#f3f3f0}}</style>
<h1>dataset-gate agent evaluation</h1><h2>Summary</h2><table>{kv}</table>
<h2>Per fault</h2><table>{tr(["fault", "runs", "caught (final verdict)", "caught by the model itself"], "th")}{pf}</table>
<h2>Every run</h2><table>{tr(head, "th")}{body}</table>"""
