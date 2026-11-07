"""Your rules and preferences, in one YAML file that both agents (dataset-gate and trainpilot) load and enforce
in code. The agents' language models only ever see them as context; they cannot loosen them.

File: $AGENT_RULES, else ~/.agent-rules.yaml (keep it out of git: it can name clients and species). No file means
no rules. See rules.example.yaml. Keys, all optional:

  name: "GNP rules"                          shown in the consoles
  privacy: {local_only: true}                training runs offline (no hub sync, no downloads); URLs refused as paths
  sensitive_classes: [rhino]                 never drawn in sample sheets or other shareable images
  split: {mode: spatial, gps_radius_m: 250}  spatially blocked split: frames close in time but far apart on the
                                             ground are expected, so a time_leakage FAIL becomes a WARN — only when
                                             gps_leakage ran and PASSED at a radius >= this one
  negatives: {from_task: "...", task_column: task}
                                             empty (label-less) tiles must come from frames of this task
  class_balance: {min_val: 10, allow_missing: [sable]}
  gpu: {min_free_vram_gb: 4, train_window: "20:00-07:00"}
                                             launches outside the window need an explicit override in the console
  limits: {batch: [1, 4]}                    hard limits for trainpilot: narrow its built-in ranges, never widen
  planner_notes: ["...", "..."]              preferences handed to the training agent (limits still apply)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import yaml


def path() -> Path:
    return Path(os.environ.get("AGENT_RULES") or Path.home() / ".agent-rules.yaml")


def load() -> dict:
    p = path()
    if not p.is_file():
        return {}
    r = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(r, dict):
        raise ValueError(f"{p} must hold a YAML mapping")
    r["_path"] = str(p)
    return r


def check_opts(rules: dict) -> dict:
    """Options the rules add to every check run (merged over the caller's options)."""
    o: dict = {}
    cb = rules.get("class_balance") or {}
    if cb.get("allow_missing"):
        o["allow_missing"] = list(cb["allow_missing"])
    if cb.get("min_val") is not None:
        o["min_val"] = int(cb["min_val"])
    if rules.get("sensitive_classes"):
        o["exclude_classes"] = [str(c).lower() for c in rules["sensitive_classes"]]
    neg = rules.get("negatives") or {}
    if neg.get("from_task"):
        o["negatives_task"] = str(neg["from_task"])
        o["task_column"] = str(neg.get("task_column", "task"))
    sp = rules.get("split") or {}
    if sp.get("gps_radius_m"):
        o["gps_radius_m"] = max(250.0, float(sp["gps_radius_m"]))      # rules can tighten, never loosen
    return o


def apply(results: list, rules: dict) -> list[str]:
    """Adjust results in place for rules that change how a result should be read; returns notes for the report."""
    notes = []
    sp = rules.get("split") or {}
    by = {r.check: r for r in results}
    t, g = by.get("time_leakage"), by.get("gps_leakage")
    if sp.get("mode") == "spatial" and t is not None and t.status == "FAIL":
        if g is not None and g.status == "PASS":
            t.status = "WARN"
            t.summary += (" — your rules say this split is spatially blocked and gps_leakage passed, so frames close "
                          "in time but far apart on the ground are expected (rule: split.mode = spatial)")
            notes.append("time_leakage FAIL → WARN (spatial split, gps_leakage passed)")
        else:
            notes.append("spatial split rule not applied: gps_leakage did not run or did not pass")
    return notes


def in_window(rules: dict, now: time.struct_time | None = None) -> tuple[bool, str]:
    win = (rules.get("gpu") or {}).get("train_window")
    if not win:
        return True, "no training window set"
    now = now or time.localtime()
    a, b = (int(x.split(":")[0]) * 60 + int(x.split(":")[1]) for x in win.split("-"))
    m = now.tm_hour * 60 + now.tm_min
    ok = (a <= m < b) if a < b else (m >= a or m < b)
    return ok, f"training window {win}" + ("" if ok else f" — it is {now.tm_hour:02d}:{now.tm_min:02d} now")


def summary(rules: dict, part: str = "all") -> list[str]:
    """Plain-English list of what is being enforced, for the consoles. part: all | data | train."""
    if not rules:
        return []
    out = []
    if (rules.get("privacy") or {}).get("local_only"):
        out.append("Nothing leaves this machine: training runs offline, URLs are refused as paths.")
    if rules.get("sensitive_classes"):
        out.append(f"Sensitive classes never appear in sample sheets: {', '.join(rules['sensitive_classes'])}.")
    sp = rules.get("split") or {}
    if sp.get("mode") == "spatial":
        out.append("Spatially blocked split: a time-leakage FAIL is read as WARN only if GPS leakage passes.")
    neg = rules.get("negatives") or {}
    if neg.get("from_task"):
        out.append(f"Empty tiles must come from task '{neg['from_task']}'.")
    cb = rules.get("class_balance") or {}
    if cb.get("allow_missing"):
        out.append(f"Allowed to be missing from a split: {', '.join(cb['allow_missing'])}.")
    if part == "data":
        return out
    gpu = rules.get("gpu") or {}
    if gpu.get("train_window"):
        out.append(f"GPU training only in {gpu['train_window']} unless you override it at launch.")
    if gpu.get("min_free_vram_gb"):
        out.append(f"GPU runs wait for at least {gpu['min_free_vram_gb']} GB free.")
    for k, v in (rules.get("limits") or {}).items():
        out.append(f"Hard limit: {k} {v[0]}..{v[1]} (tightens the built-in range).")
    for n in rules.get("planner_notes") or []:
        out.append(f"Planner preference: {n}")
    return out
