"""dsgate agent — a small local LLM (Ollama) that drives the gate on an unfamiliar dataset.

What the model may do (tools, all read-only):
  inspect_dataset     what's in the folder: splits, counts, classes, file naming, candidate manifests
  list_checks         every check and what it needs
  run_checks          run chosen checks with chosen options (validated by code)
  check_details       the evidence behind one check's result
  write_fix_plan      save a markdown fix plan next to the report (never touches the data)
Guardrails, enforced in code, not in the prompt:
  * paths must live inside the dataset folder; unknown checks/options are rejected;
  * at most MAX_STEPS tool calls; the conversation is capped;
  * the VERDICT is computed from the check results by code and stamped on the report after the
    model finishes — the model cannot pass a dataset, whatever it writes;
  * if the model never runs the checks (small models fumble tool calls), code runs the full set
    and the model is only asked to explain the results.
No API keys, no network beyond localhost: talks to Ollama at http://localhost:11434.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

from . import checks, report
from .core import load

OLLAMA = "http://localhost:11434/api/chat"
MAX_STEPS = 10
SYSTEM = """You are dsgate's dataset auditor. Before anyone trains an object detector, you check the dataset
for problems that make training results lie: leakage between train/val/test (shared frames, frames
captured seconds apart, duplicate images), broken or wrong labels, classes missing from a split, and
"empty" tiles that actually contain animals.

Work like this:
1. Call inspect_dataset first.
2. Decide which checks apply and what options they need (e.g. pass a manifest CSV if one exists, so time
   leakage can be checked). Call run_checks.
3. For every FAIL or WARN, call check_details and understand it.
4. Only state numbers that appear in tool results. Call write_fix_plan with concrete, ordered fixes (which files, what to change). Never claim to have
   changed any data; you cannot.
5. Finish with a short plain-English summary for a busy researcher: what is wrong, why it matters for
   the training result, what to do first.
You cannot override a FAIL. Be concrete and brief; quote numbers from the tool results."""

TOOLS = [
    {"type": "function", "function": {"name": "inspect_dataset", "description":
        "Overview of the dataset folder: splits, image/label counts, class names, sample file names, "
        "CSV files that could be a manifest (frame, capture_time, lat, lon).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "list_checks", "description":
        "Names of all checks, what each catches and what inputs it needs.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "run_checks", "description":
        "Run checks and get each one's status (PASS/WARN/FAIL/SKIP) and summary.",
        "parameters": {"type": "object", "properties": {
            "checks": {"type": "array", "items": {"type": "string"}, "description": "check names; empty = all"},
            "manifest": {"type": "string", "description": "CSV file name inside the dataset folder, optional"},
            "frame_regex": {"type": "string", "description": "regex stripped from tile names to get the frame"},
            "time_gap_s": {"type": "number"}, "gps_radius_m": {"type": "number"}},
            "required": []}}},
    {"type": "function", "function": {"name": "check_details", "description":
        "The evidence behind one check's last result (examples, counts, tables).",
        "parameters": {"type": "object", "properties": {"check": {"type": "string"}}, "required": ["check"]}}},
    {"type": "function", "function": {"name": "write_fix_plan", "description":
        "Save an ordered markdown fix plan for a person to apply. Does not modify any data.",
        "parameters": {"type": "object", "properties": {"markdown": {"type": "string"}},
                       "required": ["markdown"]}}},
]

CHECK_HELP = {
    "structure": "folders, image/label pairing, splits present (no inputs)",
    "labels": "malformed / out-of-range / zero-size / tiny / duplicate boxes (no inputs)",
    "class_balance": "every class in every split; rare classes in val (no inputs)",
    "class_map": "writes a per-class sample sheet for a person to eyeball (no inputs)",
    "group_leakage": "tiles of one source frame in two splits (needs a correct frame_regex)",
    "time_leakage": "cross-split frames captured < time_gap_s apart (needs manifest with capture_time)",
    "gps_leakage": "cross-split frames < gps_radius_m apart (needs manifest with lat, lon)",
    "duplicates": "byte-identical and near-identical images across splits (no inputs)",
    "missing_labels": "label-less tiles a detector thinks hold animals (needs --model at launch)",
    "class_agreement": "classes whose labels a detector mostly calls another class (needs --model at launch)",
}


class Session:
    """Holds the dataset, options fixed at launch, and every check result the model has produced."""

    def __init__(self, root: Path, out: Path, model: str | None, device: str):
        self.root, self.out = root.resolve(), out
        self.base = {"out_dir": str(out), "model": model, "device": device}
        self.results: dict[str, checks.Result] = {}
        self.plan: str | None = None
        self.log: list[dict] = []

    def _inside(self, name: str) -> Path:
        p = (self.root / name).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"path '{name}' is outside the dataset folder")
        return p

    # tools ---------------------------------------------------------------
    def inspect_dataset(self) -> dict:
        ds = load(self.root, frame_regex="")
        csvs = [str(p.relative_to(self.root)) for p in self.root.rglob("*.csv") if "_dsgate" not in p.parts][:10]
        heads = {}
        for c in csvs[:4]:
            try:
                heads[c] = list(pd.read_csv(self.root / c, nrows=2).columns)
            except Exception:  # noqa: BLE001
                heads[c] = "unreadable"
        return {"splits": ds.tiles.split.value_counts().to_dict(), "classes": ds.names,
                "boxes_per_split": ds.boxes.split.value_counts().to_dict(),
                "images_without_label_file": int(ds.tiles.label.isna().sum()),
                "sample_image_names": ds.tiles.stem.sample(min(6, len(ds.tiles)), random_state=0).tolist(),
                "csv_files_and_columns": heads,
                "hint": "tile names ending in _<x>_<y> are tiles of one frame; leave frame_regex out to use the "
                        "built-in default that strips that suffix"}

    def list_checks(self) -> dict:
        return CHECK_HELP

    def run_checks(self, checks_: list | None = None, manifest: str | None = None, frame_regex: str | None = None,
                   time_gap_s: float | None = None, gps_radius_m: float | None = None) -> dict:
        def norm(c: str) -> str | None:                  # forgiving: "frame-leakage" -> group_leakage etc.
            c = str(c).lower().replace("-", "_").replace(" ", "_")
            if c in checks.CHECKS:
                return c
            alias = {"frame_leakage": "group_leakage", "duplicate_images": "duplicates", "empty_tiles": "missing_labels",
                     "class_swap": "class_agreement", "box_validity": "labels"}
            if c in alias:
                return alias[c]
            hits = [k for k in checks.CHECKS if k.split("_")[0] == c.split("_")[0] or k in c or c in k]
            return hits[0] if len(hits) == 1 else None
        asked = [(c, norm(c)) for c in (checks_ or [])]
        names = list(dict.fromkeys(n for _, n in asked if n)) or list(checks.CHECKS)
        unknown = [c for c, n in asked if not n]
        man = self._inside(manifest) if manifest else None
        if man and not man.is_file():
            return {"error": f"manifest '{manifest}' not found"}
        default_rx = r"_\d+_\d+$"
        rx = default_rx if not frame_regex else frame_regex
        re.compile(rx)                                           # reject a broken regex early
        notes = {}
        if rx != default_rx:                                     # model's regex must group at least as well
            mine = load(self.root, man, rx).tiles
            base = load(self.root, man, default_rx).tiles
            if mine.frame.nunique() > base.frame.nunique():
                notes["frame_regex"] = (f"your regex left {mine.frame.nunique()} frames vs {base.frame.nunique()} "
                                        f"with the default — it is not grouping tiles; using the default")
                rx = default_rx
        ds = load(self.root, man, rx)
        # the model may make checks stricter, never looser than the defaults
        gap = max(float(time_gap_s or 0), 60.0)
        rad = max(float(gps_radius_m or 0), 250.0)
        if time_gap_s and float(time_gap_s) < 60:
            notes["time_gap_s"] = f"{time_gap_s} s is looser than the 60 s minimum; using 60 s"
        if gps_radius_m and float(gps_radius_m) < 250:
            notes["gps_radius_m"] = f"{gps_radius_m} m is looser than the 250 m minimum; using 250 m"
        opts = {**self.base, "time_gap_s": gap, "gps_radius_m": rad}
        out = {"_guardrail_notes": notes} if notes else {}
        for n in names:
            try:
                r = checks.CHECKS[n](ds, opts)
            except Exception as exc:  # noqa: BLE001
                r = checks.Result(n, "FAIL", f"check crashed: {type(exc).__name__}: {exc}")
            self.results[n] = r
            out[n] = f"{r.status}: {r.summary}"
        if unknown:
            out["_rejected"] = f"unknown checks ignored: {unknown}"
        return out

    def _resolve(self, check: str) -> str:
        if check in self.results:
            return check
        c = check.lower().replace(" ", "_")
        hits = [k for k in self.results if k.startswith(c) or c.startswith(k) or c in k or
                c.split("_")[0] == k.split("_")[0]]
        return hits[0] if len(hits) == 1 else check

    def check_details(self, check: str) -> dict:
        check = self._resolve(check)
        r = self.results.get(check)
        if r is None:
            return {"error": f"'{check}' has not been run yet"}
        s = json.dumps(r.details, default=str)
        return {"status": r.status, "details": json.loads(s) if len(s) < 3500 else s[:3500] + " …(truncated)"}

    def write_fix_plan(self, markdown: str) -> dict:
        self.plan = markdown
        (self.out / "fix_plan.md").write_text(markdown, encoding="utf-8")
        return {"saved": str(self.out / "fix_plan.md")}

    def call(self, name: str, args: dict) -> str:
        args = args or {}
        try:
            if name == "inspect_dataset":
                res = self.inspect_dataset()
            elif name == "list_checks":
                res = self.list_checks()
            elif name == "run_checks":
                res = self.run_checks(args.get("checks"), args.get("manifest"), args.get("frame_regex"),
                                      args.get("time_gap_s"), args.get("gps_radius_m"))
            elif name == "check_details":
                res = self.check_details(str(args.get("check", "")))
            elif name == "write_fix_plan":
                res = self.write_fix_plan(str(args.get("markdown", "")))
            else:
                res = {"error": f"unknown tool '{name}'"}
        except Exception as exc:  # noqa: BLE001 — tool errors go back to the model, not up the stack
            res = {"error": f"{type(exc).__name__}: {exc}"}
        self.log.append({"tool": name, "args": args, "result": res})
        return json.dumps(res, default=str)[:6000]


def _chat(model: str, messages: list, tools: list | None, timeout: int = 600) -> dict:
    body = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]


def run_agent(dataset: str, llm: str = "qwen2.5:3b", model: str | None = None, device: str = "cpu",
              out: str | None = None, verbose: bool = True) -> dict:
    root = Path(dataset)
    outp = Path(out) if out else root / "_dsgate"
    outp.mkdir(parents=True, exist_ok=True)
    s = Session(root, outp, model, device)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Audit the dataset at {root.name} before training."}]
    t0, steps, final = time.time(), 0, ""
    while steps < MAX_STEPS:
        m = _chat(llm, msgs, TOOLS)
        msgs.append(m)
        calls = m.get("tool_calls") or []
        if not calls:
            final = m.get("content", "")
            break
        for c in calls:
            fn = c.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            steps += 1
            if verbose:
                print(f"  agent → {fn.get('name')}({json.dumps(args)[:120]})", flush=True)
            msgs.append({"role": "tool", "content": s.call(fn.get("name", ""), args)})
    fallback = False
    if not s.results:                     # the model never ran the gate: code runs it, model explains
        fallback = True
        if verbose:
            print("  (model did not run the checks — running the full set and asking it to explain)", flush=True)
        res = s.run_checks()
        msgs.append({"role": "user", "content": "Here are the check results. Explain them as instructed:\n"
                     + json.dumps(res)})
        final = _chat(llm, msgs, None).get("content", "")
    # number guard: every number the model states must appear somewhere in what the tools returned
    evidence = json.dumps([e["result"] for e in s.log], default=str)
    nums = set(re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?", final))
    unverified = sorted(n for n in nums if n.rstrip("%").replace(",", "") not in evidence
                        and len(n.rstrip("%").replace(",", "")) > 1)
    v = report.verdict(list(s.results.values()))
    report.write(list(s.results.values()), outp, str(root.resolve()), v)
    md = [f"# dsgate agent report — {root.name}", "",
          f"**VERDICT (computed from the checks, not by the model): {v}**", "",
          "| check | status | summary |", "|---|---|---|"]
    md += [f"| {r.check} | {r.status} | {r.summary.replace('|', '/')} |" for r in s.results.values()]
    md += ["", "## The agent's explanation", "", final.strip() or "(no explanation produced)"]
    if unverified:
        md += ["", f"> **Number guard:** these numbers in the explanation do not appear in any tool result and "
                   f"may be wrong: {', '.join(unverified)}. Trust the table above."]
    if s.plan:
        md += ["", "## Fix plan (proposed, not applied)", "", s.plan.strip()]
    md += ["", f"_LLM: {llm} · tool calls: {steps} · fallback: {fallback} · {time.time() - t0:.0f} s_"]
    (outp / "agent_report.md").write_text("\n".join(md), encoding="utf-8")
    (outp / "agent_log.json").write_text(json.dumps(s.log, indent=1, default=str), encoding="utf-8")
    if verbose:
        print(f"\nVERDICT {v} — {outp / 'agent_report.md'}")
    return {"verdict": v, "steps": steps, "fallback": fallback, "explanation": final, "out": str(outp),
            "unverified_numbers": unverified}
