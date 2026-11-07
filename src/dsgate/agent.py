"""dsgate agent — a small local LLM (Ollama) that drives the gate on an unfamiliar dataset.

What the model may do (tools, all read-only):
  inspect_dataset     what's in the folder: splits, counts, classes, file naming, candidate manifests
  list_checks         every check and what it needs
  run_checks          run chosen checks with chosen options (validated by code)
  check_details       the evidence behind one check's result
  write_fix_plan      save a markdown fix plan next to the report (never touches the data)
Guardrails, enforced in code, not in the prompt:
  * paths must live inside the dataset folder; unknown checks/options are rejected;
  * at most MAX_STEPS tool calls; every answer is streamed and capped, so a looping model can't hang a run;
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
4. Before every tool call, write one short sentence saying why you are calling it.
   Only state numbers that appear in tool results. Call write_fix_plan with concrete, ordered fixes (which files, what to change). Never claim to have
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
            "checks": {"type": "array", "items": {"type": "string", "enum": list(checks.CHECKS)},
                       "description": "check names; empty = all"},
            "manifest": {"type": "string", "description": "CSV file name inside the dataset folder, optional"},
            "frame_regex": {"type": "string", "description": "regex stripped from tile names to get the frame"},
            "time_gap_s": {"type": "number"}, "gps_radius_m": {"type": "number"}},
            "required": []}}},
    {"type": "function", "function": {"name": "check_details", "description":
        "The evidence behind one check's last result (examples, counts, tables).",
        "parameters": {"type": "object", "properties": {"check": {"type": "string", "enum": list(checks.CHECKS)}},
                       "required": ["check"]}}},
    {"type": "function", "function": {"name": "write_fix_plan", "description":
        "Save an ordered markdown fix plan for a person to apply. Does not modify any data.",
        "parameters": {"type": "object", "properties": {"markdown": {"type": "string"}},
                       "required": ["markdown"]}}},
]

CHECK_HELP = {
    "structure": "folders, image/label pairing, splits present (no inputs)",
    "images": "every image opens and decodes (no inputs)",
    "image_sizes": "unexpected tile sizes; width/height swapped = rotated frames (no inputs)",
    "labels": "malformed / out-of-range / zero-size / tiny / duplicate boxes (no inputs)",
    "empty_ratio": "background (label-less) share per split compared with train (no inputs)",
    "tile_overlap": "overlapping tiles of one frame inside a split: animals counted twice (needs _x_y tile names)",
    "class_balance": "every class in every split; rare classes in val (no inputs)",
    "class_map": "writes a per-class sample sheet for a person to eyeball (no inputs)",
    "group_leakage": "tiles of one source frame in two splits (needs a correct frame_regex)",
    "time_leakage": "cross-split frames captured < time_gap_s apart (needs manifest with capture_time)",
    "gps_leakage": "cross-split frames < gps_radius_m apart (needs manifest with lat, lon)",
    "duplicates": "byte-identical and near-identical images across splits (no inputs)",
    "missing_labels": "label-less tiles a detector thinks hold animals (needs --model at launch)",
    "class_agreement": "classes whose labels a detector mostly calls another class (needs --model at launch)",
    "negatives_source": "empty tiles come only from the searched task named in your rules (needs manifest task column)",
}


class Session:
    """Holds the dataset, options fixed at launch, and every check result the model has produced."""

    def __init__(self, root: Path, out: Path, model: str | None, device: str, manifest: str | None = None):
        self.root, self.out = root.resolve(), out
        self.manifest = Path(manifest).resolve() if manifest else None
        from . import rules as rules_mod
        self.rules = rules_mod.load()
        self.base = {"out_dir": str(out), "model": model, "device": device, **rules_mod.check_opts(self.rules)}
        self.rule_notes: list[str] = []
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
        man = self._inside(manifest) if manifest and not self.manifest else self.manifest
        if man and not man.is_file():
            return {"error": f"manifest '{manifest}' not found"}
        self.manifest = man
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
        rad = max(float(gps_radius_m or 0), float(self.base.get("gps_radius_m", 250.0)), 250.0)
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
        from . import rules as rules_mod
        rn = rules_mod.apply(list(self.results.values()), self.rules)
        if rn:
            self.rule_notes += [n for n in rn if n not in self.rule_notes]
            out["_your_rules"] = rn
            for n in names:
                out[n] = f"{self.results[n].status}: {self.results[n].summary}"
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


class Cancelled(Exception):
    """Raised inside run_agent when should_stop() turns true."""


def _chat(model: str, messages: list, tools: list | None, timeout: int = 120, on_token=None, should_stop=None) -> dict:
    """Streamed chat: output is capped (a looping small model can't run forever), every chunk can be shown live
    via on_token(n_chunks, text_so_far), and should_stop() is checked between chunks. timeout is per read."""
    body = {"model": model, "messages": messages, "stream": True,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1200}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    text, calls, n, last = "", [], 0, 0.0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.strip():
                continue
            chunk = json.loads(line)
            m = chunk.get("message") or {}
            text += m.get("content") or ""
            calls += m.get("tool_calls") or []
            n += 1
            if should_stop and should_stop():
                raise Cancelled()
            if on_token and (time.time() - last > 0.4 or chunk.get("done")):
                on_token(n, text)
                last = time.time()
            if chunk.get("done"):
                break
    msg = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = calls
    return msg


STAGES = ["Inspect", "Run checks", "Read evidence", "Fix plan", "Summary", "Verdict"]
_TOOL_STAGE = {"inspect_dataset": "Inspect", "list_checks": "Inspect", "run_checks": "Run checks",
               "check_details": "Read evidence", "write_fix_plan": "Fix plan"}


def _short(name: str, res_json: str) -> str:
    """One human line about a tool result, for the live feed."""
    try:
        res = json.loads(res_json)
    except json.JSONDecodeError:
        return "returned a result"
    if not isinstance(res, dict):
        return "returned a result"
    if "error" in res:
        return f"error: {res['error']}"
    if name == "inspect_dataset":
        return (f"{sum(res.get('splits', {}).values())} images in {len(res.get('splits', {}))} splits, "
                f"{len(res.get('classes', []))} classes, {len(res.get('csv_files_and_columns', {}))} CSV file(s)")
    if name == "run_checks":
        st = [str(v).split(":")[0] for k, v in res.items() if not k.startswith("_")]
        counts = " · ".join(f"{st.count(x)} {x}" for x in ("PASS", "WARN", "FAIL", "SKIP") if st.count(x))
        notes = res.get("_guardrail_notes")
        return f"{len(st)} checks: {counts}" + (f" (guardrail: {'; '.join(notes.values())})" if notes else "")
    if name == "check_details":
        return f"read the evidence ({res.get('status', '?')})"
    if name == "write_fix_plan":
        return "fix plan saved (nothing in the data was changed)"
    if name == "list_checks":
        return f"{len(res)} checks available"
    return "done"


def run_agent(dataset: str, llm: str = "qwen2.5:3b", model: str | None = None, device: str = "cpu",
              out: str | None = None, verbose: bool = True, on_event=None, should_stop=None,
              manifest: str | None = None) -> dict:
    """on_event(dict) gets every step as it happens: kind is stage | thinking | reason | tool | result | warn."""
    def emit(kind: str, **kw):
        if on_event:
            on_event({"kind": kind, **kw})
        elif verbose and kind in ("tool", "warn", "reason"):
            print(f"  {kw.get('text', '')}", flush=True)

    def ask(msgs_, tools_):
        t = time.time()
        return _chat(llm, msgs_, tools_, should_stop=should_stop,
                     on_token=lambda n, txt: emit("thinking", tokens=n, seconds=round(time.time() - t), text=txt[-400:]))

    root = Path(dataset)
    outp = Path(out) if out else root / "_dsgate"
    outp.mkdir(parents=True, exist_ok=True)
    s = Session(root, outp, model, device, manifest)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Audit the dataset at {root.name} before training."
             + (" A manifest was given and is used automatically." if manifest else "")}]
    t0, steps, final, stalled, name = time.time(), 0, "", None, ""
    emit("stage", stage="Inspect", text="the model is deciding where to start")
    while steps < MAX_STEPS:
        try:
            m = ask(msgs, TOOLS)
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001 — model stalled / Ollama down: code takes over below
            stalled = f"{type(exc).__name__}: {exc}"
            emit("warn", text=f"the model stalled ({stalled}); code takes over")
            break
        msgs.append(m)
        calls = m.get("tool_calls") or []
        if m.get("content", "").strip() and calls:
            emit("reason", text=m["content"].strip()[:600])
        if not calls:
            final = m.get("content", "")
            break
        for c in calls:
            fn = c.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            steps += 1
            emit("stage", stage=_TOOL_STAGE.get(name, "Inspect"), text=f"step {steps}: {name}")
            emit("tool", name=name, args=args, text=f"agent → {name}({json.dumps(args)[:120]})")
            res = s.call(name, args)
            emit("result", name=name, text=_short(name, res))
            msgs.append({"role": "tool", "content": res})
            if should_stop and should_stop():
                raise Cancelled()
        emit("stage", stage=_TOOL_STAGE.get(name, "Inspect"),
             text="the model is reading the result and choosing the next step")
    fallback = False
    if not s.results:                     # the model never ran the gate: code runs it, model explains
        fallback = True
        emit("stage", stage="Run checks", text="the model did not run the checks, so code runs the full set")
        emit("warn", text="(model did not run the checks — running the full set and asking it to explain)")
        res = s.run_checks()
        emit("result", name="run_checks", text=_short("run_checks", json.dumps(res)))
        if stalled is None:
            emit("stage", stage="Summary", text="the model is writing its explanation")
            msgs.append({"role": "user", "content": "Here are the check results. Explain them as instructed:\n"
                         + json.dumps(res)})
            try:
                final = ask(msgs, None).get("content", "")
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                emit("warn", text=f"no explanation: the model stalled ({type(exc).__name__})")
    skipped = [n for n in checks.CHECKS if n not in s.results]
    if skipped:
        emit("stage", stage="Run checks", text=f"the model skipped {len(skipped)} checks; code runs them so the "
                                               "verdict covers everything")
        emit("warn", text=f"model skipped: {', '.join(skipped)} (run by code)")
        res = s.run_checks(skipped)
        emit("result", name="run_checks", text=_short("run_checks", json.dumps(res)))
    # facts from code; the model's explanation is kept only if it is an explanation, not a loop or a to-do list
    cnt = {k: [r.check for r in s.results.values() if r.status == k] for k in ("PASS", "WARN", "FAIL", "SKIP")}
    facts = (f"{len(s.results)} checks: {len(cnt['PASS'])} PASS, {len(cnt['WARN'])} WARN, {len(cnt['FAIL'])} FAIL, "
             f"{len(cnt['SKIP'])} SKIP." + (f" FAIL: {', '.join(cnt['FAIL'])}." if cnt["FAIL"] else "")
             + (f" WARN: {', '.join(cnt['WARN'])}." if cnt["WARN"] else ""))
    lines = [x.strip() for x in final.splitlines() if x.strip()]
    mentions = any(c in final for c in checks.CHECKS) or any(w in final for w in ("PASS", "FAIL", "WARN"))
    dropped = None
    if lines and (len(set(lines)) / len(lines) < 0.7 or not mentions):
        dropped = "repetitive" if len(set(lines)) / len(lines) < 0.7 else "does not discuss any check result"
        emit("warn", text=f"the model's explanation was dropped ({dropped}); the facts below are from code")
        final = ""
    if final:
        emit("stage", stage="Summary", text="explanation written")
    # number guard: every number the model states must appear somewhere in what the tools returned
    evidence = json.dumps([e["result"] for e in s.log], default=str)
    nums = set(re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?", final))
    unverified = sorted(n for n in nums if n.rstrip("%").replace(",", "") not in evidence
                        and len(n.rstrip("%").replace(",", "")) > 1)
    v = report.verdict(list(s.results.values()))
    report.write(list(s.results.values()), outp, str(root.resolve()), v, s.rule_notes)
    md = [f"# dsgate agent report — {root.name}", "",
          f"**VERDICT (computed from the checks, not by the model): {v}**", "",
          "| check | status | summary |", "|---|---|---|"]
    md += [f"| {r.check} | {r.status} | {r.summary.replace('|', '/')} |" for r in s.results.values()]
    md += ["", "## Summary (from code)", "", facts]
    if skipped:
        md += ["", f"_The model skipped {', '.join(skipped)}; code ran them._"]
    md += ["", "## The agent's explanation", "",
           final.strip() or (f"(dropped: {dropped})" if dropped else "(no explanation produced)")]
    if unverified:
        md += ["", f"> **Number guard:** these numbers in the explanation do not appear in any tool result and "
                   f"may be wrong: {', '.join(unverified)}. Trust the table above."]
    if s.rule_notes:
        md += ["", "## Your rules applied (by code)", ""] + [f"- {n}" for n in s.rule_notes]
    if s.plan:
        md += ["", "## Fix plan (proposed, not applied)", "", s.plan.strip()]
    md += ["", f"_LLM: {llm} · tool calls: {steps} · fallback: {fallback} · {time.time() - t0:.0f} s_"]
    (outp / "agent_report.md").write_text("\n".join(md), encoding="utf-8")
    (outp / "agent_log.json").write_text(json.dumps(s.log, indent=1, default=str), encoding="utf-8")
    emit("stage", stage="Verdict", text=f"{v}: computed by code from the check results, not by the model")
    if verbose and not on_event:
        print(f"\nVERDICT {v} — {outp / 'agent_report.md'}")
    return {"verdict": v, "steps": steps, "fallback": fallback, "explanation": final, "out": str(outp),
            "unverified_numbers": unverified, "stalled": stalled, "facts": facts, "skipped_by_model": skipped,
            "explanation_dropped": dropped, "seconds": round(time.time() - t0, 1)}
