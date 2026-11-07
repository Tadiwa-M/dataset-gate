"""dsgate ui — the Dataset Gate console: run the gate or the agent and watch every step (standard library only).

    dsgate ui                 # http://127.0.0.1:8765
Drop a dataset folder or .zip (or paste a path), optionally a manifest and a detector, then Run the agent or
Quick check. A live card shows which stage the agent is in, what the model is writing and why, with a Stop
button; the activity feed keeps the full history; your rules file is shown and applied. Binds to 127.0.0.1.
"""
from __future__ import annotations

import html
import io
import json
import mimetypes
import threading
import time
import traceback
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import checks, report, rules as rules_mod
from .core import load
from .web import KIT_CSS, KIT_JS, Live, finish_upload, save_upload

HOME = Path.cwd()
LIVE = Live()
RESULT: dict = {"out": None, "verdict": None, "kind": None, "results": [], "explanation": "", "fix_plan": "",
                "rules_applied": [], "dataset": None, "error": None}


class _Cancelled(Exception):
    pass


class _Tee(io.TextIOBase):
    """Bench progress lines go to the feed."""

    def write(self, s):
        for line in s.splitlines():
            if line.strip():
                LIVE.act("data", line.strip(), "info", "result")
        return len(s)


def _run(kind: str, p: dict):
    out = Path(p.get("out") or (Path(p["dataset"]) / "_dsgate"))
    rules = rules_mod.load()
    res_list, expl, plan, notes = [], "", "", []
    try:
        if kind == "check":
            LIVE.stage("Load", f"reading {Path(p['dataset']).name}")
            ds = load(p["dataset"], p.get("manifest") or None)
            out.mkdir(parents=True, exist_ok=True)
            LIVE.stage("Run checks", "running every check")

            def each(r):
                LIVE.act("data", f"{r.status} {r.check}: {r.summary[:160]}",
                         {"FAIL": "bad", "WARN": "warn"}.get(r.status, "info"), "result")
                if LIVE.should_stop():
                    raise _Cancelled()
            res_list, notes = checks.run(ds, {"out_dir": str(out), "model": p.get("model") or None, "device": "cpu"},
                                         rules=rules, on_result=each)
            for n in notes:
                LIVE.act("data", f"your rule applied: {n}", "info", "stage")
            v = report.verdict(res_list)
            report.write(res_list, out, str(Path(p["dataset"]).resolve()), v, notes)
            LIVE.stage("Verdict", f"{v}: computed by code from the check results")
        elif kind == "agent":
            from .agent import Cancelled, run_agent
            try:
                r = run_agent(p["dataset"], p.get("llm") or "qwen2.5:3b", p.get("model") or None, "cpu", str(out),
                              verbose=False, on_event=LIVE.on_event("data"), should_stop=LIVE.should_stop,
                              manifest=p.get("manifest") or None)
            except Cancelled:
                raise _Cancelled() from None
            v = r["verdict"]
            rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
            notes = rep.get("rules_applied", [])
            res_list = [checks.Result(x["check"], x["status"], x["summary"], x["details"], x["artifacts"])
                        for x in rep["results"]]
            expl = r["facts"] + ("\n\nThe model's explanation:\n" + r["explanation"] if r["explanation"] else
                                 f"\n\n(model's explanation dropped: {r['explanation_dropped']})"
                                 if r["explanation_dropped"] else "")
            if r["skipped_by_model"]:
                expl += f"\n\nThe model skipped {', '.join(r['skipped_by_model'])}; code ran them."
            plan = (out / "fix_plan.md").read_text(encoding="utf-8") if (out / "fix_plan.md").exists() else ""
            if r.get("unverified_numbers"):
                LIVE.act("data", f"number guard: {', '.join(r['unverified_numbers'])} in the explanation are not in "
                                 "any tool result — trust the table", "warn", "warn")
        elif kind == "bench":
            from .bench import bench
            work = Path(p.get("out") or (Path(p["dataset"]).parent / "_dsgate_bench"))
            LIVE.stage("Benchmark", "planting known faults in a copy and checking that the gate catches them")
            with redirect_stdout(_Tee()):
                r = bench(p["dataset"], str(work), p.get("manifest") or None, p.get("model") or None)
            out = work / "faulty" / "_dsgate"
            v = f"caught {r['caught']}/{r['testable']} planted faults, {len(r['false_alarms_on_clean'])} false alarms"
            LIVE.stage("Score", v)
        else:
            raise ValueError(kind)
        RESULT.update(out=str(out), verdict=v, kind=kind, explanation=expl, fix_plan=plan, rules_applied=notes,
                      dataset=p["dataset"], error=None,
                      results=[{"check": r.check, "status": r.status, "summary": r.summary, "details": r.details,
                                "artifacts": [Path(a).name for a in r.artifacts]} for r in res_list])
        LIVE.act("data", f"finished: {v}", "good" if str(v).startswith(("PASSED", "caught")) else "bad", "stage")
    except _Cancelled:
        LIVE.act("you", "stopped at your request — nothing was changed", "warn", "you")
    except Exception as exc:  # noqa: BLE001
        RESULT["error"] = f"{type(exc).__name__}: {exc}"
        LIVE.act("data", f"error: {type(exc).__name__}: {exc}", "bad", "warn")
        LIVE.act("data", traceback.format_exc(limit=3), "bad", "note")
    finally:
        LIVE.finish()


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Dataset Gate</title><style>""" + KIT_CSS + """
.logo{background:#16805f}
</style></head><body>
<header class="top"><div class="brand"><span class="logo">✓</span>Dataset Gate</div>
<span class="hint">checks a detection dataset before anyone trains on it · runs locally</span>
<span style="margin-left:auto" id="rulesChip" class="pill info"></span></header>
<div class="wrap"><main>
<div class="card now" id="now" hidden></div>
<div class="card"><h2>Dataset</h2><p class="m">The agent looks at the folder, picks the checks, reads the evidence for anything
that fails and writes a fix plan. The verdict is computed by code, never by the model.</p>
<label>Dataset folder (data.yaml, images/, labels/)</label><div class="drop" id="dz-dataset"><input id="dataset" placeholder="C:\\path\\to\\dataset"></div>
<div class="g2"><div><label>Manifest CSV <span class="hint">optional: frame, capture_time, lat, lon, task</span></label><div class="drop" id="dz-manifest"><input id="manifest"></div></div>
<div><label>Detector .pt <span class="hint">optional: missing-label and class-swap checks</span></label><div class="drop" id="dz-model"><input id="model"></div></div></div>
<details style="margin-top:10px"><summary>advanced</summary><div class="g2"><div><label>Local LLM (Ollama)</label><input id="llm" value="qwen2.5:3b"></div>
<div><label>Report folder <span class="hint">default: &lt;dataset&gt;/_dsgate</span></label><input id="out"></div></div></details>
<button class="btn" onclick="go('agent')">Run the agent</button><button class="btn sec" onclick="go('check')">Quick check (no LLM)</button>
<button class="btn sec" onclick="go('bench')">Benchmark: plant faults</button></div>
<div class="card" id="result" hidden></div>
</main>
<aside class="side"><div class="card"><h2>Activity</h2><div class="feedtools"><button class="on" onclick="Kit.filterFeed($('feed'),'all',this)">all</button>
<button onclick="Kit.filterFeed($('feed'),'data',this)">agent</button><button onclick="Kit.filterFeed($('feed'),'you',this)">you</button></div>
<div class="feed" id="feed"><div class="m">Every step shows up here as it happens.</div></div></div>
<div class="card" id="rulesCard" hidden><h2>Your rules</h2><p class="hint" id="rulesPath"></p><div id="rulesList"></div></div></aside></div>
<script>""" + KIT_JS + """
const $ = id => document.getElementById(id), esc = Kit.esc, NAMES = {data:"Data agent", you:"You", ui:"Console"};
const F = ["dataset","manifest","model","llm","out"];
Kit.onToast(t => { const d=document.createElement("div"); d.className="ev warn bad"; d.innerHTML=`<span class="ic">!</span><div class="tx">${esc(t)}</div>`; $("feed").appendChild(d); $("feed").scrollTop=1e9; });
try { const s = JSON.parse(localStorage.getItem("dsgate-form") || "{}"); for (const k of F) if (s[k]) $(k).value = s[k]; } catch (e) {}
function save(){ try { const s={}; for (const k of F) s[k]=$(k).value; localStorage.setItem("dsgate-form", JSON.stringify(s)); } catch (e) {} }
document.querySelectorAll("input").forEach(e => e.addEventListener("change", save));
Kit.dropField($("dz-dataset"), "dataset"); Kit.dropField($("dz-manifest"), "csv"); Kit.dropField($("dz-model"), "pt");
async function go(kind){ save(); const b = {kind}; for (const k of F) b[k] = $(k).value.trim();
  if (!b.dataset) return Kit.api("/api/run", b); await Kit.api("/api/run", b); poll(); }
function resultHtml(R){
  const rows = R.results.map(r => `<div class="row"><span class="pill ${r.status}">${r.status}</span><div><b>${esc(r.check)}</b> — ${esc(r.summary)}
    ${r.details && Object.keys(r.details).length ? `<details><summary>evidence</summary><pre>${esc(JSON.stringify(r.details,null,1)).slice(0,4000)}</pre></details>` : ""}
    ${(r.artifacts||[]).filter(a => /\\.(jpg|png)$/i.test(a)).map(a => `<img class="sheet" loading="lazy" src="/out/${encodeURIComponent(a)}?t=${Date.now()}" alt="${esc(r.check)} sheet">`).join("")}</div></div>`).join("");
  const v = String(R.verdict||""), cls = v.startsWith("BLOCKED") ? "BLOCKED" : v.startsWith("PASSED") ? "PASSED" : "";
  return `<div class="verdict ${cls}">${esc(v)}</div><div class="m">${esc(R.dataset)} · <a href="/out/report.html" target="_blank">full report</a></div>
    ${cls==="BLOCKED" ? `<div class="note">Training is blocked until the FAIL items are fixed.</div>` : ""}
    ${R.rules_applied.map(n => `<div class="rule">Your rule applied: ${esc(n)}</div>`).join("")}
    ${R.explanation ? `<h3>The agent's explanation</h3><pre>${esc(R.explanation)}</pre>` : ""}
    ${R.fix_plan ? `<details><summary>fix plan (proposed, not applied)</summary><pre>${esc(R.fix_plan)}</pre></details>` : ""}
    <h3>Checks</h3>${rows}`; }
let lastRes = "", timer = null;
async function poll(){ clearTimeout(timer); const j = await Kit.api("/api/state");
  if (j) { Kit.renderNow($("now"), j.now, NAMES, j.cancelling); Kit.renderFeed($("feed"), j.activity, NAMES);
    document.querySelectorAll("main .btn").forEach(b => { if (!b.closest(".now")) b.disabled = !!j.busy; });
    const key = JSON.stringify([j.result.verdict, j.result.out, j.result.results.length, j.result.explanation.length]);
    if (j.result.out && key !== lastRes) { lastRes = key; $("result").hidden = false; $("result").innerHTML = resultHtml(j.result); }
    $("rulesChip").textContent = j.rules.name ? "rules: " + j.rules.name : "no rules file";
    $("rulesCard").hidden = !j.rules.lines.length; $("rulesPath").textContent = j.rules.path || "";
    $("rulesList").innerHTML = j.rules.lines.map(l => `<div class="rule">${esc(l)}</div>`).join(""); }
  timer = setTimeout(poll, j && j.busy ? 600 : 2500); }
poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body, ctype: str = "application/json"):
        body = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            try:
                r = rules_mod.load()
                rules = {"name": r.get("name") or ("your rules" if r else ""), "path": r.get("_path", ""),
                         "lines": rules_mod.summary(r, "data")}
            except Exception as exc:  # noqa: BLE001 — a broken rules file must be visible, not silent
                rules = {"name": "rules file has an error", "path": str(rules_mod.path()), "lines": [str(exc)]}
            return self._send(200, {**LIVE.snapshot(), "result": RESULT, "rules": rules})
        if u.path.startswith("/out/"):
            out = RESULT.get("out")
            if not out:
                return self._send(404, b"no results yet", "text/plain")
            base = Path(out).resolve()
            f = (base / unquote(u.path[len("/out/"):])).resolve()
            if base not in f.parents or not f.is_file():                 # serve only files inside the report folder
                return self._send(404, b"not found (not produced by this run)", "text/plain")
            if f.suffix == ".md":
                body = f"<!doctype html><meta charset=utf-8><body style='font:14px/1.55 system-ui;padding:16px;" \
                       f"white-space:pre-wrap;max-width:900px'>{html.escape(f.read_text(encoding='utf-8'))}</body>"
                return self._send(200, body.encode(), "text/html; charset=utf-8")
            return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self._send(404, {"error": "not found"})

    def do_PUT(self):
        u = urlparse(self.path)
        if u.path != "/api/upload":
            return self._send(404, {"error": "not found"})
        q = parse_qs(u.query)
        try:
            save_upload(HOME, q.get("id", [""])[0], q.get("rel", [""])[0], self.rfile,
                        int(self.headers.get("Content-Length", 0)))
        except (ValueError, OSError) as exc:
            return self._send(400, {"error": str(exc)})
        self._send(200, {"ok": True})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            p = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {"error": "malformed request"})
        if path == "/api/cancel":
            LIVE.cancel = bool(LIVE.busy)
            if LIVE.busy:
                LIVE.act("you", "asked to stop — the agent stops after its current step", "warn", "you")
            return self._send(200, {"ok": True})
        if path == "/api/upload/done":
            try:
                r = finish_upload(HOME, str(p.get("id", "")), str(p.get("want", "dataset")))
            except (ValueError, OSError, KeyError) as exc:
                return self._send(400, {"error": str(exc)})
            LIVE.act("you", f"dropped files → {r['path']} ({r['files']} files)", "info", "you")
            return self._send(200, r)
        if path != "/api/run":
            return self._send(404, {"error": "not found"})
        kind = p.get("kind")
        if kind not in ("check", "agent", "bench"):
            return self._send(400, {"error": "unknown job"})
        if not str(p.get("dataset", "")).strip():
            return self._send(400, {"error": "drop a dataset folder or paste its path first"})
        if str(p.get("dataset", "")).lower().startswith(("http:", "https:")):
            return self._send(400, {"error": "datasets must be local folders"})
        ds = Path(p["dataset"])
        if not (ds / "images").is_dir():
            return self._send(400, {"error": f"'{ds}' has no images/ folder — is it a YOLO dataset?"})
        stages = {"check": ["Load", "Run checks", "Verdict"], "bench": ["Benchmark", "Score"]}
        from .agent import STAGES
        if not LIVE.start({"check": "quick check", "agent": "auditing the dataset", "bench": "benchmark"}[kind],
                          "data", stages.get(kind, STAGES)):
            return self._send(409, {"error": "a job is already running — wait for it or press Stop"})
        LIVE.act("you", f"started {kind} on {ds.name}", "info", "you")
        threading.Thread(target=_run, args=(kind, p), daemon=True).start()
        self._send(200, {"ok": True})


def serve(port: int = 8765, home: Path | None = None):
    global HOME
    HOME = Path(home or Path.cwd()).resolve()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Dataset Gate on http://127.0.0.1:{port}  (uploads go to {HOME / 'uploads'})  close this window to stop",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
