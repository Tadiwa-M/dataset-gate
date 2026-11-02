"""dsgate ui — a local web page to run the gate and watch the agent work (standard library only).

    dsgate ui                 # http://127.0.0.1:8765
Pick a dataset folder (+ optional manifest / detector / LLM), choose Check, Agent or Bench, and the
steps stream in live (each check's status, each tool the agent calls), then the report appears.
Binds to 127.0.0.1 only. One job at a time; jobs run in a background thread.
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
from urllib.parse import parse_qs, urlparse

from . import checks, report
from .core import load

STATE = {"running": False, "kind": None, "events": [], "out": None, "verdict": None, "error": None, "started": None}
LOCK = threading.Lock()


def _event(kind: str, text: str, status: str | None = None):
    with LOCK:
        STATE["events"].append({"t": round(time.time() - (STATE["started"] or time.time()), 1),
                                "kind": kind, "text": text, "status": status})


class _Tee(io.TextIOBase):
    """Forward printed lines (the agent's 'agent → tool(...)' lines, bench progress) as events."""

    def write(self, s):
        for line in s.splitlines():
            if line.strip():
                _event("log", line.strip())
        return len(s)


def _run(kind: str, p: dict):
    out = Path(p.get("out") or (Path(p["dataset"]) / "_dsgate"))
    try:
        if kind == "check":
            ds = load(p["dataset"], p.get("manifest") or None)
            opts = {"out_dir": str(out), "model": p.get("model") or None, "device": "cpu"}
            out.mkdir(parents=True, exist_ok=True)
            res = []
            for name, fn in checks.CHECKS.items():
                _event("step", f"running {name} …")
                try:
                    r = fn(ds, opts)
                except Exception as exc:  # noqa: BLE001
                    r = checks.Result(name, "FAIL", f"check crashed: {type(exc).__name__}: {exc}")
                res.append(r)
                _event("check", f"{name}: {r.summary}", r.status)
            v = report.verdict(res)
            report.write(res, out, str(Path(p["dataset"]).resolve()), v)
        elif kind == "agent":
            from .agent import run_agent
            with redirect_stdout(_Tee()):
                r = run_agent(p["dataset"], p.get("llm") or "qwen2.5:3b", p.get("model") or None, "cpu", str(out))
            v = r["verdict"]
        elif kind == "bench":
            from .bench import bench
            work = Path(p.get("out") or (Path(p["dataset"]).parent / "_dsgate_bench"))
            with redirect_stdout(_Tee()):
                r = bench(p["dataset"], str(work), p.get("manifest") or None, p.get("model") or None)
            out = work / "faulty" / "_dsgate"
            v = f"caught {r['caught']}/{r['testable']} planted faults, {len(r['false_alarms_on_clean'])} false alarms"
        else:
            raise ValueError(kind)
        with LOCK:
            STATE.update(out=str(out), verdict=v)
        _event("done", f"finished: {v}")
    except Exception as exc:  # noqa: BLE001
        with LOCK:
            STATE["error"] = f"{type(exc).__name__}: {exc}"
        _event("error", traceback.format_exc(limit=3))
    finally:
        with LOCK:
            STATE["running"] = False


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Dataset Gate</title><style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#7d7a72;--line:#e3e0d8;--card:#fff;--acc:#2d5bd7;--ok:#2f7a45;--warn:#b07a12;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#16171a;--fg:#e9e7e2;--muted:#9a978f;--line:#2e3034;--card:#1f2125;--acc:#7aa2ff}}
:root[data-theme="dark"]{--bg:#16171a;--fg:#e9e7e2;--muted:#9a978f;--line:#2e3034;--card:#1f2125;--acc:#7aa2ff}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1080px;margin:0 auto;padding:22px 16px} h1{font-size:22px;margin:0} .m{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:14px 0}
label{display:block;font-size:13px;color:var(--muted);margin-top:8px} input{width:100%;padding:8px 10px;border:1px solid var(--line);
border-radius:8px;background:transparent;color:var(--fg);font:inherit}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px} @media (max-width:640px){.row{grid-template-columns:1fr}}
button{margin:12px 8px 0 0;padding:9px 16px;border-radius:8px;border:1px solid var(--line);background:var(--acc);color:#fff;
font:inherit;cursor:pointer} button.sec{background:transparent;color:var(--fg)} button:disabled{opacity:.5;cursor:default}
#log{font:13px/1.55 ui-monospace,Consolas,monospace;max-height:340px;overflow:auto;white-space:pre-wrap}
.ev{padding:2px 0;border-bottom:1px dashed var(--line)} .t{color:var(--muted);margin-right:8px}
.pill{display:inline-block;min-width:44px;text-align:center;color:#fff;border-radius:999px;font-size:11px;padding:0 6px;margin-right:6px}
.PASS{background:var(--ok)} .WARN{background:var(--warn)} .FAIL{background:var(--bad)} .SKIP{background:var(--muted)}
.verdict{font-size:26px;font-weight:700} .BLOCKED{color:var(--bad)} .PASSED{color:var(--ok)}
iframe{width:100%;height:70vh;border:1px solid var(--line);border-radius:8px;background:#fff}
.tabs button{margin-top:0}
</style></head><body><main>
<h1>Dataset Gate</h1><p class="m">Check a detection dataset for leakage, bad labels and suspicious empty tiles before training.
Runs locally; nothing leaves this machine.</p>
<div class="card"><label>Dataset folder (contains data.yaml, images/, labels/)</label><input id="dataset" placeholder="C:\\path\\to\\dataset">
<div class="row"><div><label>Manifest CSV (optional: frame, capture_time, lat, lon)</label><input id="manifest"></div>
<div><label>Detector .pt (optional: enables missing-label and class-agreement checks)</label><input id="model"></div></div>
<div class="row"><div><label>Local LLM for the agent (Ollama)</label><input id="llm" value="qwen2.5:3b"></div>
<div><label>Output folder (optional)</label><input id="out"></div></div>
<button onclick="go('check')">Check</button><button onclick="go('agent')">Run agent</button>
<button class="sec" onclick="go('bench')">Benchmark (plant faults)</button></div>
<div class="card"><b id="status">Idle</b><div id="log" class="m">Steps appear here as they run.</div></div>
<div class="card" id="result" hidden><div id="verdict" class="verdict"></div>
<div class="tabs"><button class="sec" onclick="show('report.html')">Report</button>
<button class="sec" onclick="show('agent_report.md')">Agent explanation</button>
<button class="sec" onclick="show('class_map_sheet.jpg')">Class sample sheet</button>
<button class="sec" onclick="show('missing_labels_sheet.jpg')">Suspicious empty tiles</button></div>
<iframe id="frame" title="report"></iframe></div>
</main><script>
const $ = id => document.getElementById(id);
try { const s = JSON.parse(localStorage.getItem("dsgate-form") || "{}");
  for (const k of ["dataset","manifest","model","llm","out"]) if (s[k]) $(k).value = s[k]; } catch (e) {}
let timer = null, seen = 0;
async function go(kind) {
  const body = {kind}; for (const k of ["dataset","manifest","model","llm","out"]) body[k] = $(k).value.trim();
  try { localStorage.setItem("dsgate-form", JSON.stringify(body)); } catch (e) {}
  if (!body.dataset) { $("status").textContent = "Enter a dataset folder first."; return; }
  const r = await fetch("/api/run", {method: "POST", body: JSON.stringify(body)}); const j = await r.json();
  if (!r.ok) { $("status").textContent = j.error; return; }
  $("log").innerHTML = ""; seen = 0; $("result").hidden = true; poll();
}
function esc(s){return s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
async function poll() {
  clearTimeout(timer);
  const j = await (await fetch("/api/state")).json();
  const ev = j.events.slice(seen); seen = j.events.length;
  for (const e of ev) {
    const d = document.createElement("div"); d.className = "ev";
    d.innerHTML = `<span class="t">${e.t.toFixed(1)}s</span>` + (e.status ? `<span class="pill ${e.status}">${e.status}</span>` : "")
      + esc(e.text); $("log").appendChild(d); $("log").scrollTop = 1e9;
  }
  $("status").textContent = j.running ? `Running ${j.kind} …` : (j.error ? "Error: " + j.error : (j.kind ? "Finished" : "Idle"));
  if (j.running) { timer = setTimeout(poll, 700); return; }
  if (j.out) { $("result").hidden = false; const v = j.verdict || "";
    $("verdict").textContent = v; $("verdict").className = "verdict " + (v.startsWith("BLOCKED") ? "BLOCKED" : v.startsWith("PASSED") ? "PASSED" : "");
    show(j.kind === "agent" ? "agent_report.md" : "report.html"); }
}
function show(f) { $("frame").src = "/out/" + f + "?t=" + Date.now(); }
poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
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
            with LOCK:
                return self._send(200, json.dumps(STATE, default=str).encode(), "application/json")
        if u.path.startswith("/out/"):
            with LOCK:
                out = STATE.get("out")
            if not out:
                return self._send(404, b"no results yet", "text/plain")
            base = Path(out).resolve()
            f = (base / u.path[len("/out/"):]).resolve()
            if base not in f.parents or not f.is_file():                 # serve only files inside the report folder
                return self._send(404, b"not found (not produced by this run)", "text/plain")
            if f.suffix == ".md":                                        # render markdown as readable text
                body = f"<!doctype html><meta charset=utf-8><body style='font:14px/1.55 system-ui;padding:16px;" \
                       f"white-space:pre-wrap;max-width:900px'>{html.escape(f.read_text(encoding='utf-8'))}</body>"
                return self._send(200, body.encode(), "text/html; charset=utf-8")
            return self._send(200, f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            return self._send(404, b"{}", "application/json")
        p = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        kind = p.get("kind")
        if kind not in ("check", "agent", "bench"):
            return self._send(400, json.dumps({"error": "unknown job"}).encode(), "application/json")
        ds = Path(p.get("dataset", ""))
        if not (ds / "images").is_dir():
            return self._send(400, json.dumps({"error": f"'{ds}' has no images/ folder"}).encode(), "application/json")
        with LOCK:
            if STATE["running"]:
                return self._send(409, json.dumps({"error": "a job is already running"}).encode(), "application/json")
            STATE.update(running=True, kind=kind, events=[], out=None, verdict=None, error=None, started=time.time())
        threading.Thread(target=_run, args=(kind, p), daemon=True).start()
        self._send(200, b'{"ok": true}', "application/json")


def serve(port: int = 8765):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"dsgate ui on http://127.0.0.1:{port}  (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
