"""Shared pieces for the local consoles (dsgate ui, and trainpilot's console, which imports this module).

Live      thread-safe page state: one job at a time, the "now" card (which agent, which stage, what the model
          is writing and why), an activity feed, and a cancel flag the agents check between steps.
uploads   drag-and-drop: files are written under <home>/uploads/<id>/, paths are sanitised, zips are
          extracted safely, and the dataset root / CSV / .pt inside is found for the form.
KIT_CSS / KIT_JS   the look and the browser-side widgets (now card, feed, drop fields), inlined in each page.
Standard library only. Servers using this bind to 127.0.0.1; nothing here talks to the network.
"""
from __future__ import annotations

import re
import shutil
import threading
import time
import zipfile
from pathlib import Path


class Live:
    def __init__(self):
        self.lock = threading.Lock()
        self.activity: list[dict] = []
        self.now: dict | None = None
        self.busy: str | None = None
        self.cancel = False

    def act(self, agent: str, text: str, level: str = "info", kind: str = "note"):
        with self.lock:
            self.activity.append({"t": time.strftime("%H:%M:%S"), "agent": agent, "text": text, "level": level,
                                  "kind": kind})
            self.activity = self.activity[-500:]

    def start(self, job: str, agent: str, stages: list[str], cancellable: bool = True) -> bool:
        with self.lock:
            if self.busy:
                return False
            self.busy, self.cancel = job, False
            self.now = {"agent": agent, "job": job, "stages": stages, "stage": stages[0] if stages else "",
                        "text": "starting", "since": time.time(), "thinking": None, "reason": None,
                        "cancellable": cancellable}
        return True

    def stage(self, stage: str, text: str = ""):
        with self.lock:
            if self.now:
                self.now.update(stage=stage, text=text, thinking=None, stage_since=time.time())

    def finish(self):
        with self.lock:
            self.busy, self.now = None, None

    def should_stop(self) -> bool:
        return self.cancel

    def on_event(self, agent: str):
        """Adapter for the agents' on_event callbacks (dsgate.agent.run_agent, trainpilot.planner)."""
        def handle(e: dict):
            k, text = e.get("kind"), e.get("text", "")
            if k == "thinking":
                with self.lock:
                    if self.now:
                        self.now["thinking"] = {"tokens": e.get("tokens", 0), "seconds": e.get("seconds", 0),
                                                "text": e.get("text", "")}
                return
            if k == "stage":
                self.stage(e.get("stage", ""), text)
                self.act(agent, f"{e.get('stage')}: {text}", "info", "stage")
            elif k == "reason":
                with self.lock:
                    if self.now:
                        self.now["reason"] = text
                self.act(agent, text, "info", "reason")
            elif k == "tool":
                self.act(agent, text, "info", "tool")
            elif k == "result":
                self.act(agent, text, "info", "result")
            elif k == "warn":
                self.act(agent, text, "warn", "warn")
        return handle

    def snapshot(self) -> dict:
        with self.lock:
            now = dict(self.now) if self.now else None
            if now:
                now["elapsed"] = round(time.time() - now["since"])
            return {"busy": self.busy, "now": now, "activity": list(self.activity), "cancelling": self.cancel}


# ── uploads ──────────────────────────────────────────────────────────────────
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _clean_rel(rel: str) -> Path:
    parts = [p for p in re.split(r"[\\/]+", rel or "") if p not in ("", ".")]
    if not parts or any(p == ".." or ":" in p or p.startswith("~") for p in parts):
        raise ValueError(f"bad file path '{rel}'")
    return Path(*parts)


def save_upload(home: Path, upload_id: str, rel: str, rfile, length: int) -> Path:
    """Stream one uploaded file to <home>/uploads/<id>/<rel>; refuses ids/paths that could escape."""
    if not _ID.match(upload_id or ""):
        raise ValueError("bad upload id")
    root = (home / "uploads" / upload_id).resolve()
    dest = (root / _clean_rel(rel)).resolve()
    if root not in dest.parents:
        raise ValueError("path escapes the upload folder")
    dest.parent.mkdir(parents=True, exist_ok=True)
    left = length
    with open(dest, "wb") as f:
        while left > 0:
            buf = rfile.read(min(1 << 20, left))
            if not buf:
                break
            f.write(buf)
            left -= len(buf)
    if left:
        dest.unlink(missing_ok=True)
        raise ValueError("upload was cut short")
    return dest


def _safe_extract(z: Path, into: Path):
    with zipfile.ZipFile(z) as zf:
        for m in zf.infolist():
            if m.is_dir():
                continue
            dest = (into / _clean_rel(m.filename)).resolve()
            if into.resolve() not in dest.parents:
                raise ValueError(f"zip member escapes: {m.filename}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)


def finish_upload(home: Path, upload_id: str, want: str) -> dict:
    """want: dataset | csv | pt. Extracts zips, then returns the path the form field should get."""
    if not _ID.match(upload_id or ""):
        raise ValueError("bad upload id")
    root = home / "uploads" / upload_id
    if not root.is_dir():
        raise ValueError("nothing was uploaded")
    for z in list(root.rglob("*.zip")):
        _safe_extract(z, z.with_suffix(""))
        z.unlink()
    if want == "dataset":
        cands = sorted((p.parent for p in root.rglob("data.yaml")), key=lambda p: len(p.parts))
        cands = cands or sorted((p.parent for p in root.rglob("images") if p.is_dir()), key=lambda p: len(p.parts))
        if not cands:
            raise ValueError("no data.yaml or images/ folder in what you dropped — is it a YOLO dataset?")
        n = sum(1 for _ in cands[0].rglob("*") if _.is_file())
        return {"path": str(cands[0].resolve()), "files": n}
    ext = {"csv": ".csv", "pt": ".pt"}[want]
    hits = sorted(root.rglob(f"*{ext}"), key=lambda p: len(p.parts))
    if not hits:
        raise ValueError(f"no {ext} file in what you dropped")
    return {"path": str(hits[0].resolve()), "files": 1}


# ── browser side ─────────────────────────────────────────────────────────────
KIT_CSS = r"""
:root{--bg:#f5f4f0;--fg:#1b1b19;--muted:#6f6c65;--line:#e1ded6;--card:#fff;--soft:#efede7;--acc:#2a57cf;--accfg:#fff;--accsoft:#e6ecfb;
--ok:#2c7a45;--oksoft:#e3f1e7;--warn:#9a6a0c;--warnsoft:#f8eed9;--bad:#b3261e;--badsoft:#f9e3e1;--shadow:0 1px 2px rgba(0,0,0,.05),0 4px 14px rgba(0,0,0,.04)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121316;--fg:#e9e7e2;--muted:#9b988f;--line:#2b2d32;--card:#1b1d21;--soft:#23252a;--acc:#86a8ff;--accfg:#0d1220;--accsoft:#1f2840;
--ok:#6cc58a;--oksoft:#18301f;--warn:#e0b252;--warnsoft:#33280f;--bad:#ff8a80;--badsoft:#3a1a18;--shadow:none}}
:root[data-theme="dark"]{--bg:#121316;--fg:#e9e7e2;--muted:#9b988f;--line:#2b2d32;--card:#1b1d21;--soft:#23252a;--acc:#86a8ff;--accfg:#0d1220;--accsoft:#1f2840;
--ok:#6cc58a;--oksoft:#18301f;--warn:#e0b252;--warnsoft:#33280f;--bad:#ff8a80;--badsoft:#3a1a18;--shadow:none}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:14.5px/1.55 "Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,sans-serif}
header.top{display:flex;align-items:center;gap:12px;padding:10px 18px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0;z-index:5;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:10px;font-weight:650;font-size:16px} .logo{width:26px;height:26px;border-radius:7px;display:grid;place-items:center;color:#fff;font-size:14px}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:18px;padding:18px;max-width:1440px;margin:0 auto}
@media (max-width:1000px){.wrap{grid-template-columns:1fr}} .side{position:sticky;top:64px;align-self:start}
@media (max-width:1000px){.side{position:static}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:16px;box-shadow:var(--shadow)}
h2{font-size:16.5px;margin:0 0 6px;font-weight:650} h3{font-size:14px;margin:14px 0 6px} .m{color:var(--muted)} .hint{font-size:12px;color:var(--muted)}
label{display:block;font-size:12.5px;color:var(--muted);margin:12px 0 4px;font-weight:550}
input,select,textarea{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:9px;background:var(--card);color:var(--fg);font:inherit}
input:focus,select:focus{outline:2px solid var(--accsoft);border-color:var(--acc)}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:12px} .g3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
@media (max-width:680px){.g2,.g3{grid-template-columns:1fr}}
.btn{display:inline-flex;align-items:center;gap:7px;margin:14px 8px 0 0;padding:9px 16px;border-radius:9px;border:1px solid var(--acc);background:var(--acc);color:var(--accfg);font:inherit;font-weight:600;cursor:pointer}
.btn.sec{background:transparent;color:var(--fg);border-color:var(--line);font-weight:550} .btn.danger{background:var(--bad);border-color:var(--bad);color:#fff}
.btn.small{padding:5px 11px;margin:0;font-size:13px} .btn:disabled{opacity:.45;cursor:not-allowed}
.pill{display:inline-block;min-width:48px;text-align:center;border-radius:999px;font-size:11.5px;font-weight:650;padding:1px 9px;margin-right:8px}
.PASS,.PASSED,.finished,.good{background:var(--oksoft);color:var(--ok)} .WARN,.warn{background:var(--warnsoft);color:var(--warn)}
.FAIL,.BLOCKED,.failed,.blocked,.bad{background:var(--badsoft);color:var(--bad)} .SKIP,.not,.info{background:var(--soft);color:var(--muted)} .running{background:var(--accsoft);color:var(--acc)}
details summary{cursor:pointer;color:var(--muted);font-size:12.5px} pre{white-space:pre-wrap;font:12px/1.5 ui-monospace,Consolas,monospace;margin:6px 0 0}
.note{border-left:3px solid var(--warn);padding:7px 11px;background:var(--warnsoft);border-radius:8px;font-size:13px;margin:8px 0}
.rule{border-left:3px solid var(--acc);padding:7px 11px;background:var(--accsoft);border-radius:8px;font-size:13px;margin:8px 0}
dialog{border:1px solid var(--line);border-radius:14px;background:var(--card);color:var(--fg);max-width:540px;width:92vw;padding:20px} dialog::backdrop{background:rgba(0,0,0,.45)}
/* now card */
.now{border:1px solid var(--acc);background:linear-gradient(0deg,var(--card),var(--accsoft));}
.now .head{display:flex;align-items:center;gap:10px;flex-wrap:wrap} .now .who{font-weight:650}
.pulse{width:9px;height:9px;border-radius:50%;background:var(--acc);box-shadow:0 0 0 0 var(--acc);animation:pu 1.4s infinite}
@keyframes pu{0%{box-shadow:0 0 0 0 rgba(42,87,207,.5)}100%{box-shadow:0 0 0 9px rgba(42,87,207,0)}}
.stepper{display:flex;gap:4px;margin:12px 0 8px;flex-wrap:wrap} .stepper span{flex:1;min-width:84px;text-align:center;font-size:12px;padding:5px 6px;border-radius:7px;background:var(--soft);color:var(--muted)}
.stepper span.done{background:var(--oksoft);color:var(--ok)} .stepper span.on{background:var(--acc);color:var(--accfg);font-weight:650}
.now .doing{font-size:14px} .think{font:12px/1.45 ui-monospace,Consolas,monospace;color:var(--muted);background:var(--soft);border-radius:8px;padding:7px 9px;margin-top:8px;max-height:90px;overflow:hidden;white-space:pre-wrap}
.quote{border-left:3px solid var(--acc);padding:4px 10px;margin-top:8px;font-style:italic}
/* feed */
.feedtools{display:flex;gap:6px;margin:6px 0 8px} .feedtools button{border:1px solid var(--line);background:transparent;color:var(--muted);border-radius:999px;padding:2px 10px;font:inherit;font-size:12px;cursor:pointer}
.feedtools button.on{background:var(--soft);color:var(--fg)}
.feed{max-height:calc(100vh - 190px);overflow:auto;font-size:13px} .ev{display:grid;grid-template-columns:18px minmax(0,1fr);gap:6px;padding:6px 0;border-bottom:1px solid var(--line)}
.ev .ic{font-size:12px;text-align:center;color:var(--muted)} .ev .meta{font-size:11px;color:var(--muted)} .ev.reason .tx{font-style:italic}
.ev.warn .tx{color:var(--warn)} .ev.bad .tx{color:var(--bad)} .ev.good .tx{color:var(--ok)} .ev.you .tx{font-weight:600} .ev.tool .tx{font-family:ui-monospace,Consolas,monospace;font-size:12px}
/* drop fields */
.drop{border:1.5px dashed var(--line);border-radius:12px;padding:10px;transition:.15s;background:var(--card)}
.drop.over{border-color:var(--acc);background:var(--accsoft)} .drop .dz{display:flex;align-items:center;gap:10px;font-size:12.5px;color:var(--muted);margin-top:7px;flex-wrap:wrap}
.drop .dz b{color:var(--fg);font-weight:600} .drop .dz a{color:var(--acc);cursor:pointer;text-decoration:underline}
.bar{height:5px;border-radius:9px;background:var(--soft);overflow:hidden;margin-top:7px} .bar i{display:block;height:100%;background:var(--acc);width:0;transition:width .2s}
.verdict{font-size:24px;font-weight:750;letter-spacing:-.01em} .verdict.PASSED{background:none;color:var(--ok)} .verdict.BLOCKED{background:none;color:var(--bad)}
.row{display:flex;align-items:flex-start;gap:4px;padding:9px 0;border-bottom:1px solid var(--line)} .row:last-child{border:0}
img.sheet{max-width:100%;border:1px solid var(--line);border-radius:9px;margin-top:8px}
"""

KIT_JS = r"""
const Kit = (() => {
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
  const ICON = {stage:"◆", tool:"→", result:"✓", reason:"“", warn:"!", you:"●", note:"·"};
  let feedFilter = "all", seen = 0, toastFn = null;
  async function api(path, body) {
    const r = await fetch(path, body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    const j = await r.json().catch(() => ({error:"bad response"}));
    if (!r.ok && toastFn) toastFn(j.error || "request failed");
    return r.ok ? j : null;
  }
  function onToast(f) { toastFn = f; }
  function fmtS(s) { s = Math.max(0, s|0); return s < 60 ? s + " s" : Math.floor(s/60) + " min " + (s%60) + " s"; }
  function renderNow(el, now, names, cancelling) {
    if (!now) { el.hidden = true; el.innerHTML = ""; return; }
    el.hidden = false;
    const i = Math.max(0, now.stages.indexOf(now.stage));
    const steps = now.stages.map((s, k) => `<span class="${k < i ? "done" : k === i ? "on" : ""}">${k < i ? "✓ " : ""}${esc(s)}</span>`).join("");
    const th = now.thinking;
    el.innerHTML = `<div class="head"><span class="pulse"></span><span class="who">${esc(names[now.agent] || now.agent)}</span>
      <span class="m">· ${esc(now.job)} · ${fmtS(now.elapsed)}</span><span style="margin-left:auto"></span>
      ${now.cancellable ? `<button class="btn danger small" ${cancelling ? "disabled" : ""} onclick="Kit.cancel()">${cancelling ? "Stopping…" : "Stop"}</button>` : ""}</div>
      <div class="stepper">${steps}</div>
      <div class="doing"><b>Now:</b> ${esc(now.text || now.stage)}</div>
      ${th ? `<div class="hint">the model is writing · ${fmtS(th.seconds)} · ${th.tokens} tokens</div>${th.text ? `<div class="think">${esc(th.text.slice(-300))}</div>` : ""}` : ""}
      ${now.reason ? `<div class="quote"><span class="hint">its reasoning:</span> ${esc(now.reason)}</div>` : ""}`;
  }
  async function cancel() { await api("/api/cancel", {}); }
  function renderFeed(el, activity, names) {
    const fresh = activity.slice(seen);
    if (seen === 0 && fresh.length) el.innerHTML = "";
    if (activity.length < seen) { el.innerHTML = ""; seen = 0; return renderFeed(el, activity, names); }
    seen = activity.length;
    const atBottom = el.scrollTop + el.clientHeight > el.scrollHeight - 40;
    for (const e of fresh) {
      const d = document.createElement("div");
      d.className = `ev ${e.kind} ${e.level}`; d.dataset.agent = e.agent;
      d.innerHTML = `<span class="ic">${ICON[e.kind] || "·"}</span><div><div class="meta">${e.t} · ${esc(names[e.agent] || e.agent)}</div><div class="tx">${esc(e.text)}</div></div>`;
      d.hidden = feedFilter !== "all" && feedFilter !== e.agent;
      el.appendChild(d);
    }
    if (fresh.length && atBottom) el.scrollTop = 1e9;
  }
  function filterFeed(el, who, btn) {
    feedFilter = who;
    btn.parentElement.querySelectorAll("button").forEach(b => b.classList.toggle("on", b === btn));
    el.querySelectorAll(".ev").forEach(d => d.hidden = who !== "all" && d.dataset.agent !== who);
  }
  // drop fields: drop a folder / .zip / file, or browse, or paste a path (pasting copies nothing)
  async function walk(entry, prefix, out) {
    if (entry.isFile) { out.push(await new Promise((res, rej) => entry.file(f => res({f, rel: prefix + entry.name}), rej))); return; }
    const rd = entry.createReader(); let batch;
    do { batch = await new Promise((res, rej) => rd.readEntries(res, rej));
         for (const e of batch) await walk(e, prefix + entry.name + "/", out); } while (batch.length);
  }
  async function upload(files, want, zone, input, onDone) {
    const id = Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
    const total = files.reduce((a, x) => a + x.f.size, 0) || 1; let sent = 0, done = 0, failed = null;
    const bar = zone.querySelector(".bar i"), msg = zone.querySelector(".msg");
    zone.querySelector(".bar").hidden = false;
    const q = files.slice();
    async function worker() { while (q.length && !failed) { const x = q.shift();
      const r = await fetch(`/api/upload?id=${id}&rel=${encodeURIComponent(x.rel)}`, {method:"PUT", body:x.f});
      if (!r.ok) { failed = (await r.json().catch(() => ({}))).error || "upload failed"; return; }
      sent += x.f.size; done++; bar.style.width = (100 * sent / total) + "%"; msg.textContent = `uploading ${done}/${files.length} files…`; } }
    await Promise.all([1,2,3,4,5,6].map(worker));
    if (failed) { msg.textContent = failed; zone.querySelector(".bar").hidden = true; return; }
    msg.textContent = "unpacking…";
    const j = await api("/api/upload/done", {id, want});
    zone.querySelector(".bar").hidden = true;
    if (j) { input.value = j.path; input.dispatchEvent(new Event("change")); msg.textContent = `✓ copied ${j.files} file(s) into the console's uploads folder`; if (onDone) onDone(j.path); }
    else msg.textContent = "could not use what you dropped";
  }
  function dropField(zone, want, onDone) {
    const input = zone.querySelector("input");
    const what = want === "dataset" ? "a dataset folder or .zip" : want === "csv" ? "a .csv" : "a .pt file";
    zone.insertAdjacentHTML("beforeend", `<div class="dz"><span>⇪ <b>Drop ${what} here</b> · <a>browse</a> · or paste a path above (nothing is copied)</span><span class="msg"></span></div><div class="bar" hidden><i></i></div>`);
    const picker = document.createElement("input"); picker.type = "file"; picker.hidden = true;
    if (want === "dataset") picker.webkitdirectory = true; else picker.accept = want === "csv" ? ".csv" : ".pt";
    zone.appendChild(picker);
    zone.querySelector(".dz a").onclick = () => picker.click();
    picker.onchange = () => { const files = [...picker.files].map(f => ({f, rel: f.webkitRelativePath || f.name})); if (files.length) upload(files, want, zone, input, onDone); };
    zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("over"));
    zone.addEventListener("drop", async e => { e.preventDefault(); zone.classList.remove("over");
      const items = [...e.dataTransfer.items].map(i => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
      const files = []; zone.querySelector(".msg").textContent = "reading…";
      for (const it of items) await walk(it, "", files);
      if (!files.length) { zone.querySelector(".msg").textContent = "nothing to upload"; return; }
      upload(files, want, zone, input, onDone); });
  }
  return {esc, api, onToast, renderNow, renderFeed, filterFeed, dropField, cancel, fmtS};
})();
"""
