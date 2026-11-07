"""Write the gate's verdict as report.json (for machines / the agent) and report.html (for people)."""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

from .core import Result

COLORS = {"PASS": "var(--ok)", "WARN": "var(--warn)", "FAIL": "var(--bad)", "SKIP": "var(--muted)"}


def verdict(results: list[Result], fail_on_warn: bool = False) -> str:
    if any(r.status == "FAIL" for r in results):
        return "BLOCKED"
    if fail_on_warn and any(r.status == "WARN" for r in results):
        return "BLOCKED"
    return "PASSED"


def write(results: list[Result], out_dir: Path, dataset: str, v: str,
          rule_notes: list[str] | None = None) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    data = {"dataset": dataset, "verdict": v, "time": stamp, "rules_applied": rule_notes or [],
            "results": [{"check": r.check, "status": r.status, "summary": r.summary,
                         "details": r.details, "artifacts": r.artifacts} for r in results]}
    jp = out_dir / "report.json"
    jp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    rows = []
    for r in results:
        arts = "".join(f'<div><img src="{html.escape(Path(a).name)}" alt="" loading="lazy"></div>'
                       for a in r.artifacts if a.lower().endswith((".jpg", ".png")))
        det = html.escape(json.dumps(r.details, indent=1, default=str))
        rows.append(f"""<section><h2><span class="pill" style="background:{COLORS[r.status]}">{r.status}</span>
{html.escape(r.check)}</h2><p>{html.escape(r.summary)}</p>{arts}
<details><summary>details</summary><pre>{det}</pre></details></section>""")
    hp = out_dir / "report.html"
    hp.write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Dataset Gate Report</title><style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#8a877f;--line:#e3e0d8;--ok:#2f7a45;--warn:#b07a12;--bad:#b3261e}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#17181a;--fg:#e9e7e2;--muted:#8d8a83;--line:#2e3033}}}}
:root[data-theme="dark"]{{--bg:#17181a;--fg:#e9e7e2;--muted:#8d8a83;--line:#2e3033}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:960px;margin:0 auto;padding:24px 16px}} h1{{margin:0 0 4px;font-size:22px}}
section{{border-top:1px solid var(--line);padding:12px 0}} h2{{font-size:16px;margin:0 0 4px}}
.pill{{color:#fff;border-radius:999px;padding:1px 9px;font-size:12px;margin-right:6px}}
.big{{font-size:28px;font-weight:700;color:{'var(--bad)' if v == 'BLOCKED' else 'var(--ok)'}}}
img{{max-width:100%;border:1px solid var(--line);border-radius:6px;margin:6px 0}}
pre{{font-size:12px;overflow-x:auto}} .m{{color:var(--muted)}}
</style></head><body><main><h1>Dataset gate report</h1>
<p class="m">{html.escape(dataset)} · {stamp}</p><p class="big">{v}</p>
{''.join(f'<p class="m">Your rule applied: {html.escape(n)}</p>' for n in rule_notes or [])}
{''.join(rows)}</main></body></html>""", encoding="utf-8")
    return jp, hp
