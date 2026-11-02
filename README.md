# dsgate — a pre-training gate for detection datasets

`dsgate check <dataset>` audits a YOLO-format dataset before you spend GPU hours on it, writes an
HTML + JSON report, and **exits 1 (BLOCKED) if anything would make the run's numbers lie**:

```bash
dsgate check data/my_set --manifest frames.csv && python train.py     # train only if the gate passes
```

Every check comes from a real failure on aerial wildlife survey data:

| Check | Blocks on | Real failure it encodes |
|---|---|---|
| `structure` | missing train/val, labels without images | — |
| `labels` | class id out of range, boxes outside the image, zero-size boxes | — |
| `class_balance` | a class with 0 instances in val/test | silently drops out of mAP |
| `class_map` | (writes a per-class sample sheet to eyeball) | swapped id→name maps |
| `group_leakage` | tiles of one source frame in two splits | tile-level splitting |
| `time_leakage` | cross-split frames < N s apart (`--manifest` capture_time) | survey frames 2 s apart = same animals; a 0.65 mAP that was really 0.37 |
| `gps_leakage` | cross-split frames < N m apart (`--manifest` lat/lon) | adjacent transects / re-flights |
| `duplicates` | byte-identical or near-identical (dHash) images across splits | — |
| `missing_labels`* | (warns) empty tiles where a detector sees a herd | buffalo herds in "searched, empty" tiles |
| `class_agreement`* | a class the detector mostly calls another class | Kenya giraffes trained as "zebra" |

\* needs a trusted detector: `pip install dsgate[detector]` and `--model weights.pt`. The model is a
second opinion that flags things for a person; it never decides on its own.

Checks are deterministic code — no LLM decides whether data passes. A check that crashes counts as
FAIL, and a check whose inputs are missing reports SKIP instead of quietly passing.

## Dataset layout
```
<root>/data.yaml            names: [...]
<root>/images/<split>/*.jpg  train / val / test
<root>/labels/<split>/*.txt  "cls cx cy w h" (normalised)
```
Tiles are grouped into source frames by stripping `--frame-regex` (default `_<x>_<y>` tile offsets).
An optional `--manifest` CSV (`frame`, optional `tile`, `capture_time`, `lat`, `lon`) enables the
time/GPS checks.

## Does it work? — `dsgate bench`
`dsgate bench <clean dataset> --work <dir>` takes a stratified sample, runs the gate (false alarms?),
plants known faults (frame leak, exact & near duplicates, time leak, corrupt boxes, a class removed
from val, a swapped class pair, dropped labels), runs it again and scores what it caught.

## Agent — `dsgate agent <dataset>`
A small local LLM (default `qwen2.5:3b` via [Ollama](https://ollama.com), ~2.5 GB, free) drives the gate
on an unfamiliar dataset: inspects it, picks checks and options, reads the evidence, and writes a
plain-English explanation and a fix plan (`agent_report.md`). Guardrails live in code, not the prompt:
read-only tools; paths confined to the dataset; the model may make checks **stricter, never looser**
(time gap >= 60 s, radius >= 250 m; a frame regex that groups worse than the default is replaced);
the **verdict is computed from the checks**, never by the model; a **number guard** flags any number in
the explanation that no tool returned; if the model fumbles tool calls, code runs the checks and the
model only explains.

```bash
ollama pull qwen2.5:3b
dsgate agent data/my_set            # exit 1 when BLOCKED, like `check`
```

## Roadmap
1. Footprint-based ground-overlap leakage when camera geometry is known.
3. Point-annotation and COCO inputs.
