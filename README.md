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

## Roadmap
1. A local agent layer (Ollama, small open model, tool calling) that inspects an unknown dataset,
   picks the checks, explains failures in plain English and proposes fixes into a new folder — never
   overriding a FAIL and never editing data in place.
2. Footprint-based ground-overlap leakage when camera geometry is known.
3. Point-annotation and COCO inputs.
