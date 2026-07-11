# Phase 6 Walkthrough — Inspector, PropertyScene, Quality, Export

**Status:** Complete  
**Product:** PropertyScene + cleaned PLY + reports (not “PLY is the architecture”)  

---

## What was built

| Module | Role |
|--------|------|
| `optimize/ply_io.py` | Minimal ASCII PLY read/write |
| `optimize/inspector.py` | Needles / floaters / huge / tiny prune |
| `quality/scorer.py` | Geometry-dominated final score |
| `scene/builder.py` | Assemble `PropertyScene` |
| `export/ply.py` | PLY export (prefer cleaned) |
| `export/scene_json.py` | Archive `property_scene.json` |
| Stages `s14`–`s17` | Inspect → quality → assemble → export |
| CLI | `propertyscan export` |

---

## Pipeline (end of product path)

```text
… train
  → inspect (clean artifacts)
  → quality score
  → PropertyScene assembly
  → export (scene.ply + property_scene.json + final_report.json)
```

---

## How to run

```bash
cd revamped_code
pip install -e ".[dev]"

propertyscan export --input tests/fixtures/frames --out ./_out_export \
  --engine mock --train-backend mock

pytest -q tests/unit/test_export_phase6.py
```

Outputs (typical):

```text
_out_export/
  scene.ply
  property_scene.json
  final_report.json
  provenance.json
  metadata.json
  export/
    scene.ply
    cleaned_scene.ply
    property_scene.json
```

---

## Inspector rules (when PLY has attributes)

| Artifact | Rule (defaults) |
|----------|------------------|
| Floater | opacity &lt; 0.05 |
| Needle | aspect &gt; 20 and max scale &gt; 1.5 |
| Huge | max scale &gt; 5 |
| Tiny | max scale &lt; 1e-4 |

xyz-only PLYs (mock): honest **passthrough** with note — no fake pruning.

---

## Intentionally deferred (Phase 7)

- Colab notebook  
- Benchmark harness + history  
- Richer binary PLY attribute parser  
- SOG / streaming exporters  

---

## Why this phase matters (simple English)

### Where Phase 6 sits in the whole product

```text
Capture good frames → lock cameras → train Gaussians
        ↓
Phase 6: clean the mess, score the result, package the product  ← here
        ↓
User gets a tour file + a report they can trust
```

Without Phase 6, training might finish but you still ship **needles, floaters, and a lonely PLY** with no story about quality.

### Why we needed this phase

Gaussian training is not perfect. Even with good poses, some bad splats appear.

Old tools often:

- exported raw PLY as “done”  
- left floaters and spikes for the viewer  
- gave no clear score for “is this house tour good enough?”  

Phase 6 adds the **finishing and accountability** layer:

1. **Inspector** — surgical clean (not re-train)  
2. **Quality score** — geometry first (cameras matter more than pretty PSNR)  
3. **PropertyScene** — the engine owns the scene; PLY is only one export  
4. **Exporters** — plug-in formats without rewriting reconstruction  

### What effect / impact this has on results

| Piece | Impact on what the user sees |
|-------|------------------------------|
| Floater/needle prune | Cleaner walls/space; less “junk in the air” |
| Huge/tiny prune | Smaller files, less weird blobs |
| Geometry-dominated score | A high score means structure is OK, not just a high training loss |
| PropertyScene JSON | Full history for debugging when a tour looks wrong |
| Prefer cleaned PLY | Viewer opens the cleaned version by default |
| Provenance + final_report | You can compare Colab runs later with facts |

### One-sentence summary

**Phase 6 turns a trained splat into a shippable property tour: cleaned, scored, and packaged as a PropertyScene — not just a raw PLY dump.**

---

**Next:** wait for user prompt **start phase 7**.
