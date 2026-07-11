# PropertyScan AI — Revamped Geometry Engine

Clean-room **AI-first Geometry Engine** for real-estate 3D reconstruction.

| | |
|---|---|
| **Location** | `revamped_code/` |
| **Primary stack** | MASt3R · DUSt3R · Depth Anything V2 |
| **Not used** | COLMAP, SIFT, `ns-process-data` |
| **Phases** | **1–7 complete** — see [`docs/PHASES.md`](docs/PHASES.md) |

## Install

```bash
cd revamped_code
pip install -e ".[dev]"
propertyscan version
propertyscan doctor
pytest -q
```

## Full product path (no GPU)

```bash
propertyscan export --input tests/fixtures/frames --out ./_out \
  --engine mock --train-backend mock
```

Produces: `scene.ply`, `property_scene.json`, `final_report.json`, `provenance.json`.

## Commands

| Command | Stage |
|---------|--------|
| `frames` | Capture + frame intelligence |
| `geometry` | Through pre-train health gate |
| `train` | Dataset + Gaussian train |
| `export` | Inspect + PropertyScene + PLY |
| `benchmark` | Multi-scene + `history.jsonl` |
| `doctor` / `config` | Environment + settings |

## Colab T4 — real dataset test

1. **Full step-by-step (MediaFire + GitHub + every cell):**  
   [`docs/COLAB_REAL_TEST_GUIDE.md`](docs/COLAB_REAL_TEST_GUIDE.md)  
2. Notebook: [`notebooks/colab_real_test.ipynb`](notebooks/colab_real_test.ipynb)  
3. Operator overview: [`docs/OPERATOR_GUIDE.md`](docs/OPERATOR_GUIDE.md)  
4. MASt3R install detail: [`docs/PHASE4_WALKTHROUGH.md`](docs/PHASE4_WALKTHROUGH.md)  

**Colab working directory:** `/content/3d-ai` (GitHub repo root — package files live there, not under `revamped_code/`).

Live progress: every stage prints `[PROGRESS] …` **every 10 seconds** so you know it is not stuck.  

```bash
propertyscan export -i walk.mp4 -o ./_out --profile colab_t4 \
  --engine mast3r --train-backend splatfacto
```

## Docs

- [Operator guide](docs/OPERATOR_GUIDE.md)  
- [Architecture](docs/ARCHITECTURE.md)  
- [Phases](docs/PHASES.md)  
- [Reliable frame validation](docs/VALIDATION_RELIABLE_V2.md)  
- Phase walkthroughs 1–7 (each ends with **why it matters** in plain English)  
- Root plan: [`../REVAMPED_ARCHITECTURE_PLAN.md`](../REVAMPED_ARCHITECTURE_PLAN.md)  

## Philosophy

The engine owns a **PropertyScene**. Exporters translate it. Geometry providers are interchangeable. Reliable frame validation + health gates stop bad alignment from becoming a bad Gaussian tour.
