# Phase 1 Walkthrough — Foundations & Domain Skeleton

**Status:** Complete  
**Tests:** 20 passed  
**Date:** 2026-07-11

---

## What was built

A clean-room, installable engine skeleton under `revamped_code/` with:

- Typed configuration (YAML profiles + Pydantic)
- Stage protocol + `RunContext` + provenance hooks
- Full domain model surface for later geometry/depth/training
- CLI: `version`, `doctor`, `config`
- Unit tests and architecture docs

**No model inference yet** (by design). Real MASt3R/DUSt3R loads land in Phase 4.

---

## Folder / file map

```text
revamped_code/
├── pyproject.toml
├── README.md
├── configs/
│   ├── default.yaml
│   ├── colab_t4.yaml          ← your primary test profile
│   ├── quality_gpu.yaml       ← 4090+ later (config only)
│   └── quality/{draft,standard,high}.yaml
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DECISIONS.md
│   ├── PHASES.md
│   └── PHASE1_WALKTHROUGH.md  ← this file
├── propertyscan/
│   ├── __init__.py            # version 0.1.0
│   ├── cli.py
│   ├── core/
│   │   ├── config.py          # load_config, EngineConfig
│   │   ├── context.py         # RunContext
│   │   ├── stage.py           # Stage ABC, StageResult
│   │   ├── device.py          # CUDA/CPU discovery
│   │   ├── exceptions.py
│   │   └── logging.py
│   ├── domain/
│   │   ├── frames.py          # FrameMetadata, FrameSet
│   │   ├── capture.py         # CaptureManifest, CaptureKind
│   │   ├── geometry.py        # GeometryResult, PoseGraph, PointCloud, …
│   │   ├── depth.py           # DepthResult (first-class)
│   │   ├── quality.py         # HealthReport, InspectionReport, QualityReport
│   │   ├── gaussian.py
│   │   ├── scene.py           # PropertyScene
│   │   └── provenance.py
│   └── pipeline/              # empty placeholder
└── tests/unit/                # 20 tests
```

---

## Interfaces introduced (Phase 1 contracts)

| Interface / type | Role |
|------------------|------|
| `EngineConfig` | Validated settings for entire run |
| `Stage` / `StageResult` | Single-responsibility pipeline units |
| `RunContext` | Job paths, typed state, stage history, provenance |
| `GeometryResult` / `PoseGraph` / `ConfidenceMap` | Provider-agnostic geometry heart |
| `DepthResult` | First-class depth product |
| `HealthReport` | Pre-train gate shape |
| `PropertyScene` | Canonical product (not PLY) |
| `ProvenanceRecord` | Research / debugging trail |

---

## Config highlights (alignment quality path)

Default / Colab T4 profiles already pin **official large neural checkpoints**:

```text
mast3r_model: naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
dust3r_model: naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt
allow_cpu_geometry: false
```

- **T4 profile** = same large models, fewer keyframes, Depth Small, swin pair graph  
- **quality_gpu** = same models, more keyframes, Depth Base, denser pairing  
- **No COLMAP** anywhere in config  

---

## How to run

```bash
cd revamped_code
pip install -e ".[dev]"

propertyscan version
propertyscan doctor --profile colab_t4
propertyscan config --profile colab_t4

pytest -q
```

Verified on this machine:

- `propertyscan 0.1.0`
- doctor reports `colab_t4` + ViT-Large model IDs
- **20 passed** unit tests

---

## Data flow (Phase 1 only)

```text
CLI / tests
  → load_config(profile)
  → EngineConfig
  → RunContext (work dirs, provenance)
  → Stage.run()  [demo stages in tests only]
  → StageResult + provenance JSON
```

No capture → geometry → train path yet.

---

## Intentional limitations (deferred)

| Deferred | Phase |
|----------|------:|
| Video / image capture adapters | 2 |
| Frame quality, dedup, keyframes | 2 |
| GeometryProvider ABC + router + mocks | 3 |
| Real MASt3R / DUSt3R / Depth Anything on GPU | 4 |
| 3DGS dataset + training | 5 |
| Inspector + export | 6 |
| Colab notebook + benchmarks | 7 |

---

## Alignment / failure stance (carried forward)

Your old tours failed on **alignment**. Phase 1 encodes the anti-patterns we will not repeat:

1. Official large model IDs are already the default (not shorthand stubs).  
2. `allow_cpu_geometry: false` — no silent fake success on CPU.  
3. `GeometryResult` / `HealthReport` types require honest metrics later (no assumed 100% reg).  
4. Profiles prepare T4 testing now and 4090 upgrade later without rewrite.

---

## Why this phase matters (simple English)

### Where Phase 1 sits in the whole product

Phase 1 is the **foundation of the house**, not the furniture:

- It creates the project structure, settings, logging, and shared data types.  
- Later phases (frames → geometry → training → export) all plug into this foundation.  
- Without it, every feature would reinvent config, errors, and file layouts — and the code would become spaghetti again.

### Why we needed this phase

The old worker grew around COLMAP and mixed many jobs in few files. That made it hard to:

- swap AI models safely  
- test pieces without a full GPU job  
- know *why* a scan failed  

Phase 1 starts a **clean, modular engine** designed for MASt3R / DUSt3R / Depth Anything — not another COLMAP wrapper.

### What effect this has on final results

Phase 1 does not improve a single 3D tour by itself (no photos processed yet). Its impact is **indirect but huge**:

1. **Stability** — Same config and stage pattern for every run (T4 today, 4090 later).  
2. **Honesty** — Types like `GeometryResult` and health reports are built so we cannot “fake 100% success” later.  
3. **Speed of improvement** — New research models can plug in without rewriting the whole pipeline.  
4. **Debugging** — Provenance and metadata mean we can answer: what profile, what device, what failed.

### One-sentence summary

**Phase 1 builds the rules and language of the engine so every later phase can produce reliable real-estate 3D tours without chaos.**

---

## Readiness for Phase 2

Phase 2 can start when you say so. It will implement:

- Capture adapters (video, image folder, sequence)
- Frame intelligence (quality, dedup, keyframes, scene classify)
- CLI: `propertyscan frames --input … --out …`
- Artifact tree under `work/artifacts/frames/`

**Waiting for your prompt:** e.g. `start phase 2`
