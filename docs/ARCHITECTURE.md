# PropertyScan AI — Revamped Architecture (Phase 1)

## Purpose

This document describes the **clean-room** architecture under `revamped_code/`.

It is not a mirror of the legacy `worker/` package. Legacy code is reference only.

## Core philosophy

1. **Engine owns the scene** — `PropertyScene` is the product; PLY is an export.
2. **Confidence over poses** — geometry providers emit confidence, not fake 100% registration.
3. **AI-first geometry** — MASt3R / DUSt3R / Depth Anything V2; **no COLMAP** in this rewrite.
4. **Honest failure** — bad alignment aborts before Gaussian training.
5. **Replaceable modules** — providers and stages are interface-driven.

## Package map (Phase 1)

```text
propertyscan/
  core/       config, context, stage protocol, device, logging, exceptions
  domain/     FrameSet, GeometryResult, DepthResult, PropertyScene, provenance, …
  pipeline/   (orchestrator + stages — later phases)
  cli.py      version / doctor / config
```

## Configuration

- Base: `configs/default.yaml`
- Profiles: `colab_t4.yaml` (current testing), `quality_gpu.yaml` (4090+ later)
- Quality overlays: `configs/quality/{draft,standard,high}.yaml`
- Env prefix: `PSCAN_` (e.g. `PSCAN_GEOMETRY_ENGINE=mast3r`)

Default geometry model IDs (official large neural weights):

- MASt3R: `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`
- DUSt3R: `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`

## Stage contract

```text
Stage.validate(ctx) → Stage.execute(ctx) → StageResult
RunContext records stage history + provenance
```

## Domain heart (geometry)

| Type | Role |
|------|------|
| `GeometryResult` | Provider output (poses, points, conf, metrics) |
| `PoseGraph` / `CameraPose` | Cameras without COLMAP assumptions |
| `PointCloud` | Dense/sparse points (inline or path) |
| `ConfidenceMap` | Multi-level confidence |
| `SceneDescriptor` | Routing features for Auto mode |
| `DepthResult` | First-class depth product |
| `HealthReport` | Pre-train gate |
| `PropertyScene` | Canonical assembled product |

## GPU strategy

| Now | Later |
|-----|--------|
| Colab **Tesla T4** profile | **RTX 4090+** `quality_gpu` profile |
| Same ViT-Large weights | Same code; denser pairs / stronger depth |

T4 reduces pair windows / keyframe count — **not** model quality class.

## Out of scope (this rewrite)

- COLMAP / SIFT / `ns-process-data`
- Bundle Adjustment as primary solver
- Silent success with stub geometry

## Phase roadmap

See [PHASES.md](PHASES.md) and root [REVAMPED_ARCHITECTURE_PLAN.md](../../REVAMPED_ARCHITECTURE_PLAN.md).
