# PropertyScan AI — Operator Guide

End-to-end guide for running the **revamped** engine (`revamped_code/`).

## What this product does

Turns a phone walkthrough (video or photos) into:

1. Selected keyframes  
2. AI camera geometry (MASt3R / DUSt3R)  
3. Optional monocular depth  
4. Gaussian training  
5. Cleaned PLY + PropertyScene + quality report  

**COLMAP is not used.**

## Quick start (no GPU)

```bash
cd revamped_code
pip install -e ".[dev]"
propertyscan doctor
propertyscan export --input tests/fixtures/frames --out ./_out \
  --engine mock --train-backend mock
```

Outputs: `scene.ply`, `property_scene.json`, `final_report.json`, `provenance.json`.

## Commands

| Command | Purpose |
|---------|---------|
| `propertyscan version` | Package version |
| `propertyscan doctor` | Deps + CUDA + foundation probes |
| `propertyscan config --profile colab_t4` | Dump merged config |
| `propertyscan frames -i … -o …` | Capture + frame intelligence only |
| `propertyscan geometry -i … -o …` | Through health gate |
| `propertyscan train -i … -o …` | Through Gaussian train |
| `propertyscan export -i … -o …` | Full product path |
| `propertyscan benchmark --data … -o …` | Multi-scene + history.jsonl |

## Profiles

| Profile | When |
|---------|------|
| `default` | General |
| `colab_t4` | Free Colab T4 (tighter keyframes, depth small, swin pairs) |
| `quality_gpu` | 4090+ later (more keyframes, depth base) |

```bash
propertyscan export -i walk.mp4 -o ./_out --profile colab_t4 \
  --engine mast3r --train-backend splatfacto
```

## Colab day checklist (real MediaFire dataset)

**Full guide with every cell:** [COLAB_REAL_TEST_GUIDE.md](COLAB_REAL_TEST_GUIDE.md)  
**Notebook:** `notebooks/colab_real_test.ipynb`  
**Dataset ZIP:** MediaFire archive (see guide) · **Code:** https://github.com/moizogg/3d-ai  

1. Runtime → GPU (T4)  
2. Clone/upload repo → `cd revamped_code`  
3. Download/unzip dataset (wget or manual upload if MediaFire blocks)  
4. Mock export first (progress every 10s)  
5. Install dust3r/mast3r (Phase 4)  
6. Real `mast3r` then full export  
7. Splatfacto only after Nerfstudio is installed  

## Research / benchmarks

```bash
# data/bench_scenes/scene_a/*.jpg
# data/bench_scenes/scene_b/*.jpg
propertyscan benchmark --data data/bench_scenes --out ./_bench \
  --engine mock --train-backend mock
```

Writes:

- `runs/<scene>/…` full pipeline outs  
- `research/<scene>/` Frames, Geometry, Depth, PLY, Quality_Report, Metadata  
- `registry/history.jsonl` experiment log  

## Reading quality

- `final_report.json` → overall score + inspection  
- Prefer **camera/geometry** over photometric vanity  
- Health gate may stop early: fix capture/geometry, don’t train junk  

## Docs map

| Doc | Content |
|-----|---------|
| [PHASES.md](PHASES.md) | Phase status |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design |
| [PHASE4_WALKTHROUGH.md](PHASE4_WALKTHROUGH.md) | Real model install |
| [VALIDATION_RELIABLE_V2.md](VALIDATION_RELIABLE_V2.md) | Frame validation policy |
| Phase walkthroughs 1–7 | What shipped + why it matters |
