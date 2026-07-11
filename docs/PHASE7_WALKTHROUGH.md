# Phase 7 Walkthrough — Research Harness, Colab, Polish

**Status:** Complete  
**Closes** the planned phase ladder for the AI-first rewrite  

---

## What was built

| Module / asset | Role |
|----------------|------|
| `research/metrics.py` | Flatten run metrics for history |
| `research/experiment.py` | ExperimentRecord + JSONL registry |
| `research/artifacts.py` | Standard research tree (Frames, Geometry, …) |
| `research/benchmark.py` | Multi-scene runner |
| CLI `propertyscan benchmark` | Operator entrypoint |
| `notebooks/colab_pipeline.ipynb` | Colab T4 notebook skeleton |
| `docs/OPERATOR_GUIDE.md` | End-to-end operator manual |
| README | Updated full-command surface |

---

## How to run

```bash
cd revamped_code
pip install -e ".[dev]"

# Single-scene product path
propertyscan export -i tests/fixtures/frames -o ./_out \
  --engine mock --train-backend mock

# Multi-scene benchmark (fixture root = one scene, or subfolders)
propertyscan benchmark --data tests/fixtures --out ./_bench \
  --engine mock --train-backend mock

pytest -q
```

Research tree example:

```text
_bench/research/<scene_id>/
  Frames/
  Geometry/
  Depth/
  Transforms/
  Training_Logs/
  PLY/
  Quality_Report/
  Metadata/
  metrics.json
_bench/registry/history.jsonl
```

---

## Colab

Open [`notebooks/colab_pipeline.ipynb`](../notebooks/colab_pipeline.ipynb):

1. GPU runtime  
2. Install package  
3. Mock export smoke  
4. Optional real MASt3R install (Phase 4 doc)  
5. Real export when ready  

---

## Intentionally light (by design)

- No full PSNR/SSIM suite yet (needs holdout renders)  
- No SOG compressor (viewer-side later)  
- Binary PLY surgical prune still limited (ASCII attrs preferred)  

These can grow **without** rewriting the pipeline — registry + metrics are already in place.

---

## Why this phase matters (simple English)

### Where Phase 7 sits in the whole product

Phases 1–6 built the **engine** that makes a tour.

Phase 7 builds the **lab and cockpit** around that engine:

- How do we run many houses the same way?  
- How do we remember what we tried?  
- How do we run this on free Colab without tribal knowledge?  

### Why we needed this phase

A single successful mock run is not enough for a research-grade real-estate engine.

You need:

1. **Reproducible history** — which model, profile, score, commit  
2. **Comparable folders** — every scene laid out the same for diffs  
3. **Operator clarity** — Colab notebook + guide so you don’t reinvent commands  
4. **Benchmark loop** — prove MASt3R helps *before* calling it “default forever”  

Without Phase 7, quality improvements become anecdotes. With it, they become **data**.

### What effect / impact this has on results

| Piece | Impact |
|-------|--------|
| `history.jsonl` | See if a change helped or hurt across scenes |
| Research layout | Diff PLY / health / quality side by side |
| Metrics dict | Same fields every run → fair comparison |
| Colab notebook | Faster path to real T4 experiments |
| Operator guide | Fewer wrong flags / wasted GPU hours |
| Benchmark CLI | One command to smoke-test the whole product path |

Phase 7 does not by itself raise PSNR. It makes **every future raise** measurable — so you don’t ship another silent alignment failure.

### One-sentence summary

**Phase 7 turns PropertyScan from “a pipeline that runs” into “a research platform you can measure, repeat, and run on Colab with confidence.”**

---

## Phase ladder complete

| Phase | Outcome |
|------:|---------|
| 1 | Foundation + types |
| 2 | Reliable frames |
| 3 | Geometry interfaces + health |
| 4 | Real MASt3R/DUSt3R/Depth paths |
| 5 | Dataset + train |
| 6 | Inspect + PropertyScene + export |
| 7 | Research + Colab + polish polish |

**Next (product):** run full Colab validation with real weights when ready.  
**Next (code):** optional improvements only as measured experiments via the registry.
