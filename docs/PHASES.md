# Phase Tracker — Revamped Engine

| Phase | Name | Status |
|------:|------|--------|
| 1 | Foundations & Domain Skeleton | **Complete** |
| 2 | Capture + Frame Intelligence (+ reliable validation) | **Complete** |
| 3 | Geometry / Depth Interfaces + Router | **Complete** |
| 4 | Real MASt3R / DUSt3R / Depth Anything | **Complete** |
| 5 | Dataset + Training | **Complete** |
| 6 | Inspector + PropertyScene + Export | **Complete** |
| 7 | Research Harness + Colab + Polish | **Complete** |

## Walkthrough standard

Every phase ends with **Why this phase matters (simple English)**.

## Phase 7 — Complete

Experiment registry, benchmark runner, research artifact layout, Colab notebook, operator guide.  
See [PHASE7_WALKTHROUGH.md](PHASE7_WALKTHROUGH.md) and [OPERATOR_GUIDE.md](OPERATOR_GUIDE.md).

```bash
propertyscan benchmark --data tests/fixtures --out ./_bench \
  --engine mock --train-backend mock
pytest -q
```

## Product next step

Colab T4 validation with real foundation weights (install per Phase 4 doc).
