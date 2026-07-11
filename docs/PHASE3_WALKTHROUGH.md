# Phase 3 Walkthrough — Geometry Interfaces, Router, Fusion, Health

**Status:** Complete  
**COLMAP:** not included  
**Real MASt3R/DUSt3R weights:** Phase 4  

---

## What was built

The **heart of the platform** as replaceable interfaces:

| Module | Role |
|--------|------|
| `geometry/base.py` | `GeometryProvider` ABC |
| `geometry/providers/mast3r.py` | MASt3R stub (honest failure until Phase 4) |
| `geometry/providers/dust3r.py` | DUSt3R stub (honest failure until Phase 4) |
| `geometry/providers/arkit.py` | Apple ARKit stub |
| `geometry/providers/mock.py` | Synthetic poses for CI / wiring |
| `geometry/router.py` | `mast3r \| dust3r \| auto \| mock` — **no COLMAP** |
| `geometry/depth/*` | `DepthProvider` + Depth Anything V2 stub + mock |
| `geometry/fusion/fuse.py` | Attach depth; poses stay primary |
| `geometry/validation/*` | Geometric checks + pre-train health gate |
| Stages `s07`–`s11` | Route → reconstruct → depth → fuse → health |
| CLI | `propertyscan geometry --engine mock` |

---

## Data flow

```text
FrameSet + SceneDescriptor
    → GeometryRouter (auto ranks MASt3R higher for low-texture)
    → GeometryResult (honest success/failure)
    → DepthResult (soft; may fail)
    → FusedGeometry
    → GeometryValidationReport
    → HealthReport (gate)
```

---

## How to run

```bash
cd revamped_code
pip install -e ".[dev]"

# Full frames + geometry with mock (no GPU weights)
propertyscan geometry --input tests/fixtures/frames --out ./_out_geo --engine mock

pytest -q
```

With real engine names on CPU (expected honest failure until Phase 4):

```bash
propertyscan geometry --input ./my_folder --out ./_out --engine mast3r
# → fails clearly: Phase 4 / CUDA required — does NOT fake poses
```

---

## Routing rules (auto)

- **MASt3R** preferred for low-texture / reflective interiors  
- **DUSt3R** competitive general indoor  
- **ARKit** only if tagged ARKit capture  
- **mock** only when `engine=mock`  
- **Never COLMAP**

---

## Health gate

Aborts training path when:

- registration fraction &lt; configured min  
- health score &lt; `health.min_score`  
- needle probability too high  
- validation issues (missing poses, etc.)

---

## Intentionally deferred (Phase 4)

- Load official ViT-Large MASt3R / DUSt3R checkpoints  
- Real Depth Anything V2 inference  
- Pair graph / VRAM windowing for T4  
- Global alignment metrics from real models  

---

## Why this phase matters (simple English)

### Where Phase 3 sits in the whole product

Think of PropertyScan like building a 3D house tour from a phone video:

1. **Phase 2** picks the best photos from the walk.  
2. **Phase 3 (this phase)** decides *how* we turn those photos into a 3D camera map — and checks whether that map is trustworthy.  
3. **Phase 4** actually runs the big AI models (MASt3R / DUSt3R / Depth Anything).  
4. **Later phases** train the pretty Gaussian splat and clean/export it.

Phase 3 is the **blueprint and quality checkpoint** for 3D geometry. It does not paint the final tour yet — it makes sure the “where was the camera?” system is modular, honest, and safe.

### Why we needed this phase

Your old scans often failed because of **bad camera alignment**. Once cameras are wrong, Gaussian training cannot fix the house — it invents needles, fog, and floaters instead.

Phase 3 fixes the *process*, not yet the model weights:

| Without Phase 3 | With Phase 3 |
|-----------------|--------------|
| Geometry mixed into one big messy stage | Clear stages: pick engine → reconstruct → depth → fuse → health check |
| Easy to fake “success” with empty/wrong poses | Providers must fail **honestly** if models are not ready |
| Hard to swap MASt3R / DUSt3R / future Apple LiDAR | Plug-in providers — one interface for all engines |
| COLMAP-shaped thinking | AI geometry first; COLMAP not in the design |
| Train even when geometry is garbage | **Health gate** can stop bad jobs before wasting GPU time |

### What effect this has on final results

Even though real MASt3R weights come in Phase 4, Phase 3 already changes outcomes:

1. **Better reliability** — We no longer pretend reconstruction worked when it didn’t. Bad geometry is caught early.  
2. **Cleaner future upgrades** — When a better model appears (new MASt3R, VGGT, Apple depth), we add a provider instead of rewriting the whole app.  
3. **Smarter model choice** — “Auto” routing can prefer MASt3R for hard white-wall homes, DUSt3R when it fits, without hardcoding one path forever.  
4. **Depth is first-class** — Walls and flat surfaces get a planned path for depth help (real depth in Phase 4), which should mean fewer floaters later.  
5. **Less wasted money/time** — Health gate is designed so we don’t spend 15 minutes training a splat on a broken camera path.  
6. **Mock path for testing** — We can test the full pipeline on a laptop without a GPU, so the plumbing stays solid before Colab T4 runs.

### One-sentence summary

**Phase 3 builds the “geometry engine room” of PropertyScan: how we choose AI reconstruction, how we combine it with depth, and how we refuse to train a 3D tour when the camera map is not good enough.**

---

**Next:** wait for user prompt **start phase 4**.
