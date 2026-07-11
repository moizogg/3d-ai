# Phase 5 Walkthrough — Dataset Builder + Gaussian Training

**Status:** Complete  
**Real splatfacto:** requires Nerfstudio `ns-train` on Colab (test later)  
**CI path:** `--engine mock --train-backend mock`  

---

## What was built

| Module | Role |
|--------|------|
| `domain/dataset.py` | `TrainingDataset` typed package |
| `dataset/builder.py` | Assemble images + transforms + depth + init PLY |
| `training/base.py` | `TrainerBackend` / `TrainResult` |
| `training/presets.py` | Quality + dense-init iteration policy |
| `training/mock.py` | CI trainer (tiny placeholder PLY) |
| `training/splatfacto.py` | Honest `ns-train splatfacto` adapter |
| `training/factory.py` | `mock` \| `splatfacto` |
| Stages `s12`–`s13` | Build dataset → train |
| CLI | `propertyscan train` |

---

## Dataset layout

```text
artifacts/dataset/
  images/
  depths/              # if depth succeeded
  images_N/            # optional Nerfstudio downscale
  transforms.json      # poses + depth_file_path + ply_file_path
  sparse_pc.ply        # optional foundation init cloud
```

Training **never invents poses**. It only consumes geometry that already passed the health gate.

---

## How to run

```bash
cd revamped_code
pip install -e ".[dev]"

# Full mock path (no GPU, no ns-train)
propertyscan train --input tests/fixtures/frames --out ./_out_train \
  --engine mock --train-backend mock

pytest -q tests/unit/test_train_phase5.py
```

On Colab (after Phases complete + nerfstudio installed):

```bash
propertyscan train --input ./walk.mp4 --out ./_out \
  --engine mast3r --train-backend splatfacto --profile colab_t4 --quality standard
```

---

## Config knobs

```yaml
training:
  backend: splatfacto   # or mock
  quality: standard
  downscale_factor: 4
  reduce_iters_for_dense_geometry: true
  dense_geometry_max_iters: 8000
  require_health_pass: true
```

---

## Intentionally deferred (Phase 6+)

- Post-train inspector (needles / floaters)  
- PropertyScene assembly + multi-format export  
- Colab notebook polish  

---

## Why this phase matters (simple English)

### Where Phase 5 sits in the whole product

```text
Phase 2: pick good photos
Phase 3–4: figure out where the camera was (geometry + depth)
Phase 5: package that into a clean training set and grow the 3D Gaussian tour  ← here
Phase 6: clean artifacts and export the final file
```

This is the first phase that produces an actual **3D Gaussian scene** (or a mock stand-in for testing).

### Why we needed this phase

Good geometry alone is not enough. Training needs a **neat folder**:

- every camera linked to the right image  
- optional depth maps for walls  
- optional dense point cloud so Gaussians start on real surfaces  

Without that, splat training guesses, wastes GPU time, and recreates the “needles / fog” problems you already saw.

Phase 5 also **refuses to train** if the health gate failed — so a broken alignment does not become a 15‑minute fake tour.

### What effect / impact this has on results

| Design choice | Impact on the tour |
|---------------|--------------------|
| Dataset from validated poses only | Training refines appearance, doesn’t invent camera paths |
| Depth linked in transforms | Walls stay flatter; fewer floaters in empty space |
| Init cloud from MASt3R/DUSt3R | Faster convergence, fewer random “fog” Gaussians |
| Fewer iters when dense init exists | Shorter Colab runs without throwing quality away |
| Downscale for T4 | Fits free GPU RAM; still trains a usable scene |
| mock backend for CI | Pipeline stays tested before Colab day |
| Honest ns-train failure | Clear install message instead of silent junk PLY |

### One-sentence summary

**Phase 5 turns a trusted camera map into a real (or test) Gaussian 3D tour — without training on garbage geometry.**

---

**Next:** wait for user prompt **start phase 6**.
