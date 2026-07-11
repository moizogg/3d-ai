# Phase 4 Walkthrough — Real MASt3R / DUSt3R / Depth Anything V2

**Status:** Complete (code paths live; needs GPU + package install to run neural weights)  
**COLMAP:** still not used  
**Default models:** official NAVER ViT-Large + Depth Anything V2  

---

## What was built

| Module | Role |
|--------|------|
| `geometry/foundation_infer.py` | Shared dense recon: load images → pairs → inference → global align → poses/PLY |
| `geometry/runtime.py` | CUDA device, VRAM peak, cache clear, pair-graph VRAM safety |
| `geometry/deps.py` | Honest dependency probes (no COLMAP fallback text) |
| `providers/mast3r.py` | **Real** MASt3R when packages+CUDA present |
| `providers/dust3r.py` | **Real** DUSt3R when packages+CUDA present |
| `depth/anything_v2.py` | **Real** Depth Anything V2 via HuggingFace pipeline |
| CLI `doctor` | Shows dust3r/mast3r/transformers/CUDA readiness |

---

## Default model IDs (quality-first)

```text
MASt3R: naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
DUSt3R: naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt
Depth:  depth-anything/Depth-Anything-V2-Small-hf   # colab_t4
        depth-anything/Depth-Anything-V2-Base-hf    # quality_gpu
```

- **batch_size = 1** always for T4 safety  
- **pair_graph**: `swin-5` default; `complete` auto-downgrades to swin-5 if N > 40  
- On failure: **error message only** — never COLMAP, never fake 100% registration  

---

## Install (Colab T4 / GPU machine)

Unit tests do **not** need this. Real scans do.

```bash
# 1) PyTorch CUDA (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2) Depth Anything path
pip install "transformers>=4.38" einops

# 3) DUSt3R (required by both engines)
git clone --recursive https://github.com/naver/dust3r
cd dust3r && pip install -e . && cd ..

# 4) MASt3R
git clone --recursive https://github.com/naver/mast3r
cd mast3r && pip install -e . && cd ..

# 5) PropertyScan
cd revamped_code && pip install -e ".[dev,geometry]"
propertyscan doctor --profile colab_t4
```

First run downloads multi‑GB weights from HuggingFace / NAVER.

---

## How to run

```bash
# Plumbing without neural stack
propertyscan geometry --input ./frames --out ./_out --engine mock

# Real MASt3R on GPU (after install)
propertyscan geometry --input ./walk_or_folder --out ./_out --engine mast3r --profile colab_t4

# Auto routing (MASt3R preferred for hard interiors)
propertyscan geometry --input ./walk_or_folder --out ./_out --engine auto --profile colab_t4
```

Artifacts under `work/.../artifacts/geometry/<provider>/`:

- `transforms.json`  
- `mast3r_dense.ply` / `dust3r_dense.ply` (when save succeeds)  
- `depth/*.png` (16-bit relative depth)  
- `geometry_report.json` at run output root  

---

## Tests

```bash
pytest -q                 # includes mocked success path; no GPU required
pytest -q -m gpu          # optional real GPU smoke (skips if no CUDA/packages)
```

---

## Intentionally deferred (Phase 5+)

- 3DGS dataset packaging + splatfacto/gsplat training  
- Adaptive training from health report  
- Inspector / PropertyScene export polish  

---

## Why this phase matters (simple English)

### Where Phase 4 sits in the whole product

```text
Phase 2: pick good photos
Phase 3: define the “geometry engine room” (interfaces, health gate)
Phase 4: actually RUN the AI that finds camera positions  ← you are here
Phase 5+: train the pretty 3D Gaussian tour from those cameras
```

This is the phase that most directly decides whether your **3D tour aligns** or collapses into needles and fog.

### Why we needed this phase

Phases 1–3 built pipes and rules. Without Phase 4, the system still could not solve real geometry from a walkthrough.

Your old failures were often **alignment** failures. Phase 4 plugs in the modern answer:

- **MASt3R / DUSt3R** (large neural models) estimate cameras and dense 3D points **without COLMAP / SIFT**  
- **Depth Anything V2** adds dense depth so walls and flat surfaces have geometric guidance later  

We also refuse the old trap: if the model or GPU is missing, we **stop with a clear error** instead of pretending success or falling back to COLMAP.

### What effect / impact this has on results

| Capability | Impact on the final tour |
|------------|---------------------------|
| Real ViT-Large MASt3R/DUSt3R | Stronger camera poses on indoor homes → **less warp, fewer needles** |
| No COLMAP fallback | One clear geometry path; failures are visible and fixable |
| Depth Anything maps | Better wall/floor lock when training starts → **fewer floaters / fog** |
| T4 pair windows (`swin-5`) | Real scans can run on free Colab without always OOM’ing |
| Honest health gate on real metrics | Bad alignments don’t burn 15 minutes of splat training |
| Official model IDs in config | You get the **large quality models**, not accidental tiny stubs |

**Honest limit:** On a machine without GPU packages installed, Phase 4 code **exists** but will fail clearly until you install torch + dust3r/mast3r on Colab T4 (or a 4090 later). That is intentional.

### One-sentence summary

**Phase 4 is where PropertyScan actually runs the big AI geometry models so the cameras of your walkthrough lock into real 3D — the step that most decides whether the final tour looks solid or broken.**

---

**Next:** wait for user prompt **start phase 5**.
