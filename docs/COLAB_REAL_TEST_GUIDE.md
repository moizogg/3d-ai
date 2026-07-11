# PropertyScan AI — Real Dataset Test on Google Colab (Full Guide)

This guide walks you through a **real end-to-end test** using:

| Item | Link / value |
|------|----------------|
| **Dataset (MediaFire ZIP)** | [Download archive.zip](https://download853.mediafire.com/rtmlk9ypemegyIXh1yTx9nFxnjYYGn5l5Y7jgmzy4HCb7cekTEY__64r_wSqKfc3Cns4Iohj56JzG_hKv6SpxVLLkM8y_ucaQCHgrNLuS14oFWQmjlp4IuwbXMKOG3aP7MOpotpk3DGZor2n-Kk5rH-m3LedwvDJXId-PhLfrTqM_Q/zknvd5nfvsygnsr/archive.zip) |
| **Code (GitHub)** | https://github.com/moizogg/3d-ai |
| **Revamped engine path** | `revamped_code/` inside the repo |
| **GPU** | Colab Free **T4** (recommended first) |

You will see **live progress every ~10 seconds** (`[PROGRESS] … still running…`) so you know the job is not stuck.

---

## What you will get

```text
_out_real/
  scene.ply                 # 3D Gaussian point cloud (cleaned when possible)
  property_scene.json       # full product metadata
  final_report.json         # quality + inspection
  provenance.json           # what ran (models, stages, device)
  work/.../artifacts/       # frames, geometry, depth, dataset, training
```

---

## Before you start (checklist)

1. Google account + [Google Colab](https://colab.research.google.com/)  
2. **Runtime → Change runtime type → GPU (T4)**  
3. Repo available on GitHub: `moizogg/3d-ai` (or upload a ZIP of your local clone)  
4. Dataset ZIP on MediaFire (link above) — if the direct link expires, re-copy “Download” from MediaFire UI  

---

# Part A — Get the code on Colab

## Cell A1 — GPU check

```python
!nvidia-smi
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
```

**Expect:** a T4 (or similar) listed. If CUDA is `False`, fix runtime before continuing.

---

## Cell A2 — Clone GitHub repo

```python
import os
from pathlib import Path

%cd /content
REPO = "https://github.com/moizogg/3d-ai.git"
DIR = Path("/content/3d-ai")

if not DIR.exists():
    !git clone --depth 1 {REPO}
else:
    print("Repo already present:", DIR)

%cd /content/3d-ai
!ls -la
```

### If the repo is private

```python
# Option 1: upload a zip of the project via Colab Files sidebar, then:
# !unzip -q /content/3d-ai.zip -d /content
# %cd /content/3d-ai

# Option 2: use a GitHub personal access token (do not share tokens)
# !git clone https://<TOKEN>@github.com/moizogg/3d-ai.git
```

### If `revamped_code` is only on your PC (not pushed yet)

Upload the `revamped_code` folder as a ZIP:

1. Zip `revamped_code` on your PC  
2. Colab left sidebar → **Files** → upload `revamped_code.zip`  
3. Run:

```python
%cd /content
!unzip -q /content/revamped_code.zip -d /content/3d-ai
%cd /content/3d-ai/revamped_code
!ls
```

**Working directory for all later cells must be:**

```text
/content/3d-ai/revamped_code
```

```python
%cd /content/3d-ai/revamped_code
!pwd
!ls propertyscan configs docs
```

---

## Cell A3 — Install PropertyScan package

```python
%cd /content/3d-ai/revamped_code
!pip -q install -e ".[dev]"
!propertyscan version
!propertyscan doctor --profile colab_t4
```

**Expect:** `propertyscan 0.1.0` and doctor listing profile `colab_t4`.

---

# Part B — Download & prepare the MediaFire dataset

## Cell B1 — Download the ZIP

```python
from pathlib import Path

DATA = Path("/content/data")
DATA.mkdir(parents=True, exist_ok=True)
ZIP = DATA / "archive.zip"

# MediaFire direct link (if this 403s, use B1-alt below)
URL = "https://download853.mediafire.com/rtmlk9ypemegyIXh1yTx9nFxnjYYGn5l5Y7jgmzy4HCb7cekTEY__64r_wSqKfc3Cns4Iohj56JzG_hKv6SpxVLLkM8y_ucaQCHgrNLuS14oFWQmjlp4IuwbXMKOG3aP7MOpotpk3DGZor2n-Kk5rH-m3LedwvDJXId-PhLfrTqM_Q/zknvd5nfvsygnsr/archive.zip"

!wget -c --content-disposition -O "{ZIP}" "{URL}" || true
print("size_mb", ZIP.stat().st_size / 1e6 if ZIP.exists() else "MISSING")
```

### Cell B1-alt — If wget fails (MediaFire often blocks bots)

**Manual upload (most reliable):**

1. On your PC, open the MediaFire link and download `archive.zip`  
2. Colab → **Files** → upload `archive.zip` to `/content/data/`  
3. Confirm:

```python
from pathlib import Path
ZIP = Path("/content/data/archive.zip")
assert ZIP.exists() and ZIP.stat().st_size > 1_000_000, "Upload archive.zip first"
print("OK", ZIP, "MB", ZIP.stat().st_size / 1e6)
```

**Or upload to Google Drive** and mount:

```python
from google.colab import drive
drive.mount('/content/drive')
# then copy your file, e.g.:
# !cp "/content/drive/MyDrive/archive.zip" /content/data/archive.zip
```

---

## Cell B2 — Unzip and find images / video

```python
from pathlib import Path
import zipfile

DATA = Path("/content/data")
ZIP = DATA / "archive.zip"
EXTRACT = DATA / "kaggle_scene"
EXTRACT.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ZIP, "r") as zf:
    zf.extractall(EXTRACT)

print("Top-level:")
for p in sorted(EXTRACT.iterdir())[:30]:
    print(" ", p.name, "DIR" if p.is_dir() else f"file {p.stat().st_size}")

# Recursively find a good INPUT (video or image folder)
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

def find_input(root: Path):
    videos = list(root.rglob("*"))
    vids = [p for p in videos if p.is_file() and p.suffix.lower() in VIDEO_EXT]
    if vids:
        return sorted(vids, key=lambda p: -p.stat().st_size)[0]
    # folder with most images
    best, best_n = None, 0
    for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        n = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT) if d.exists() else 0
        # non-recursive count only
        try:
            n = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)
        except Exception:
            n = 0
        if n > best_n:
            best, best_n = d, n
    return best

INPUT = find_input(EXTRACT)
print("INPUT =>", INPUT)
assert INPUT is not None, "Could not find video or image folder"
```

**Tip:** If you already know the path (example):

```python
INPUT = Path("/content/data/kaggle_scene/images")  # adjust after listing
```

---

# Part C — Two test modes

| Mode | When | Command flags |
|------|------|----------------|
| **Smoke (mock)** | Prove pipeline + progress works | `--engine mock --train-backend mock` |
| **Real (AI geometry)** | Real MASt3R + depth (+ splatfacto if installed) | `--engine mast3r --train-backend mock` first, then splatfacto |

Always run **mock once** before real models (faster failure if paths wrong).

---

## Cell C1 — SMOKE: full export with mock (5–15 min depending on frames)

```python
%cd /content/3d-ai/revamped_code
from pathlib import Path

OUT = Path("/content/out_mock")
OUT.mkdir(exist_ok=True)

# Progress lines every 10s look like:
# [PROGRESS] stage:frame_quality | still running… elapsed=30s

!propertyscan export \
  --input "{INPUT}" \
  --out "{OUT}" \
  --profile colab_t4 \
  --engine mock \
  --train-backend mock

!ls -la "{OUT}"
!ls -la "{OUT}/export" || true
print("scene.ply exists:", (OUT / "scene.ply").exists())
print("final_report:", (OUT / "final_report.json").exists())
```

**Healthy signs while running:**

```text
[PROGRESS] stage:validate_capture | started | elapsed=0s
[PROGRESS] stage:frame_quality | still running… status=measure 40/200 | tick=2 | elapsed=20s
[PROGRESS] stage:reconstruct_geometry | still running… elapsed=10s
...
[PROGRESS] stage:export | finished OK | elapsed=1.2s
✅ Export pipeline complete.
```

If you see **no** `[PROGRESS]` for >30s, the cell may be stuck on download/install — scroll up for errors.

---

## Cell C2 — Install real foundation stack (MASt3R / DUSt3R / Depth)

Only after smoke works.

```python
# PyTorch CUDA (Colab usually already has torch — keep if needed)
# !pip -q install torch torchvision --index-url https://download.pytorch.org/whl/cu121

!pip -q install transformers einops opencv-python-headless

# DUSt3R
!git clone --recursive https://github.com/naver/dust3r /content/dust3r
!pip -q install -e /content/dust3r

# MASt3R
!git clone --recursive https://github.com/naver/mast3r /content/mast3r
!pip -q install -e /content/mast3r

%cd /content/3d-ai/revamped_code
!propertyscan doctor --profile colab_t4
```

**Expect doctor:** CUDA yes; dust3r/mast3r packages ✓ if install succeeded.

First real run will **download multi‑GB weights** (can take 10–30+ min once). Progress heartbeats continue during long stages.

---

## Cell C3 — REAL geometry only (MASt3R) — recommended before full train

```python
%cd /content/3d-ai/revamped_code
from pathlib import Path

OUT_G = Path("/content/out_mast3r_geo")
OUT_G.mkdir(exist_ok=True)

# Live [PROGRESS] every 10s during MASt3R pair inference + global alignment
!propertyscan geometry \
  --input "{INPUT}" \
  --out "{OUT_G}" \
  --profile colab_t4 \
  --engine mast3r

!ls -la "{OUT_G}"
!cat "{OUT_G}/geometry_report.json" | head -c 2000
```

**What “healthy” looks like:**

- `[PROGRESS] mast3r_foundation | loading model …`  
- `[PROGRESS] mast3r_foundation | pair inference … still running… elapsed=120s`  
- `[PROGRESS] mast3r_foundation | global alignment …`  
- `geometry_success: true` in report  

**If it fails honestly** (good — not fake poses):

- missing package → install Cell C2  
- CUDA OOM → reduce keyframes: use fewer frames folder or lower `max_keyframes` in config  
- bad capture → check frame_intelligence report  

---

## Cell C4 — REAL full export (geometry + mock train first)

Safer than full splatfacto until geometry looks good:

```python
OUT_R = Path("/content/out_real_export")
OUT_R.mkdir(exist_ok=True)

!propertyscan export \
  --input "{INPUT}" \
  --out "{OUT_R}" \
  --profile colab_t4 \
  --engine mast3r \
  --train-backend mock

!ls -la "{OUT_R}"
!python - <<'PY'
import json
from pathlib import Path
p = Path("/content/out_real_export/final_report.json")
print(json.dumps(json.loads(p.read_text()), indent=2)[:2500])
PY
```

---

## Cell C5 — Optional: real Gaussian training (Nerfstudio)

Only if you want real splatfacto (heavier install):

```python
# Follow current Nerfstudio install docs for Colab; example sketch:
# !pip install nerfstudio
# !ns-install-cli
# then:
# !propertyscan export --input "{INPUT}" --out /content/out_full \
#   --profile colab_t4 --engine mast3r --train-backend splatfacto
```

During training you will see:

```text
[PROGRESS] splatfacto_8000iters | still running… elapsed=600s
```

and detail in `work/.../artifacts/training/ns_train.log`.

---

## Cell C6 — Download results to your PC

```python
from google.colab import files
from pathlib import Path
import shutil

OUT = Path("/content/out_real_export")  # change to your OUT
bundle = Path("/content/propertyscan_results")
if bundle.exists():
    shutil.rmtree(bundle)
bundle.mkdir()
for name in ["scene.ply", "property_scene.json", "final_report.json", "provenance.json"]:
    src = OUT / name
    if src.exists():
        shutil.copy2(src, bundle / name)
!zip -r /content/propertyscan_results.zip /content/propertyscan_results
files.download("/content/propertyscan_results.zip")
```

Or open the Colab **Files** panel and download `scene.ply` manually.

---

# Part D — How progress works (so you know it’s not stuck)

Every pipeline **stage** prints:

| When | Example |
|------|---------|
| Start | `[PROGRESS] stage:frame_quality \| started \| elapsed=0s` |
| Every **10 seconds** | `[PROGRESS] stage:frame_quality \| still running… status=… \| tick=3 \| elapsed=30s` |
| Detail updates | `[PROGRESS] mast3r_foundation \| global alignment (can take several minutes)` |
| End | `[PROGRESS] stage:export \| finished OK \| elapsed=2.1s` |

Long GPU work also has **named** heartbeats:

- `frame_quality`  
- `mast3r_foundation` / `dust3r_foundation`  
- `depth_anything_v2`  
- `splatfacto_<N>iters`  

If **tick increases** and **elapsed grows**, the process is alive even when the model is silent.

---

# Part E — Recommended order (do this)

| Step | Cell | Purpose |
|-----:|------|---------|
| 1 | A1 | GPU on |
| 2 | A2 | Clone / place code |
| 3 | A3 | Install package |
| 4 | B1/B1-alt + B2 | Dataset on disk |
| 5 | **C1 mock export** | Prove E2E + progress |
| 6 | C2 | Install MASt3R stack |
| 7 | **C3 geometry mast3r** | Real alignment |
| 8 | **C4 export mast3r + mock train** | Product files |
| 9 | C5 | Optional real splatfacto |
| 10 | C6 | Download results |

---

# Part F — Troubleshooting

| Problem | Fix |
|---------|-----|
| MediaFire `403` / tiny file | Manual upload (B1-alt) |
| `No module named propertyscan` | `%cd` into `revamped_code` and re-run `pip install -e .` |
| `MASt3R package missing` | Cell C2 |
| CUDA OOM on MASt3R | Fewer keyframes (`colab_t4` already caps); use shorter video; `pair_graph: swin-5` |
| Health gate failed | Check `geometry_report.json` / capture quality; re-record slower walk |
| No progress lines | Ensure latest `revamped_code` with `core/progress.py` is installed |
| Repo empty / no revamped_code | Push local work or upload ZIP of `revamped_code` |

---

# Part G — GitHub note for `moizogg/3d-ai`

Make sure the GitHub repo contains the **revamped engine**:

```text
3d-ai/
  revamped_code/
    propertyscan/
    configs/
    docs/
    notebooks/
    pyproject.toml
```

If only the **legacy** `worker/` is on GitHub, either:

1. Push `revamped_code/` to that repo, or  
2. Upload `revamped_code.zip` in Colab (Cell A2 manual path)

---

## One-command reminder (after setup)

```bash
# Smoke
propertyscan export -i "$INPUT" -o /content/out_mock --engine mock --train-backend mock --profile colab_t4

# Real geometry product path
propertyscan export -i "$INPUT" -o /content/out_real --engine mast3r --train-backend mock --profile colab_t4
```

Watch for `[PROGRESS] … every 10s` until `✅ Export pipeline complete.`

---

*Guide matches PropertyScan revamped engine with 10s ProgressHeartbeat on all stages + heavy GPU work.*
