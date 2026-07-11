# PropertyScan AI — Colab Real Test Guide

> **Copy buttons:** open the HTML version in a browser for a **Copy** button on every cell:  
> **[`COLAB_REAL_TEST_GUIDE.html`](COLAB_REAL_TEST_GUIDE.html)**  
> (Double-click the file, or right-click → Open with browser.)  
> Plain Markdown viewers often hide copy controls; the HTML page always shows them.

| Item | Value |
|------|--------|
| Dataset | [MediaFire archive.zip](https://download853.mediafire.com/rtmlk9ypemegyIXh1yTx9nFxnjYYGn5l5Y7jgmzy4HCb7cekTEY__64r_wSqKfc3Cns4Iohj56JzG_hKv6SpxVLLkM8y_ucaQCHgrNLuS14oFWQmjlp4IuwbXMKOG3aP7MOpotpk3DGZor2n-Kk5rH-m3LedwvDJXId-PhLfrTqM_Q/zknvd5nfvsygnsr/archive.zip) |
| Code | https://github.com/moizogg/3d-ai |
| **Working directory** | **`/content/3d-ai`** (repo root — not `revamped_code/`) |
| GPU | Colab Free **T4** |
| Progress | `[PROGRESS] …` every **10 seconds** |
| Frames | **`colab_t4` keeps all selectable frames** (`max_keyframes: 0`, `max_candidate_frames: 0`) — no 80-cap |

---

## Before you start

1. Colab → **Runtime → GPU (T4)**  
2. Repo root must contain `propertyscan/`, `configs/`, `pyproject.toml`  
3. If MediaFire wget fails → download on PC → upload to `/content/data/archive.zip`  
4. **Keep-all frames:** a full ~400-image tour is used end-to-end. Mock export is fine; real MASt3R on T4 may be slow or OOM — if so, lower `pair_graph` further or use a 4090 later, do **not** re-enable an 80-frame cap unless you choose to

---

## Cell A1 — GPU check

```python
!nvidia-smi
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
```

---

## Cell A2 — Clone GitHub (work in `/content/3d-ai`)

```python
from pathlib import Path

%cd /content
REPO = "https://github.com/moizogg/3d-ai.git"
DIR = Path("/content/3d-ai")

if not DIR.exists():
    !git clone --depth 1 {REPO}
else:
    print("Repo already present:", DIR)

%cd /content/3d-ai
!pwd
!ls -la
!ls propertyscan configs pyproject.toml
```

---

## Cell A3 — Install package

```python
%cd /content/3d-ai
!pip -q install -e ".[dev]"
!propertyscan version
!propertyscan doctor --profile colab_t4
```

---

## Cell B1 — Download MediaFire ZIP

```python
from pathlib import Path

DATA = Path("/content/data")
DATA.mkdir(parents=True, exist_ok=True)
ZIP = DATA / "archive.zip"

URL = "https://download853.mediafire.com/rtmlk9ypemegyIXh1yTx9nFxnjYYGn5l5Y7jgmzy4HCb7cekTEY__64r_wSqKfc3Cns4Iohj56JzG_hKv6SpxVLLkM8y_ucaQCHgrNLuS14oFWQmjlp4IuwbXMKOG3aP7MOpotpk3DGZor2n-Kk5rH-m3LedwvDJXId-PhLfrTqM_Q/zknvd5nfvsygnsr/archive.zip"

!wget -c -O "{ZIP}" "{URL}" || true
print("size_mb", ZIP.stat().st_size / 1e6 if ZIP.exists() else "MISSING")
```

---

## Cell B1-alt — Confirm manual upload

```python
from pathlib import Path

ZIP = Path("/content/data/archive.zip")
assert ZIP.exists() and ZIP.stat().st_size > 1_000_000, "Upload archive.zip to /content/data/ first"
print("OK", ZIP, "MB", ZIP.stat().st_size / 1e6)
```

---

## Cell B2 — Unzip + find INPUT

**What “Top-level” means:** only names **directly inside** the extract folder (one level).  
It is **not** a full count of every image in the ZIP.

Example:

```text
kaggle_scene/                 ← EXTRACT root (top-level)
  folder_a/                   ← listed as DIR at top-level
    img_0001.jpg              ← NOT listed in top-level (nested)
    ...
    img_0400.jpg
  readme.txt                  ← listed as file at top-level
```

So if you see **39 images at top-level**, the other **~361** are almost always in **subfolders**. That is normal for Kaggle ZIPs.

Cell B2 below:

1. Prints top-level (quick peek)  
2. Counts **all** images recursively  
3. Picks the folder with the **most** images (or a video)  
4. If images are scattered across many folders, **copies them into one flat folder** for the pipeline  

```python
from pathlib import Path
import zipfile
import shutil

DATA = Path("/content/data")
ZIP = DATA / "archive.zip"
EXTRACT = DATA / "kaggle_scene"
EXTRACT.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ZIP, "r") as zf:
    zf.extractall(EXTRACT)

print("=== Top-level only (NOT all files in zip) ===")
for p in sorted(EXTRACT.iterdir())[:40]:
    kind = "DIR" if p.is_dir() else f"file {p.stat().st_size}"
    print(" ", p.name, kind)

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

all_images = [
    p for p in EXTRACT.rglob("*")
    if p.is_file() and p.suffix.lower() in IMG_EXT
]
all_videos = [
    p for p in EXTRACT.rglob("*")
    if p.is_file() and p.suffix.lower() in VIDEO_EXT
]
print("=== Full recursive counts ===")
print("Total images in zip tree:", len(all_images))
print("Total videos in zip tree:", len(all_videos))

def count_images_in_dir(d: Path) -> int:
    try:
        return sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)
    except Exception:
        return 0

# Prefer a single folder that already holds most images
best, best_n = None, 0
for d in [EXTRACT] + [p for p in EXTRACT.rglob("*") if p.is_dir()]:
    n = count_images_in_dir(d)
    if n > best_n:
        best, best_n = d, n

if all_videos:
    INPUT = sorted(all_videos, key=lambda p: -p.stat().st_size)[0]
    print("Using VIDEO:", INPUT)
elif best is not None and best_n >= max(8, int(0.8 * len(all_images)) if all_images else 8):
    # One folder has ~all images
    INPUT = best
    print(f"Using image folder with {best_n} images:", INPUT)
elif len(all_images) >= 8:
    # Scattered across subfolders → flatten into one INPUT dir
    INPUT = DATA / "all_frames_flat"
    if INPUT.exists():
        shutil.rmtree(INPUT)
    INPUT.mkdir(parents=True)
    for i, src in enumerate(sorted(all_images)):
        dest = INPUT / f"frame_{i:05d}{src.suffix.lower()}"
        shutil.copy2(src, dest)
    print(f"Images were nested/scattered. Flattened {len(all_images)} →", INPUT)
else:
    INPUT = None

print("INPUT =>", INPUT)
assert INPUT is not None, "Could not find video or enough images"
```

---

## Cell C1 — SMOKE export (mock)

```python
%cd /content/3d-ai
from pathlib import Path

OUT = Path("/content/out_mock")
OUT.mkdir(exist_ok=True)

!propertyscan export --input "{INPUT}" --out "{OUT}" --profile colab_t4 --engine mock --train-backend mock

!ls -la "{OUT}"
print("scene.ply exists:", (OUT / "scene.ply").exists())
print("final_report:", (OUT / "final_report.json").exists())
```

---

## Cell C2 — Install MASt3R / DUSt3R / Depth

```python
!pip -q install transformers einops opencv-python-headless

!git clone --recursive https://github.com/naver/dust3r /content/dust3r
!pip -q install -e /content/dust3r

!git clone --recursive https://github.com/naver/mast3r /content/mast3r
!pip -q install -e /content/mast3r

%cd /content/3d-ai
!propertyscan doctor --profile colab_t4
```

---

## Cell C3 — Real MASt3R geometry

```python
%cd /content/3d-ai
from pathlib import Path

OUT_G = Path("/content/out_mast3r_geo")
OUT_G.mkdir(exist_ok=True)

!propertyscan geometry --input "{INPUT}" --out "{OUT_G}" --profile colab_t4 --engine mast3r

!ls -la "{OUT_G}"
!python -c "from pathlib import Path; p=Path('/content/out_mast3r_geo/geometry_report.json'); print(p.read_text()[:2000] if p.exists() else 'no report')"
```

---

## Cell C4 — Real export (mast3r + mock train)

```python
%cd /content/3d-ai
from pathlib import Path

OUT_R = Path("/content/out_real_export")
OUT_R.mkdir(exist_ok=True)

!propertyscan export --input "{INPUT}" --out "{OUT_R}" --profile colab_t4 --engine mast3r --train-backend mock

!ls -la "{OUT_R}"
!python -c "import json; from pathlib import Path; p=Path('/content/out_real_export/final_report.json'); print(json.dumps(json.loads(p.read_text()), indent=2)[:2500] if p.exists() else 'missing')"
```

---

## Cell C6 — Download results

```python
from google.colab import files
from pathlib import Path
import shutil

OUT = Path("/content/out_real_export")
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

---

## Progress (healthy)

```text
[PROGRESS] stage:frame_quality | still running… elapsed=20s
[PROGRESS] mast3r_foundation | global alignment … elapsed=180s
[PROGRESS] stage:export | finished OK | elapsed=2.1s
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No Copy button in MD preview | Open **`COLAB_REAL_TEST_GUIDE.html`** in a browser |
| MediaFire 403 | Manual upload to `/content/data/archive.zip` |
| `No module named propertyscan` | `%cd /content/3d-ai` then `pip install -e .` |
| Wrong directory | Always **`/content/3d-ai`** (repo root) |

---

## Layout on GitHub / Colab

```text
/content/3d-ai/
  propertyscan/
  configs/
  docs/
  pyproject.toml
```
