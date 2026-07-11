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

---

## Before you start

1. Colab → **Runtime → GPU (T4)**  
2. Repo root must contain `propertyscan/`, `configs/`, `pyproject.toml`  
3. If MediaFire wget fails → download on PC → upload to `/content/data/archive.zip`

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
for p in sorted(EXTRACT.iterdir())[:40]:
    print(" ", p.name, "DIR" if p.is_dir() else f"file {p.stat().st_size}")

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

def find_input(root: Path):
    vids = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT]
    if vids:
        return sorted(vids, key=lambda p: -p.stat().st_size)[0]
    best, best_n = None, 0
    for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
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
