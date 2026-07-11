# Phase 2 Walkthrough — Capture + Frame Intelligence

**Status:** Complete  
**Tests:** 37 passed  
**CLI verified:** `propertyscan frames` on synthetic fixtures  

---

## What was built

End-to-end path from **video / image folder / image sequence → scored keyframes + SceneDescriptor**.

This is the first reliability filter before MASt3R/DUSt3R (Phase 4). Bad frames no longer silently enter geometry.

---

## New modules

```text
propertyscan/
  capture/
    base.py, detect.py
    video.py          # FFmpeg → OpenCV fallback
    image_folder.py
    image_sequence.py
    arkit.py          # stub (clear error)
  intelligence/
    quality.py        # Laplacian blur, exposure, contrast, confidence
    dedup.py          # dHash Hamming
    keyframes.py      # rank + cap + promote low-confidence if needed
    classify.py       # SceneDescriptor heuristics
    pipeline.py       # report writers
  pipeline/
    frame_pipeline.py
    stages/s01…s06_*.py
```

---

## Pipeline stages (1–6)

| Stage | Name | Output |
|------:|------|--------|
| 1 | `validate_capture` | `CaptureManifest` |
| 2 | `decode_frames` | candidate image paths |
| 3 | `frame_quality` | per-frame metrics + status |
| 4 | `dedup` | near-duplicates marked |
| 5 | `keyframes` | selected `FrameSet` on disk |
| 6 | `classify_scene` | `SceneDescriptor` |

Artifacts:

```text
output/
  frame_intelligence.json
  metadata.json
  provenance.json
  work/<job_id>/artifacts/
    frames/candidates/
    frames/selected/keyframe_XXXX.jpg
    frames/quarantine/quarantine.json
    reports/frame_intelligence.json
```

---

## How to run

```bash
cd revamped_code
pip install -e ".[dev]"

propertyscan frames --input tests/fixtures/frames --out ./_out_frames --profile colab_t4
# or a real walkthrough:
propertyscan frames --input path/to/walk.mp4 --out ./_out_frames --profile colab_t4

pytest -q
```

Smoke result on fixtures:

- `capture_kind: image_sequence`
- `accepted_keyframes: 10` (with default min_frames / max_keyframes)
- report written successfully

---

## Status rules (quality) — **reliable_v2** (legacy Laplacian/dHash removed)

1. **Hard reject only:** unreadable, near-total black/white clip, true motion smear  
2. **Soft rank:** clip-relative Tenengrad + texture-aware protection for white walls  
3. **Redundancy:** camera motion (optical flow / MAD) — **not** aggressive dHash  
4. **Low-texture walls:** never culled as “blurry” or motion-redundant  
5. **Keyframes:** diversity + quality budget (`max_keyframes` / `min_frames`)  

See [VALIDATION_RELIABLE_V2.md](VALIDATION_RELIABLE_V2.md).

---

## Config knobs

| Key | Meaning |
|-----|---------|
| `capture.video_fps` | Candidate extraction rate |
| `capture.max_candidate_frames` | Hard cap on extract |
| `capture.min_frames` | Fail if fewer keyframes |
| `frame_intelligence.blur_threshold` | Laplacian floor |
| `frame_intelligence.phash_threshold` | Dedup Hamming |
| `frame_intelligence.max_keyframes` | Output cap; `0` = keep all (`colab_t4` uses unlimited) |

---

## Intentionally not done (Phase 3+)

- GeometryProvider / router / MASt3R / DUSt3R  
- Depth Anything  
- Real ARKit pose/depth ingest (stub only)  
- Gaussian training  

---

## Why this phase matters (simple English)

### Where Phase 2 sits in the whole product

```text
Phone video / photos
        ↓
  Phase 2: pick the best frames   ← you are here
        ↓
  Phase 3–4: figure out camera positions in 3D
        ↓
  Phase 5+: train and clean the Gaussian splat tour
```

Garbage frames in → garbage 3D out. Phase 2 is the **filter and editor** that prepares a clean photo set before any heavy AI geometry runs.

### Why we needed this phase

Indoor real-estate video is messy: blur from walking, freezes when you stop, dark corners, endless near-identical hallway shots.

The **legacy** approach (strict Laplacian blur + aggressive “duplicate” hashing) often **deleted good frames** (especially white walls) and kept the wrong ones. That starved alignment and helped destroy your old 3D scans.

Phase 2 rebuilds validation to be **reliable**:

- Hard-delete only truly broken frames (black/white crash, real motion smear).  
- Soft-score the rest.  
- Remove true freezes using **camera motion**, not “looks similar on a tiny hash.”  
- Protect low-texture walls so they are not labeled “blurry.”

### What effect this has on final results

| Better Phase 2 | Effect on the tour |
|----------------|--------------------|
| Fewer false “blur / duplicate” kills | More useful views for MASt3R/DUSt3R → **stronger alignment** |
| Less true freeze frames | Less wasted compute, less pose confusion |
| Keyframes with real motion diversity | Better room coverage → fewer missing walls |
| Clear quarantine reports | You can see *why* a frame was dropped |
| Good SceneDescriptor | Later router can pick a better geometry model |

**Bottom line:** Phase 2 does not draw the 3D house, but it decides which photos the AI is allowed to trust. That decision often matters more than training settings.

### One-sentence summary

**Phase 2 turns a messy walkthrough video into a short list of trustworthy keyframes so geometry AI has a fair chance to lock the cameras correctly.**

---

## Readiness for Phase 3

`RunContext` now can hold:

- `capture_manifest`
- `frame_set` (selected keyframes)
- `scene_descriptor`

Phase 3 will add `GeometryProvider` ABCs, router, depth interfaces, mocks — consuming this `FrameSet`.

**Waiting for your prompt:** `start phase 3`
