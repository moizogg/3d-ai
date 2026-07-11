# Reliable Frame Validation v2

**Replaces:** legacy absolute Laplacian blur + aggressive dHash hard-reject  
**Why:** That approach destroyed real-estate scans (false blur on white walls, mass false duplicates on slow walks).

## Policy

### Hard reject only

| Status | When |
|--------|------|
| `unreadable` | decode failure |
| `clipped_black` | ≥92% near-black pixels |
| `clipped_white` | ≥92% near-white pixels |
| `motion_smear` | low **clip-relative** sharpness **and** sufficient texture **and** high motion |

### Soft rank (never killed for being a white wall)

- **Tenengrad** sharpness + **clip-relative percentile**
- **Texture / edge density** — low texture is *protected*, not punished
- Soft exposure / contrast influence score only
- `low_rank` = soft flag; still selectable if needed

### Redundancy = camera motion, not dHash

- Optical flow (OpenCV) or MAD fallback
- `redundant` only if **near-stationary** since last kept frame
- Slow walk past similar walls **keeps** frames

### Keyframes

- Greedy temporal diversity + quality rank
- Budget: `max_keyframes` (`0` or negative = keep all selectable frames)
- Floor: `min_frames`

## Config (meaningful keys)

```yaml
frame_intelligence:
  max_keyframes: 120
  low_rank_threshold: 25.0
  clip_black_pct: 0.92
  clip_white_pct: 0.92
  min_motion_to_keep: 0.8
  motion_smear_sharpness_percentile: 8.0
  motion_smear_min_texture: 0.06
  motion_smear_min_flow: 2.5
```

Legacy `blur_threshold` / `phash_threshold` are **ignored**.
