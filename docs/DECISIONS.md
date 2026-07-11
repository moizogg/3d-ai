# Architecture Decision Records (Revamped Engine)

## ADR-001 — Clean-room package under `revamped_code/`

**Decision:** Build a new package instead of refactoring `worker/`.

**Why:** Legacy is COLMAP-centric; mission requires AI-first geometry without COLMAP wrappers.

## ADR-002 — No COLMAP in Phases 1–7

**Decision:** Omit COLMAP/SIFT/BA primary path entirely for this rewrite.

**Why:** User mission; optional future provider only behind the same interface.

## ADR-003 — Official ViT-Large foundation checkpoints

**Decision:** Default to NAVER MASt3R / DUSt3R ViT-Large 512 models.

**Why:** Best alignment quality; user priority is tours that do not fail on alignment.  
“~10GB” refers mainly to **peak VRAM during alignment**, not a single 10GB weight file.

## ADR-004 — T4 now, 4090 later via config profiles

**Decision:** `colab_t4` primary test profile; `quality_gpu` for upgrades.

**Why:** User will validate on Colab T4, then upgrade hardware without rewriting architecture.

## ADR-005 — PropertyScene over PLY-as-product

**Decision:** Canonical product is `PropertyScene`; exporters translate.

**Why:** Reconstruction Bible Part 4; enables provenance, multi-export, future formats.
