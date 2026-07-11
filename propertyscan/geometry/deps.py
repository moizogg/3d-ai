"""Optional dependency probes for foundation models (no COLMAP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DepStatus:
    name: str
    available: bool
    detail: str = ""


def probe_torch() -> DepStatus:
    try:
        import torch  # type: ignore

        cuda = torch.cuda.is_available()
        return DepStatus(
            "torch",
            True,
            f"version={torch.__version__} cuda={cuda}",
        )
    except ImportError as exc:
        return DepStatus("torch", False, str(exc))


def probe_dust3r() -> DepStatus:
    try:
        import dust3r  # type: ignore  # noqa: F401
        from dust3r.model import AsymmetricCroCo3DStereo  # noqa: F401
        from dust3r.inference import inference  # noqa: F401
        from dust3r.image_pairs import make_pairs  # noqa: F401
        from dust3r.cloud_opt import global_aligner  # noqa: F401

        return DepStatus("dust3r", True, "import ok")
    except Exception as exc:
        return DepStatus("dust3r", False, str(exc))


def probe_mast3r() -> DepStatus:
    try:
        from mast3r.model import AsymmetricMASt3R  # noqa: F401

        return DepStatus("mast3r", True, "import ok")
    except Exception as exc:
        return DepStatus("mast3r", False, str(exc))


def probe_depth_anything() -> DepStatus:
    try:
        import transformers  # type: ignore  # noqa: F401

        return DepStatus("transformers", True, "import ok (Depth Anything pipeline)")
    except Exception as exc:
        return DepStatus("transformers", False, str(exc))


def foundation_ready(*, need_mast3r: bool = False, need_dust3r: bool = False) -> dict[str, Any]:
    """Return readiness map for Phase 4 engines."""
    t = probe_torch()
    d = probe_dust3r()
    m = probe_mast3r()
    da = probe_depth_anything()
    cuda = False
    if t.available:
        try:
            import torch  # type: ignore

            cuda = bool(torch.cuda.is_available())
        except Exception:
            cuda = False

    ok_dust3r = t.available and d.available
    ok_mast3r = t.available and d.available and m.available
    return {
        "torch": t,
        "dust3r": d,
        "mast3r": m,
        "depth_anything": da,
        "cuda": cuda,
        "can_run_dust3r": ok_dust3r and (cuda or True),  # device policy applied later
        "can_run_mast3r": ok_mast3r,
        "can_run_depth": t.available and da.available,
        "need_mast3r_ok": (not need_mast3r) or ok_mast3r,
        "need_dust3r_ok": (not need_dust3r) or ok_dust3r,
    }


INSTALL_HINT_MAST3R = (
    "Install DUSt3R + MASt3R (NAVER) and PyTorch CUDA. "
    "See docs/PHASE4_WALKTHROUGH.md. "
    "Typical: clone naver/dust3r and naver/mast3r, pip install -e, "
    "then load naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric."
)

INSTALL_HINT_DUST3R = (
    "Install DUSt3R and PyTorch CUDA. See docs/PHASE4_WALKTHROUGH.md. "
    "Model: naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt."
)

INSTALL_HINT_DEPTH = (
    "Install: pip install transformers torch torchvision pillow. "
    "Model: depth-anything/Depth-Anything-V2-Small-hf (T4) or Base/Large."
)
