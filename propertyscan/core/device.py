"""GPU / CPU device discovery and VRAM helpers.

Target environments:
  - Colab Tesla T4 (~16GB) for current testing
  - Future quality GPUs (e.g. RTX 4090) via config profile only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Snapshot of the compute device available for this run.

    Purpose:
        Give stages a single, serializable view of hardware capability.

    Fields:
        device: torch-style device string (``cuda`` or ``cpu``).
        cuda_available: Whether CUDA is usable.
        device_name: Human-readable GPU name when present.
        total_vram_gb: Total VRAM in GiB when CUDA is available.
        free_vram_gb: Free VRAM estimate when queryable.
        cuda_version: CUDA runtime version string if known.
    """

    device: str
    cuda_available: bool
    device_name: str | None = None
    total_vram_gb: float | None = None
    free_vram_gb: float | None = None
    cuda_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "cuda_available": self.cuda_available,
            "device_name": self.device_name,
            "total_vram_gb": self.total_vram_gb,
            "free_vram_gb": self.free_vram_gb,
            "cuda_version": self.cuda_version,
        }


def _try_import_torch() -> Any | None:
    try:
        import torch  # type: ignore

        return torch
    except ImportError:
        return None


def resolve_device(*, prefer_cuda: bool = True) -> DeviceInfo:
    """Resolve the best available compute device.

    Inputs:
        prefer_cuda: If True, use CUDA when available.

    Outputs:
        DeviceInfo describing the selected device.

    Limitations:
        Without torch installed, always returns CPU. Geometry stages in later
        phases will refuse to invent fake poses on CPU unless explicitly allowed.
    """
    torch = _try_import_torch()
    if torch is None:
        return DeviceInfo(device="cpu", cuda_available=False)

    cuda_ok = bool(prefer_cuda and torch.cuda.is_available())
    if not cuda_ok:
        return DeviceInfo(
            device="cpu",
            cuda_available=False,
            cuda_version=getattr(torch.version, "cuda", None),
        )

    props = torch.cuda.get_device_properties(0)
    total = float(props.total_memory) / (1024**3)
    free: float | None = None
    try:
        free_b, _total_b = torch.cuda.mem_get_info(0)
        free = float(free_b) / (1024**3)
    except Exception:
        free = None

    return DeviceInfo(
        device="cuda",
        cuda_available=True,
        device_name=torch.cuda.get_device_name(0),
        total_vram_gb=round(total, 2),
        free_vram_gb=round(free, 2) if free is not None else None,
        cuda_version=getattr(torch.version, "cuda", None),
    )


def empty_cuda_cache() -> None:
    """Release cached CUDA memory when torch + CUDA are available."""
    torch = _try_import_torch()
    if torch is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def peak_vram_gb() -> float | None:
    """Return peak allocated CUDA memory in GiB for device 0, if available."""
    torch = _try_import_torch()
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        return round(float(torch.cuda.max_memory_allocated(0)) / (1024**3), 3)
    except Exception:
        return None
