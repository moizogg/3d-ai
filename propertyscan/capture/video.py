"""Video capture adapter — FFmpeg primary, OpenCV fallback."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from propertyscan.capture.base import CaptureAdapter
from propertyscan.core.config import EngineConfig
from propertyscan.core.exceptions import ValidationError
from propertyscan.domain.capture import CaptureKind, CaptureManifest

logger = logging.getLogger("propertyscan.capture.video")


class VideoCaptureAdapter(CaptureAdapter):
    """Extract candidate frames from a walkthrough video."""

    kind = CaptureKind.VIDEO

    def can_handle(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in {
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".webm",
            ".m4v",
        }

    def load_manifest(self, path: Path, config: EngineConfig) -> CaptureManifest:
        path = Path(path)
        if not path.is_file():
            raise ValidationError(
                f"Video file not found: {path}",
                suggestion="Pass a path to an .mp4 / .mov / .mkv file.",
            )
        if path.stat().st_size == 0:
            raise ValidationError(
                f"Video file is empty: {path}",
                suggestion="Re-export or re-record the walkthrough.",
            )
        suffix = path.suffix.lower()
        if suffix not in {e.lower() for e in config.capture.video_extensions}:
            raise ValidationError(
                f"Unsupported video extension: {suffix}",
                suggestion=f"Supported: {config.capture.video_extensions}",
            )

        width = height = None
        duration_s = fps = None
        warnings: list[str] = []
        try:
            meta = _probe_video_opencv(path)
            if meta:
                width, height, duration_s, fps = meta
        except Exception as exc:
            warnings.append(f"Could not probe video metadata: {exc}")

        return CaptureManifest(
            kind=CaptureKind.VIDEO,
            source_path=path.resolve(),
            exists=True,
            file_count=1,
            duration_s=duration_s,
            fps=fps,
            width=width,
            height=height,
            warnings=warnings,
        )

    def materialize_frames(
        self,
        manifest: CaptureManifest,
        work_dir: Path,
        config: EngineConfig,
    ) -> list[Path]:
        out_dir = Path(work_dir) / "candidates"
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = Path(manifest.source_path)
        fps = config.capture.video_fps
        # ``max_candidate_frames <= 0`` = extract all fps-sampled frames (no hard cap)
        max_frames = int(config.capture.max_candidate_frames)

        frames = _extract_ffmpeg(video_path, out_dir, fps=fps, max_frames=max_frames)
        if not frames:
            frames = _extract_opencv(video_path, out_dir, fps=fps, max_frames=max_frames)
        if not frames:
            raise ValidationError(
                f"Failed to extract frames from video: {video_path}",
                suggestion=(
                    "Install FFmpeg on PATH or opencv-python-headless, "
                    "and ensure the codec is readable."
                ),
            )
        logger.info("extracted %d candidate frames from %s", len(frames), video_path.name)
        return frames


def _extract_ffmpeg(
    video_path: Path,
    out_dir: Path,
    *,
    fps: float,
    max_frames: int,
) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    pattern = str(out_dir / "frame_%04d.jpg")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
    ]
    # Only apply a hard frame count when max_frames > 0
    if max_frames > 0:
        cmd.extend(["-frames:v", str(max_frames)])
    cmd.append(pattern)
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception as exc:
        logger.warning("FFmpeg extraction failed: %s", exc)
        return []
    return sorted(out_dir.glob("frame_*.jpg"))


def _extract_opencv(
    video_path: Path,
    out_dir: Path,
    *,
    fps: float,
    max_frames: int,
) -> list[Path]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / max(fps, 0.1))), 1)
    paths: list[Path] = []
    idx = 0
    saved = 0
    unlimited = max_frames <= 0
    while unlimited or saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            out = out_dir / f"frame_{saved + 1:04d}.jpg"
            cv2.imwrite(str(out), frame)
            paths.append(out)
            saved += 1
        idx += 1
    cap.release()
    return paths


def _probe_video_opencv(
    path: Path,
) -> tuple[int | None, int | None, float | None, float | None] | None:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or None
    n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (n / fps) if fps and n else None
    cap.release()
    return w, h, duration, fps
