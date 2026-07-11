"""Build a clean Nerfstudio-style dataset from frames + geometry + depth.

Layout::

    dataset/
      images/<frame files>
      depths/<optional depth pngs>
      transforms.json
      sparse_pc.ply   # optional dense/sparse init cloud
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path

from propertyscan.core.config import EngineConfig
from propertyscan.core.exceptions import EngineError
from propertyscan.domain.dataset import TrainingDataset
from propertyscan.domain.depth import DepthResult
from propertyscan.domain.frames import FrameSet
from propertyscan.domain.geometry import GeometryResult
from propertyscan.geometry.fusion.fuse import FusedGeometry

logger = logging.getLogger("propertyscan.dataset.builder")


class DatasetBuildError(EngineError):
    """Dataset assembly failed validation."""


class DatasetBuilder:
    """Assemble training inputs from validated geometry.

    Purpose:
        Produce the cleanest possible 3DGS inputs (poses, images, depth, init cloud).

    Inputs:
        FrameSet (accepted keyframes), GeometryResult or FusedGeometry, optional DepthResult.

    Outputs:
        TrainingDataset on disk.

    Non-responsibilities:
        Pose estimation, Gaussian optimization.
    """

    def build(
        self,
        *,
        frame_set: FrameSet,
        geometry: GeometryResult | FusedGeometry,
        output_dir: Path,
        config: EngineConfig,
        depth: DepthResult | None = None,
    ) -> TrainingDataset:
        if isinstance(geometry, FusedGeometry):
            geom = geometry.geometry
            if depth is None:
                depth = geometry.depth
        else:
            geom = geometry

        if not geom.success or geom.pose_graph is None:
            raise DatasetBuildError(
                f"Cannot build dataset from failed geometry: {geom.error_message}",
                suggestion="Re-run geometry with a successful provider (mast3r/dust3r/mock).",
            )

        registered = [c for c in geom.pose_graph.cameras if c.registered and c.c2w]
        if len(registered) < 2:
            raise DatasetBuildError(
                f"Need >= 2 registered cameras with poses; got {len(registered)}.",
                suggestion="Check geometry registration before training.",
            )

        root = Path(output_dir)
        images_dir = root / "images"
        depths_dir = root / "depths"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Map image_name -> keyframe path
        accepted = {f.filename: f for f in frame_set.get_accepted()}
        notes: list[str] = []
        copied = 0
        frames_json: list[dict] = []

        depth_by_name: dict[str, Path] = {}
        if depth is not None and depth.success:
            for dm in depth.depth_maps:
                depth_by_name[dm.image_name] = Path(dm.path)

        for cam in registered:
            src_frame = accepted.get(cam.image_name)
            if src_frame is None:
                # Try basename match
                for name, fr in accepted.items():
                    if name == cam.image_name or Path(name).stem == Path(cam.image_name).stem:
                        src_frame = fr
                        break
            if src_frame is None:
                notes.append(f"skip camera without keyframe file: {cam.image_name}")
                continue

            src = Path(src_frame.filepath)
            if not src.is_file():
                notes.append(f"missing image file: {src}")
                continue

            dest_name = src.name
            dest = images_dir / dest_name
            _link_or_copy(src, dest)
            copied += 1

            w = cam.width or src_frame.width or 0
            h = cam.height or src_frame.height or 0
            fl_x = cam.fx or (0.9 * max(w, 1))
            fl_y = cam.fy or fl_x
            cx = cam.cx if cam.cx is not None else w / 2.0
            cy = cam.cy if cam.cy is not None else h / 2.0

            entry: dict = {
                "file_path": f"images/{dest_name}",
                "transform_matrix": cam.c2w,
                "w": w,
                "h": h,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "cx": cx,
                "cy": cy,
                "camera_angle_x": 2 * math.atan(w / (2 * fl_x)) if fl_x and w else 0.0,
            }

            # Depth linkage (relative monocular maps)
            depth_src = depth_by_name.get(cam.image_name) or depth_by_name.get(src.name)
            if depth_src is None:
                # stem match
                stem = Path(cam.image_name).stem
                for k, p in depth_by_name.items():
                    if Path(k).stem == stem:
                        depth_src = p
                        break
            if depth_src is not None and Path(depth_src).is_file():
                depths_dir.mkdir(parents=True, exist_ok=True)
                depth_dest = depths_dir / f"{Path(dest_name).stem}.png"
                _link_or_copy(Path(depth_src), depth_dest)
                entry["depth_file_path"] = f"depths/{depth_dest.name}"

            frames_json.append(entry)

        if len(frames_json) < 2:
            raise DatasetBuildError(
                f"Dataset has only {len(frames_json)} frames with images+poses.",
                suggestion="Ensure keyframe files exist on disk for registered cameras.",
                details={"notes": notes},
            )

        # Init point cloud
        init_ply: Path | None = None
        has_cloud = False
        if geom.point_cloud and geom.point_cloud.path and Path(geom.point_cloud.path).is_file():
            init_ply = root / "sparse_pc.ply"
            shutil.copy2(geom.point_cloud.path, init_ply)
            has_cloud = True
            notes.append(f"init_cloud_from={geom.point_cloud.path}")
        elif geom.artifacts.get("dense_ply") and Path(geom.artifacts["dense_ply"]).is_file():
            init_ply = root / "sparse_pc.ply"
            shutil.copy2(geom.artifacts["dense_ply"], init_ply)
            has_cloud = True
            notes.append(f"init_cloud_from={geom.artifacts['dense_ply']}")

        has_depth = any("depth_file_path" in f for f in frames_json)
        transforms = {
            "camera_model": "OPENCV",
            "frames": frames_json,
            "provider": geom.provider_name,
            "propertyscan_dataset": True,
        }
        if has_cloud and init_ply is not None:
            transforms["ply_file_path"] = init_ply.name

        transforms_path = root / "transforms.json"
        transforms_path.write_text(json.dumps(transforms, indent=2), encoding="utf-8")

        # Optional pre-downscale for T4 RAM (images_N folders for nerfstudio)
        downscale = max(1, int(config.training.downscale_factor))
        if downscale > 1:
            self._write_downscaled(images_dir, downscale)
            notes.append(f"downscale_factor={downscale}")

        dataset = TrainingDataset(
            root=root,
            images_dir=images_dir,
            transforms_path=transforms_path,
            frame_count=len(frames_json),
            has_depth=has_depth,
            has_init_point_cloud=has_cloud,
            init_ply_path=init_ply,
            depth_dir=depths_dir if has_depth else None,
            downscale_factor=downscale,
            provider_name=geom.provider_name,
            notes=notes,
            metadata={
                "registered_cameras": len(registered),
                "copied_images": copied,
            },
        )
        self.validate(dataset)
        logger.info(
            "Dataset ready: %d frames, depth=%s, init_cloud=%s → %s",
            dataset.frame_count,
            dataset.has_depth,
            dataset.has_init_point_cloud,
            root,
        )
        return dataset

    def validate(self, dataset: TrainingDataset) -> None:
        """Hard checks before training is allowed to start."""
        if not dataset.transforms_path.is_file():
            raise DatasetBuildError("transforms.json missing")
        data = json.loads(dataset.transforms_path.read_text(encoding="utf-8"))
        frames = data.get("frames") or []
        if len(frames) < 2:
            raise DatasetBuildError("transforms.json has fewer than 2 frames")
        for fr in frames:
            rel = fr.get("file_path")
            if not rel:
                raise DatasetBuildError("frame missing file_path")
            img = dataset.root / rel
            if not img.is_file():
                raise DatasetBuildError(f"image missing for transforms entry: {rel}")
            if "transform_matrix" not in fr:
                raise DatasetBuildError(f"frame missing transform_matrix: {rel}")

    @staticmethod
    def _write_downscaled(images_dir: Path, factor: int) -> None:
        """Create images_{factor} for Nerfstudio --downscale-factor."""
        from PIL import Image

        out_dir = images_dir.parent / f"images_{factor}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(images_dir.iterdir()):
            if src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            dest = out_dir / src.name
            if dest.exists():
                continue
            with Image.open(src) as im:
                w, h = im.size
                nw, nh = max(1, w // factor), max(1, h // factor)
                im.resize((nw, nh), Image.Resampling.LANCZOS).save(dest, quality=90)


def _link_or_copy(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dest)
