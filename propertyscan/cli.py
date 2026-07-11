"""Command-line interface for the PropertyScan engine.

Commands:
  version  — print package version
  doctor   — environment / device / config health check
  config   — dump resolved EngineConfig as JSON
  frames   — Phase 2 capture + frame intelligence pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from propertyscan import __version__
from propertyscan.core.config import load_config
from propertyscan.core.device import resolve_device
from propertyscan.core.exceptions import EngineError
from propertyscan.core.logging import setup_logging


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"propertyscan {__version__}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    setup_logging(level="INFO", fmt="console")
    print("=" * 60)
    print(" PropertyScan AI — Doctor")
    print("=" * 60)

    print(f"\n[package] version={__version__}")

    # Config
    try:
        cfg = load_config(profile=args.profile, quality=args.quality)
        print(f"[config]  profile={cfg.engine.profile}  geometry.engine={cfg.geometry.engine}")
        print(f"[config]  mast3r_model={cfg.geometry.mast3r_model}")
        print(f"[config]  dust3r_model={cfg.geometry.dust3r_model}")
        print(f"[config]  depth.size={cfg.depth.size}  prefer_cuda={cfg.device.prefer_cuda}")
    except Exception as exc:
        print(f"[config]  ERROR: {exc}")
        return 1

    # Python deps
    print("\n[deps]")
    for mod in ("yaml", "pydantic", "numpy", "PIL"):
        try:
            __import__("PIL" if mod == "PIL" else mod if mod != "yaml" else "yaml")
            print(f"  ✓ {mod}")
        except ImportError:
            print(f"  ✗ {mod} missing")

    # Optional heavy deps
    torch_ok = False
    try:
        import torch  # type: ignore

        torch_ok = True
        print(f"  ✓ torch {torch.__version__}")
    except ImportError:
        print("  · torch not installed (optional until Phase 4)")

    try:
        import cv2  # type: ignore

        print(f"  ✓ opencv {cv2.__version__}")
    except ImportError:
        print("  · opencv not installed (optional video fallback)")

    import shutil

    print(f"  {'✓' if shutil.which('ffmpeg') else '·'} ffmpeg on PATH")

    # Device
    device = resolve_device(prefer_cuda=cfg.device.prefer_cuda)
    print("\n[device]")
    print(f"  device={device.device}  cuda_available={device.cuda_available}")
    if device.device_name:
        print(f"  name={device.device_name}")
    if device.total_vram_gb is not None:
        print(f"  vram_total_gb={device.total_vram_gb}  free_gb={device.free_vram_gb}")
    if device.cuda_version:
        print(f"  cuda={device.cuda_version}")

    print("\n[readiness]")
    print("  Phase 2: `propertyscan frames`")
    print("  Phase 3–4: `propertyscan geometry --engine mock|mast3r|dust3r|auto`")
    print("  Phase 5: `propertyscan train --engine mock --train-backend mock`")
    print("  Phase 6: `propertyscan export --engine mock --train-backend mock`")
    print("  Phase 7: `propertyscan benchmark --data tests/fixtures --out ./_bench`")
    try:
        from propertyscan.geometry.deps import foundation_ready

        fr = foundation_ready(need_mast3r=True, need_dust3r=True)
        print(f"  dust3r package : {'✓' if fr['dust3r'].available else '· missing'}")
        print(f"  mast3r package : {'✓' if fr['mast3r'].available else '· missing'}")
        print(
            f"  transformers   : {'✓' if fr['depth_anything'].available else '· missing (depth)'}"
        )
        print(f"  cuda for models: {'✓' if fr['cuda'] else '· no'}")
    except Exception as exc:
        print(f"  foundation probe error: {exc}")
    if not device.cuda_available:
        print("  WARN: CUDA not available. Real MASt3R/DUSt3R need a GPU (e.g. Colab T4).")
        print("        use --engine mock for plumbing tests; no silent fake poses.")
    if torch_ok and device.cuda_available:
        print("  OK: CUDA visible — ready for Phase 4 real foundation models.")

    print("\n[status] doctor complete")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config(profile=args.profile, quality=args.quality)
    print(json.dumps(cfg.to_dict(), indent=2))
    return 0


def _cmd_frames(args: argparse.Namespace) -> int:
    """Run Phase 2 capture + frame intelligence."""
    from propertyscan.pipeline.frame_pipeline import frames_summary, run_frames_pipeline

    setup_logging(level="INFO", fmt="console")
    input_path = Path(args.input)
    output_dir = Path(args.out)
    print("=" * 60)
    print(" PropertyScan AI — Frame Intelligence (Phase 2)")
    print("=" * 60)
    print(f"  input : {input_path}")
    print(f"  out   : {output_dir}")
    print(f"  profile: {args.profile or 'default'}")
    try:
        ctx = run_frames_pipeline(
            input_path,
            output_dir,
            profile=args.profile,
            quality=args.quality,
        )
    except EngineError as exc:
        print(f"\nERROR: {exc}")
        return 1
    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    summary = frames_summary(ctx)
    print("\n[result]")
    print(f"  capture_kind      : {summary['capture_kind']}")
    print(f"  accepted_keyframes: {summary['accepted_keyframes']}")
    print(f"  rejected          : {summary['rejected']}")
    print(f"  scene_type        : {summary['scene_type']}")
    print(f"  texture_score     : {summary['texture_score']}")
    print(f"  report            : {output_dir / 'frame_intelligence.json'}")
    print(f"  job_id            : {summary['job_id']}")
    print("\n✅ Frame intelligence complete.")
    return 0


def _cmd_geometry(args: argparse.Namespace) -> int:
    """Run Phase 2 frames + Phase 3 geometry (use --engine mock without GPU weights)."""
    from propertyscan.core.exceptions import HealthGateError
    from propertyscan.pipeline.geometry_pipeline import (
        geometry_summary,
        run_geometry_pipeline,
    )

    setup_logging(level="INFO", fmt="console")
    print("=" * 60)
    print(" PropertyScan AI — Geometry Pipeline (Phase 3)")
    print("=" * 60)
    print(f"  input : {args.input}")
    print(f"  out   : {args.out}")
    print(f"  engine: {args.engine or 'from config'}")
    try:
        ctx = run_geometry_pipeline(
            Path(args.input),
            Path(args.out),
            profile=args.profile,
            quality=args.quality,
            engine=args.engine,
        )
    except HealthGateError as exc:
        print(f"\nHEALTH GATE: {exc}")
        return 2
    except EngineError as exc:
        print(f"\nERROR: {exc}")
        return 1
    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    s = geometry_summary(ctx)
    print("\n[result]")
    print(f"  provider           : {s['provider']}")
    print(f"  geometry_success   : {s['geometry_success']}")
    print(f"  registered         : {s['registered']} ({s['registered_fraction']:.0%})")
    print(f"  health_score       : {s['health_score']} passed={s['health_passed']}")
    print(f"  depth_attached     : {s['depth_attached']}")
    print(f"  report             : {Path(args.out) / 'geometry_report.json'}")
    print("\n✅ Geometry pipeline complete.")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """Frames → geometry → dataset → Gaussian train."""
    from propertyscan.core.exceptions import HealthGateError
    from propertyscan.pipeline.train_pipeline import run_train_pipeline, train_summary

    setup_logging(level="INFO", fmt="console")
    print("=" * 60)
    print(" PropertyScan AI — Train Pipeline (Phase 5)")
    print("=" * 60)
    print(f"  input         : {args.input}")
    print(f"  out           : {args.out}")
    print(f"  geometry      : {args.engine or 'from config'}")
    print(f"  train backend : {args.train_backend or 'from config'}")
    try:
        ctx = run_train_pipeline(
            Path(args.input),
            Path(args.out),
            profile=args.profile,
            quality=args.quality,
            engine=args.engine,
            train_backend=args.train_backend,
        )
    except HealthGateError as exc:
        print(f"\nHEALTH GATE: {exc}")
        return 2
    except EngineError as exc:
        print(f"\nERROR: {exc}")
        return 1
    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    s = train_summary(ctx)
    print("\n[result]")
    print(f"  dataset_frames : {s['dataset_frames']}")
    print(f"  depth / init   : {s['has_depth']} / {s['has_init_cloud']}")
    print(f"  train_backend  : {s['train_backend']} success={s['train_success']}")
    print(f"  iterations     : {s['iterations']}")
    print(f"  ply            : {s['ply']}")
    print(f"  report         : {Path(args.out) / 'train_report.json'}")
    print("\n✅ Train pipeline complete.")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """Full pipeline through inspect + PropertyScene + export."""
    from propertyscan.core.exceptions import HealthGateError
    from propertyscan.pipeline.export_pipeline import export_summary, run_export_pipeline

    setup_logging(level="INFO", fmt="console")
    print("=" * 60)
    print(" PropertyScan AI — Export Pipeline (Phase 6)")
    print("=" * 60)
    print(f"  input         : {args.input}")
    print(f"  out           : {args.out}")
    print(f"  geometry      : {args.engine or 'from config'}")
    print(f"  train backend : {args.train_backend or 'from config'}")
    try:
        ctx = run_export_pipeline(
            Path(args.input),
            Path(args.out),
            profile=args.profile,
            quality=args.quality,
            engine=args.engine,
            train_backend=args.train_backend,
        )
    except HealthGateError as exc:
        print(f"\nHEALTH GATE: {exc}")
        return 2
    except EngineError as exc:
        print(f"\nERROR: {exc}")
        return 1
    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    s = export_summary(ctx)
    print("\n[result]")
    print(f"  quality        : {s['quality_overall']} ({s['quality_status']})")
    print(f"  gaussians      : {s['gaussians_before']} → {s['gaussians_after']}")
    print(f"  exports        : {s['exports']}")
    print(f"  scene_id       : {s['scene_id']}")
    print(f"  final_report   : {Path(args.out) / 'final_report.json'}")
    print("\n✅ Export pipeline complete.")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Phase 7: multi-scene benchmark + experiment registry."""
    from propertyscan.research.benchmark import BenchmarkRunner

    setup_logging(level="INFO", fmt="console")
    print("=" * 60)
    print(" PropertyScan AI — Benchmark (Phase 7)")
    print("=" * 60)
    print(f"  data   : {args.data}")
    print(f"  out    : {args.out}")
    print(f"  engine : {args.engine}  train={args.train_backend}")
    try:
        runner = BenchmarkRunner(
            data_dir=Path(args.data),
            output_dir=Path(args.out),
            profile=args.profile,
            quality=args.quality,
            engine=args.engine,
            train_backend=args.train_backend,
        )
        results = runner.run_all()
    except EngineError as exc:
        print(f"\nERROR: {exc}")
        return 1
    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    ok = sum(1 for r in results if r.success)
    print(f"\n[result] {ok}/{len(results)} scenes succeeded")
    for r in results:
        flag = "OK" if r.success else "FAIL"
        q = (r.metrics or {}).get("quality_overall")
        print(f"  [{flag}] {r.scene_id}  quality={q}  err={r.error or '-'}")
    print(f"  summary : {Path(args.out) / 'benchmark_summary.json'}")
    print(f"  history : {Path(args.out) / 'registry' / 'history.jsonl'}")
    print("\n✅ Benchmark complete.")
    return 0 if ok == len(results) and results else (0 if not results else 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="propertyscan",
        description="PropertyScan AI — AI-first Geometry Engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ver = sub.add_parser("version", help="Print package version")
    p_ver.set_defaults(func=_cmd_version)

    p_doc = sub.add_parser("doctor", help="Environment and config health check")
    p_doc.add_argument(
        "--profile",
        default=None,
        help="Config profile: default | colab_t4 | quality_gpu",
    )
    p_doc.add_argument(
        "--quality",
        default=None,
        choices=["draft", "standard", "high"],
        help="Training quality preset overlay",
    )
    p_doc.set_defaults(func=_cmd_doctor)

    p_cfg = sub.add_parser("config", help="Dump resolved EngineConfig as JSON")
    p_cfg.add_argument("--profile", default=None)
    p_cfg.add_argument(
        "--quality",
        default=None,
        choices=["draft", "standard", "high"],
    )
    p_cfg.set_defaults(func=_cmd_config)

    p_frames = sub.add_parser(
        "frames",
        help="Run capture + frame intelligence (Phase 2)",
    )
    p_frames.add_argument("--input", "-i", required=True, help="Video file or image folder")
    p_frames.add_argument("--out", "-o", required=True, help="Output directory")
    p_frames.add_argument(
        "--profile",
        default=None,
        help="Config profile: default | colab_t4 | quality_gpu",
    )
    p_frames.add_argument(
        "--quality",
        default=None,
        choices=["draft", "standard", "high"],
    )
    p_frames.set_defaults(func=_cmd_frames)

    p_geom = sub.add_parser(
        "geometry",
        help="Run frames + geometry router/fusion/health (Phase 3; use --engine mock)",
    )
    p_geom.add_argument("--input", "-i", required=True, help="Video or image folder")
    p_geom.add_argument("--out", "-o", required=True, help="Output directory")
    p_geom.add_argument("--profile", default=None)
    p_geom.add_argument(
        "--quality",
        default=None,
        choices=["draft", "standard", "high"],
    )
    p_geom.add_argument(
        "--engine",
        default=None,
        choices=["mast3r", "dust3r", "auto", "mock", "arkit"],
        help="Geometry engine (mock for CI without CUDA/weights)",
    )
    p_geom.set_defaults(func=_cmd_geometry)

    p_train = sub.add_parser(
        "train",
        help="Frames + geometry + dataset + Gaussian training (Phase 5)",
    )
    p_train.add_argument("--input", "-i", required=True)
    p_train.add_argument("--out", "-o", required=True)
    p_train.add_argument("--profile", default=None)
    p_train.add_argument(
        "--quality",
        default=None,
        choices=["draft", "standard", "high"],
    )
    p_train.add_argument(
        "--engine",
        default=None,
        choices=["mast3r", "dust3r", "auto", "mock", "arkit"],
        help="Geometry engine (mock without GPU weights)",
    )
    p_train.add_argument(
        "--train-backend",
        default=None,
        choices=["mock", "splatfacto"],
        help="Training backend (mock without ns-train)",
    )
    p_train.set_defaults(func=_cmd_train)

    p_export = sub.add_parser(
        "export",
        help="Full pipeline through inspect + PropertyScene + PLY export (Phase 6)",
    )
    p_export.add_argument("--input", "-i", required=True)
    p_export.add_argument("--out", "-o", required=True)
    p_export.add_argument("--profile", default=None)
    p_export.add_argument(
        "--quality",
        default=None,
        choices=["draft", "standard", "high"],
    )
    p_export.add_argument(
        "--engine",
        default=None,
        choices=["mast3r", "dust3r", "auto", "mock", "arkit"],
    )
    p_export.add_argument(
        "--train-backend",
        default=None,
        choices=["mock", "splatfacto"],
    )
    p_export.set_defaults(func=_cmd_export)

    p_bench = sub.add_parser(
        "benchmark",
        help="Run pipeline on a folder of scenes; write research layout + history.jsonl",
    )
    p_bench.add_argument(
        "--data",
        required=True,
        help="Benchmarks root (subfolders = scenes) or a single scene folder",
    )
    p_bench.add_argument("--out", "-o", required=True, help="Benchmark output root")
    p_bench.add_argument("--profile", default=None)
    p_bench.add_argument(
        "--quality",
        default=None,
        choices=["draft", "standard", "high"],
    )
    p_bench.add_argument(
        "--engine",
        default="mock",
        choices=["mast3r", "dust3r", "auto", "mock", "arkit"],
    )
    p_bench.add_argument(
        "--train-backend",
        default="mock",
        choices=["mock", "splatfacto"],
    )
    p_bench.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
