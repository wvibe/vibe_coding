#!/usr/bin/env python3
"""CLI for motion estimation and stabilization on pre-extracted frame sets.

Subcommands
-----------
estimate   — Run affine2d motion estimation on a frame set.
stabilize  — Apply stabilization and produce before/after comparison.
report     — Summarize analysis results.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analyze_motion")


def _resolve_frame_dir(frame_set: Path, fps: float) -> Path:
    """Resolve the actual frame directory for a given fps."""
    fps_dir = f"{int(fps)}fps" if fps == int(fps) else f"{fps}fps"
    frame_dir = frame_set / fps_dir
    if not frame_dir.is_dir():
        logger.error("Frame directory not found: %s", frame_dir)
        sys.exit(1)
    return frame_dir


# ---------------------------------------------------------------------------
# estimate
# ---------------------------------------------------------------------------

def cmd_estimate(args: argparse.Namespace) -> None:
    """Run motion estimation on a frame set."""
    frame_set = Path(args.frame_set).expanduser().resolve()
    frame_dir = _resolve_frame_dir(frame_set, args.fps)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    method = getattr(args, "method", "affine2d")

    if method == "dpvo":
        _cmd_estimate_dpvo(args, frame_dir, output_dir)
    elif method == "raft":
        _cmd_estimate_raft(args, frame_dir, output_dir)
    else:
        _cmd_estimate_affine(args, frame_dir, output_dir)


def _cmd_estimate_affine(args: argparse.Namespace, frame_dir: Path, output_dir: Path) -> None:
    from vibelab.ego_video.motion.motion_analysis import (
        _load_manifest_metadata,
        _write_json_atomic,
        compute_cumulative_transforms,
        compute_trajectory,
        estimate_frame_motion,
    )

    manifest_fs = _load_manifest_metadata(frame_dir, args.fps)

    logger.info("Estimating motion: %s (affine2d)", frame_dir)
    motion_data = estimate_frame_motion(
        frame_dir=frame_dir,
        manifest_fs=manifest_fs,
        max_corners=args.max_corners,
        quality_level=args.quality_level,
        min_distance=args.min_distance,
    )

    transforms_data = compute_cumulative_transforms(motion_data)
    trajectory_data = compute_trajectory(motion_data, smoothing_radius=args.smoothing_radius)

    valid_frames = [f for f in motion_data["frames"] if not f.get("skipped")]
    inliers = [f["inlier_count"] for f in valid_frames if "inlier_count" in f]
    trans_mags = [
        (f["dx"] ** 2 + f["dy"] ** 2) ** 0.5
        for f in valid_frames if "dx" in f
    ]
    summary = {
        "method": "affine2d",
        "num_frames": len(valid_frames),
        "avg_translation_mag": round(float(sum(trans_mags) / len(trans_mags)), 4) if trans_mags else 0,
        "max_translation_mag": round(float(max(trans_mags)), 4) if trans_mags else 0,
        "avg_inlier_count": round(float(sum(inliers) / len(inliers)), 1) if inliers else 0,
    }

    from vibelab.ego_video.motion.motion_analysis import _write_json_atomic
    _write_json_atomic(output_dir / "motion.json", motion_data)
    _write_json_atomic(output_dir / "transforms.json", transforms_data)
    _write_json_atomic(output_dir / "trajectory.json", trajectory_data)
    _write_json_atomic(output_dir / "motion_summary.json", summary)

    logger.info("Motion estimation complete → %s", output_dir)
    logger.info(
        "Summary: %d frames, avg_translation=%.2f, avg_inliers=%.0f",
        summary["num_frames"], summary["avg_translation_mag"], summary["avg_inlier_count"],
    )


def _cmd_estimate_dpvo(args: argparse.Namespace, frame_dir: Path, output_dir: Path) -> None:
    from vibelab.ego_video.motion.dpvo_bridge import (
        _write_json_atomic,
        build_trajectory_json,
        estimate_dpvo_motion,
    )
    from vibelab.ego_video.motion.motion_analysis import _load_manifest_metadata

    manifest_fs = _load_manifest_metadata(frame_dir, args.fps)

    # Load calibration
    if args.calibration:
        cal_path = Path(args.calibration).expanduser().resolve()
        calibration = json.loads(cal_path.read_text(encoding="utf-8"))
    elif manifest_fs:
        # Try to get from parent manifest
        parent_manifest = frame_dir.parent / "manifest.json"
        if parent_manifest.exists():
            m = json.loads(parent_manifest.read_text(encoding="utf-8"))
            calibration = m.get("calibration", {})
        else:
            logger.error("No calibration found. Use --calibration or ensure manifest has calibration.")
            sys.exit(1)
    else:
        logger.error("--calibration required for dpvo method")
        sys.exit(1)

    try:
        motion_data = estimate_dpvo_motion(
            frame_dir=frame_dir,
            calibration=calibration,
            output_dir=output_dir,
            manifest_fs=manifest_fs,
            source_fps=args.fps,
            smoothing_radius=args.smoothing_radius,
            dpvo_model_path=args.dpvo_model,
            dpvo_config=args.dpvo_config,
            stride=args.stride,
        )
    except ImportError as e:
        logger.error("%s", e)
        sys.exit(1)

    # Build trajectory.json for stabilize step
    corrections = motion_data["frames"]
    actual_radius = min(args.smoothing_radius, len(corrections) // 2)
    is_fisheye = calibration.get("model", "").lower() == "fisheye" or bool(calibration.get("distortion"))
    trajectory_data = build_trajectory_json(
        corrections, args.smoothing_radius, actual_radius,
        motion_data.get("calibration_pinhole", {}), is_fisheye)

    # Summary
    valid = [f for f in corrections if f.get("confidence") == "normal"]
    summary = {
        "method": "dpvo",
        "num_frames": motion_data["num_frames"],
        "num_poses": motion_data["num_poses"],
        "num_aligned": motion_data["num_aligned"],
        "num_normal_confidence": len(valid),
        "avg_rotation_magnitude_deg": round(
            float(np.mean([f["rotation_magnitude_deg"] for f in corrections])), 4
        ) if corrections else 0,
    }

    # Build transforms.json (cumulative corrections for consistency with other methods)
    transforms_data = {
        "method": "dpvo",
        "transform_convention": {
            "direction": "correction_for_frame_n",
            "composition_order": "direct_application",
            "point_convention": "column_vector",
            "definition": "H_corr applied via warpPerspective to stabilize frame n",
        },
        "transforms": [
            {"frame_index": c["frame_index"], "matrix_3x3": c["H_corr_3x3"]}
            for c in corrections
        ],
    }

    _write_json_atomic(output_dir / "motion.json", motion_data)
    _write_json_atomic(output_dir / "transforms.json", transforms_data)
    _write_json_atomic(output_dir / "trajectory.json", trajectory_data)
    _write_json_atomic(output_dir / "motion_summary.json", summary)

    logger.info("DPVO estimation complete → %s", output_dir)
    logger.info("Summary: %d frames, %d poses, %d aligned, %d normal confidence",
                summary["num_frames"], summary["num_poses"],
                summary["num_aligned"], summary["num_normal_confidence"])


def _cmd_estimate_raft(args: argparse.Namespace, frame_dir: Path, output_dir: Path) -> None:
    from vibelab.ego_video.motion.motion_analysis import _load_manifest_metadata
    from vibelab.ego_video.motion.raft_flow import (
        _write_json_atomic,
        compute_cumulative_transforms_raft,
        compute_homography_trajectory,
        estimate_raft_motion,
    )

    manifest_fs = _load_manifest_metadata(frame_dir, args.fps)

    logger.info("Estimating motion: %s (RAFT)", frame_dir)
    motion_data = estimate_raft_motion(
        frame_dir=frame_dir,
        model_name=getattr(args, "raft_model", "raft_small"),
        device_str=getattr(args, "device", "auto"),
        flow_resize=getattr(args, "flow_resize", 512),
        ransac_threshold=getattr(args, "ransac_threshold", 3.0),
        manifest_fs=manifest_fs,
    )

    transforms_data = compute_cumulative_transforms_raft(motion_data)
    trajectory_data = compute_homography_trajectory(
        transforms_data,
        smoothing_radius=args.smoothing_radius,
    )

    valid = [f for f in motion_data["frames"] if not f.get("skipped")]
    summary = {
        "method": "raft",
        "num_frames": len(valid),
        "avg_inlier_ratio": round(float(np.mean([f.get("inlier_ratio", 0) for f in valid])), 4) if valid else 0,
        "avg_flow_magnitude": round(float(np.mean([f.get("flow_magnitude_mean", 0) for f in valid])), 4) if valid else 0,
        "model_types": {mt: sum(1 for f in valid if f.get("model_type") == mt)
                        for mt in ("homography", "affine", "identity")},
    }

    _write_json_atomic(output_dir / "motion.json", motion_data)
    _write_json_atomic(output_dir / "transforms.json", transforms_data)
    _write_json_atomic(output_dir / "trajectory.json", trajectory_data)
    _write_json_atomic(output_dir / "motion_summary.json", summary)

    logger.info("RAFT estimation complete → %s", output_dir)
    logger.info(
        "Summary: %d frames, avg_inlier=%.2f, models=%s",
        summary["num_frames"], summary["avg_inlier_ratio"], summary["model_types"],
    )


# ---------------------------------------------------------------------------
# stabilize
# ---------------------------------------------------------------------------

def cmd_stabilize(args: argparse.Namespace) -> None:
    """Apply stabilization and produce before/after comparison."""
    frame_set = Path(args.frame_set).expanduser().resolve()
    frame_dir = _resolve_frame_dir(frame_set, args.fps)
    output_dir = Path(args.output_dir).expanduser().resolve()

    traj_path = output_dir / "trajectory.json"
    if not traj_path.exists():
        logger.error("trajectory.json not found in %s — run 'estimate' first", output_dir)
        sys.exit(1)
    trajectory_data = json.loads(traj_path.read_text(encoding="utf-8"))

    method = trajectory_data.get("method", "affine2d")
    logger.info("Stabilizing: %s → %s (method=%s)", frame_dir, output_dir, method)

    if method in ("raft", "dpvo"):
        from vibelab.ego_video.motion.raft_flow import _write_json_atomic, apply_raft_stabilization
        report = apply_raft_stabilization(
            frame_dir=frame_dir,
            trajectory_data=trajectory_data,
            output_dir=output_dir,
            comparison_width=args.comparison_width,
        )
    else:
        from vibelab.ego_video.motion.motion_analysis import _write_json_atomic, apply_stabilization
        report = apply_stabilization(
            frame_dir=frame_dir,
            trajectory_data=trajectory_data,
            output_dir=output_dir,
            crop_ratio=args.crop_ratio,
            comparison_width=args.comparison_width,
        )

    _write_json_atomic(output_dir / "stabilization_report.json", report)

    summary = report.get("summary", {})
    logger.info("Stabilization complete → %s", output_dir)
    if summary:
        logger.info(
            "Summary: improvement=%.1f%%, center_crop=%.1f%%",
            summary.get("improvement_ratio", 0) * 100,
            summary.get("avg_center_crop_improvement", 0) * 100,
        )


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate stabilization with background-aware metrics."""
    from vibelab.ego_video.motion.stabilization_metrics import evaluate_stabilization

    frame_set = Path(args.frame_set).expanduser().resolve()
    frame_dir = _resolve_frame_dir(frame_set, args.fps)
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()
    stab_dir = analysis_dir / "stabilized"

    if not stab_dir.is_dir():
        logger.error("stabilized/ not found in %s — run 'stabilize' first", analysis_dir)
        sys.exit(1)

    # Detect method from motion.json
    motion_path = analysis_dir / "motion.json"
    method = "unknown"
    if motion_path.exists():
        method = json.loads(motion_path.read_text()).get("method", "unknown")

    logger.info("Evaluating: raw=%s vs stab=%s (method=%s)", frame_dir, stab_dir, method)
    report = evaluate_stabilization(
        raw_frame_dir=frame_dir,
        stabilized_frame_dir=stab_dir,
        method=method,
    )

    # Write report
    report_path = analysis_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Print summary
    summary = report.get("summary", {})
    logger.info("Evaluation complete → %s", report_path)
    s05 = summary.get("stability_0.5", {})
    rm = summary.get("residual_motion", {})
    if s05:
        logger.info("Stability (crop 0.5): raw=%.4f, stab=%.4f, improvement=%.1f%%",
                     s05.get("avg_raw_combined", 0), s05.get("avg_stab_combined", 0),
                     s05.get("avg_improvement", 0) * 100)
    if rm:
        logger.info("Residual motion: raw=%.2f, stab=%.2f, improvement=%.1f%%",
                     rm.get("avg_raw_magnitude", 0), rm.get("avg_stab_magnitude", 0),
                     rm.get("avg_improvement", 0) * 100)


def cmd_report(args: argparse.Namespace) -> None:
    """Summarize analysis results."""
    analysis_dir = Path(args.analysis_dir).expanduser().resolve()

    result: dict = {"analysis_dir": str(analysis_dir)}

    # Load what's available
    for name in ["motion_summary.json", "motion.json", "transforms.json",
                  "trajectory.json", "stabilization_report.json"]:
        path = analysis_dir / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            key = name.replace(".json", "")
            if name == "stabilization_report.json":
                result["stabilization_summary"] = data.get("summary", {})
            elif name == "motion_summary.json":
                result["motion_summary"] = data
            elif name == "motion.json":
                result["num_frames"] = data.get("num_frames")
                result["method"] = data.get("method")

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # Human-readable
    print(f"Analysis: {analysis_dir}")
    print(f"Method: {result.get('method', 'N/A')}")
    print(f"Frames: {result.get('num_frames', 'N/A')}")

    ms = result.get("motion_summary", {})
    if ms:
        print(f"\nMotion Summary:")
        print(f"  Avg translation: {ms.get('avg_translation_mag', 'N/A')}")
        print(f"  Max translation: {ms.get('max_translation_mag', 'N/A')}")
        print(f"  Avg inliers: {ms.get('avg_inlier_count', 'N/A')}")

    ss = result.get("stabilization_summary", {})
    if ss:
        print(f"\nStabilization Summary:")
        print(f"  Avg correction: {ss.get('avg_correction_magnitude', 'N/A')}")
        print(f"  Improvement: {ss.get('improvement_ratio', 0) * 100:.1f}%")
        adj = ss.get("avg_adjacent_improvement")
        if adj is not None:
            print(f"  Adjacent improvement: {adj * 100:.1f}%")
        cc = ss.get("avg_center_crop_improvement")
        if cc is not None:
            print(f"  Center-crop improvement: {cc * 100:.1f}%")
    elif not ms:
        print("\nNo analysis data found.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_motion",
        description="Motion estimation and stabilization on pre-extracted frame sets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- estimate ---
    est = sub.add_parser("estimate", help="Run motion estimation.")
    est.add_argument("--frame-set", required=True, help="Path to frame set directory.")
    est.add_argument("--fps", type=float, required=True, help="Target fps to analyze.")
    est.add_argument("--method", choices=["affine2d", "raft", "dpvo"], default="affine2d", help="Estimation method.")
    est.add_argument("--output-dir", required=True, help="Output directory for results.")
    est.add_argument("--smoothing-radius", type=int, default=5, help="Smoothing radius.")
    # affine2d params
    est.add_argument("--max-corners", type=int, default=200, help="Max corners (affine2d).")
    est.add_argument("--quality-level", type=float, default=0.01, help="Corner quality (affine2d).")
    est.add_argument("--min-distance", type=float, default=30.0, help="Min distance (affine2d).")
    # raft params
    est.add_argument("--raft-model", default="raft_small", help="RAFT model name.")
    est.add_argument("--device", default="auto", help="Device: auto/mps/cpu.")
    est.add_argument("--flow-resize", type=int, default=512, help="Max dim for RAFT inference.")
    est.add_argument("--ransac-threshold", type=float, default=3.0, help="RANSAC threshold (raft).")
    # dpvo params
    est.add_argument("--dpvo-model", default="dpvo.pth", help="DPVO model weights path.")
    est.add_argument("--dpvo-config", default="config/default.yaml", help="DPVO config path.")
    est.add_argument("--stride", type=int, default=1, help="Frame stride for DPVO.")
    est.add_argument("--calibration", help="Path to calibration JSON (required for dpvo).")
    est.set_defaults(func=cmd_estimate)

    # --- stabilize ---
    stab = sub.add_parser("stabilize", help="Apply stabilization + comparison.")
    stab.add_argument("--frame-set", required=True, help="Path to frame set directory.")
    stab.add_argument("--fps", type=float, required=True, help="Target fps to stabilize.")
    stab.add_argument("--output-dir", required=True, help="Analysis directory (with trajectory.json).")
    stab.add_argument("--crop-ratio", type=float, default=0.0, help="Crop border ratio.")
    stab.add_argument("--comparison-width", type=int, default=960, help="Per-side width for comparison images.")
    stab.set_defaults(func=cmd_stabilize)

    # --- evaluate ---
    evl = sub.add_parser("evaluate", help="Evaluate stabilization with background-aware metrics.")
    evl.add_argument("--frame-set", required=True, help="Path to frame set directory.")
    evl.add_argument("--fps", type=float, required=True, help="Target fps.")
    evl.add_argument("--analysis-dir", required=True, help="Analysis dir (with stabilized/).")
    evl.set_defaults(func=cmd_evaluate)

    # --- report ---
    rep = sub.add_parser("report", help="Summarize analysis results.")
    rep.add_argument("--analysis-dir", required=True, help="Analysis output directory.")
    rep.add_argument("--json", action="store_true", help="JSON output.")
    rep.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
