"""Frame-based motion estimation and stabilization for pre-extracted image sets.

Operates on directories of sequential PNG frames (from prepare_frames.py),
not on video files directly.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def _load_frames_sorted(frame_dir: Path) -> list[tuple[str, np.ndarray]]:
    """Load PNG frames from directory in sorted order. Returns (filename, bgr) pairs."""
    pngs = sorted(frame_dir.glob("frame_*.png"))
    results: list[tuple[str, np.ndarray]] = []
    for p in pngs:
        img = cv2.imread(str(p))
        if img is None:
            logger.warning("Skipping unreadable frame: %s", p.name)
            results.append((p.name, None))  # type: ignore[arg-type]
        else:
            results.append((p.name, img))
    return results


def _load_manifest_metadata(frame_set_dir: Path, fps: float) -> dict | None:
    """Try to load manifest from parent frame set and extract metadata for target fps."""
    manifest_path = frame_set_dir / "manifest.json"
    if not manifest_path.exists():
        # Try parent (frame_dir is e.g. clip_001/30fps, manifest is clip_001/manifest.json)
        manifest_path = frame_set_dir.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fps_dir = f"{int(fps)}fps" if fps == int(fps) else f"{fps}fps"
        for fs in manifest.get("frame_sets", []):
            if fs.get("directory") == fps_dir:
                return fs
    except Exception as e:
        logger.warning("Failed to read manifest: %s", e)
    return None


# ---------------------------------------------------------------------------
# Motion estimation (affine2d)
# ---------------------------------------------------------------------------

def _estimate_affine2d_pair(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    max_corners: int = 200,
    quality_level: float = 0.01,
    min_distance: float = 30.0,
) -> dict:
    """Estimate affine2d motion between a pair of grayscale frames."""
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=max_corners,
        qualityLevel=quality_level, minDistance=min_distance,
    )
    identity = {"dx": 0.0, "dy": 0.0, "da": 0.0, "scale": 1.0,
                "inlier_count": 0,
                "matrix_2x3": [[1, 0, 0], [0, 1, 0]]}

    if prev_pts is None or len(prev_pts) < 4:
        return identity

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    good_prev = prev_pts[status.flatten() == 1]
    good_curr = curr_pts[status.flatten() == 1]

    if len(good_prev) < 4 or len(good_curr) < 4:
        return {**identity, "inlier_count": int(len(good_curr))}

    transform, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr)
    if transform is None:
        return identity

    a, b = float(transform[0, 0]), float(transform[1, 0])
    dx = float(transform[0, 2])
    dy = float(transform[1, 2])
    da = float(np.arctan2(b, a))
    scale = float(np.sqrt(a * a + b * b))
    inlier_count = int(inliers.sum()) if inliers is not None else int(len(good_curr))

    return {
        "dx": round(dx, 6),
        "dy": round(dy, 6),
        "da": round(da, 8),
        "scale": round(scale, 6),
        "inlier_count": inlier_count,
        "matrix_2x3": [[round(float(transform[r, c]), 8) for c in range(3)] for r in range(2)],
    }


def estimate_frame_motion(
    frame_dir: Path,
    manifest_fs: dict | None = None,
    max_corners: int = 200,
    quality_level: float = 0.01,
    min_distance: float = 30.0,
) -> dict:
    """Estimate affine2d frame-to-frame motion on sequential PNG frames.

    Returns dict matching motion.json schema.
    """
    loaded = _load_frames_sorted(frame_dir)
    if not loaded:
        raise ValueError(f"No frames found in {frame_dir}")

    frames_meta: list[dict] = []

    # Frame 0: identity
    first_entry: dict = {
        "frame_index": 0,
        "dx": 0.0, "dy": 0.0, "da": 0.0, "scale": 1.0,
        "inlier_count": 0,
        "matrix_2x3": [[1, 0, 0], [0, 1, 0]],
    }
    if manifest_fs and manifest_fs.get("frames"):
        mf = manifest_fs["frames"][0]
        first_entry["source_frame"] = mf.get("source_frame")
        first_entry["timestamp_sec"] = mf.get("timestamp_sec")
    frames_meta.append(first_entry)

    prev_gray = None
    for i, (fname, bgr) in enumerate(loaded):
        if bgr is None:
            if i > 0:
                entry: dict = {
                    "frame_index": i,
                    "skipped": True,
                    "reason": "corrupt_or_missing",
                }
                frames_meta.append(entry)
            prev_gray = None
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if i == 0:
            prev_gray = gray
            continue

        if prev_gray is None:
            # Previous frame was skipped — use identity
            entry = {
                "frame_index": i,
                "dx": 0.0, "dy": 0.0, "da": 0.0, "scale": 1.0,
                "inlier_count": 0,
                "matrix_2x3": [[1, 0, 0], [0, 1, 0]],
            }
        else:
            entry = _estimate_affine2d_pair(
                prev_gray, gray,
                max_corners=max_corners,
                quality_level=quality_level,
                min_distance=min_distance,
            )
            entry["frame_index"] = i

        # Add manifest metadata if available
        if manifest_fs and manifest_fs.get("frames") and i < len(manifest_fs["frames"]):
            mf = manifest_fs["frames"][i]
            entry["source_frame"] = mf.get("source_frame")
            entry["timestamp_sec"] = mf.get("timestamp_sec")

        frames_meta.append(entry)
        prev_gray = gray

    # Get fps info from manifest
    source_fps = manifest_fs.get("target_fps", 0) if manifest_fs else 0
    actual_fps = manifest_fs.get("actual_fps", source_fps) if manifest_fs else source_fps

    return {
        "method": "affine2d",
        "source_fps": source_fps,
        "actual_fps": actual_fps,
        "num_frames": len(loaded),
        "frames": frames_meta,
    }


# ---------------------------------------------------------------------------
# Cumulative transforms
# ---------------------------------------------------------------------------

def _embed_2x3_to_3x3(m: list[list[float]]) -> np.ndarray:
    """Embed a 2×3 affine matrix into a 3×3 homogeneous matrix."""
    return np.array([m[0], m[1], [0, 0, 1]], dtype=np.float64)


def compute_cumulative_transforms(motion_data: dict) -> dict:
    """Accumulate per-frame motions into cumulative 3×3 transforms."""
    transforms: list[dict] = []
    cumulative = np.eye(3, dtype=np.float64)

    for frame in motion_data["frames"]:
        if frame.get("skipped"):
            transforms.append({
                "frame_index": frame["frame_index"],
                "skipped": True,
                "matrix_3x3": cumulative.tolist(),
            })
            continue

        if frame["frame_index"] == 0:
            transforms.append({
                "frame_index": 0,
                "matrix_3x3": np.eye(3).tolist(),
            })
            continue

        rel = _embed_2x3_to_3x3(frame["matrix_2x3"])
        cumulative = rel @ cumulative
        transforms.append({
            "frame_index": frame["frame_index"],
            "matrix_3x3": [[round(float(v), 8) for v in row] for row in cumulative.tolist()],
        })

    return {
        "method": motion_data["method"],
        "transform_convention": {
            "direction": "frame_0_to_frame_n",
            "composition_order": "right_multiply",
            "point_convention": "column_vector",
            "definition": "T_0->n = T_{n-1->n} @ T_0->{n-1}. Column-vector: p_n = T_0->n @ p_0",
        },
        "transforms": transforms,
    }


# ---------------------------------------------------------------------------
# Trajectory + smoothing + correction
# ---------------------------------------------------------------------------

def _moving_average_1d(values: np.ndarray, radius: int) -> np.ndarray:
    """Moving average smoothing, edge-padded."""
    if radius <= 0 or len(values) < 2:
        return values.copy()
    kernel_size = radius * 2 + 1
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(padded, kernel, mode="valid")


def compute_trajectory(
    motion_data: dict,
    smoothing_radius: int = 5,
) -> dict:
    """Compute raw and smoothed cumulative trajectory + corrections."""
    frames = [f for f in motion_data["frames"] if not f.get("skipped")]
    n = len(frames)

    if n < 2:
        return {
            "method": motion_data["method"],
            "smoothing_radius": smoothing_radius,
            "smoothing_radius_actual": 0,
            "frames": [{
                "frame_index": 0,
                "raw_cum_dx": 0, "raw_cum_dy": 0, "raw_cum_da": 0,
                "smooth_cum_dx": 0, "smooth_cum_dy": 0, "smooth_cum_da": 0,
                "corr_dx": 0, "corr_dy": 0, "corr_da": 0,
            }] if n == 1 else [],
        }

    # Clamp smoothing radius
    actual_radius = min(smoothing_radius, n // 2)
    if actual_radius < smoothing_radius:
        logger.warning(
            "Smoothing radius clamped from %d to %d (only %d frames)",
            smoothing_radius, actual_radius, n,
        )

    # Build per-frame deltas (frame 0 = zero)
    dxs = np.array([f["dx"] for f in frames], dtype=np.float64)
    dys = np.array([f["dy"] for f in frames], dtype=np.float64)
    das = np.array([f["da"] for f in frames], dtype=np.float64)

    # Cumulative
    cum_dx = np.cumsum(dxs)
    cum_dy = np.cumsum(dys)
    cum_da = np.cumsum(das)

    # Smooth
    smooth_dx = _moving_average_1d(cum_dx, actual_radius)
    smooth_dy = _moving_average_1d(cum_dy, actual_radius)
    smooth_da = _moving_average_1d(cum_da, actual_radius)

    # Correction
    corr_dx = smooth_dx - cum_dx
    corr_dy = smooth_dy - cum_dy
    corr_da = smooth_da - cum_da

    traj_frames = []
    for i, f in enumerate(frames):
        traj_frames.append({
            "frame_index": f["frame_index"],
            "raw_cum_dx": round(float(cum_dx[i]), 6),
            "raw_cum_dy": round(float(cum_dy[i]), 6),
            "raw_cum_da": round(float(cum_da[i]), 8),
            "smooth_cum_dx": round(float(smooth_dx[i]), 6),
            "smooth_cum_dy": round(float(smooth_dy[i]), 6),
            "smooth_cum_da": round(float(smooth_da[i]), 8),
            "corr_dx": round(float(corr_dx[i]), 6),
            "corr_dy": round(float(corr_dy[i]), 6),
            "corr_da": round(float(corr_da[i]), 8),
        })

    return {
        "method": motion_data["method"],
        "smoothing_radius": smoothing_radius,
        "smoothing_radius_actual": actual_radius,
        "frames": traj_frames,
    }


# ---------------------------------------------------------------------------
# Stabilization + comparison
# ---------------------------------------------------------------------------

def _alignment_score(ref_gray: np.ndarray, cand_gray: np.ndarray) -> float:
    """Mean absolute grayscale difference on valid (non-black) pixels."""
    valid = (ref_gray > 5) & (cand_gray > 5)
    if valid.sum() == 0:
        return float("nan")
    diff = np.abs(ref_gray[valid].astype(np.float32) - cand_gray[valid].astype(np.float32))
    return float(diff.mean())


def _center_crop(img: np.ndarray, ratio: float = 0.33) -> np.ndarray:
    """Extract center crop from image."""
    h, w = img.shape[:2]
    cx, cy = int(w * (1 - ratio) / 2), int(h * (1 - ratio) / 2)
    return img[cy:h - cy, cx:w - cx]


def apply_stabilization(
    frame_dir: Path,
    trajectory_data: dict,
    output_dir: Path,
    crop_ratio: float = 0.0,
    comparison_width: int = 960,
) -> dict:
    """Apply affine2d stabilization and produce comparison images.

    Returns dict matching stabilization_report.json schema.
    """
    loaded = _load_frames_sorted(frame_dir)
    if not loaded:
        raise ValueError(f"No frames found in {frame_dir}")

    traj_frames = trajectory_data["frames"]
    # Build lookup by frame_index
    corr_by_idx = {f["frame_index"]: f for f in traj_frames}

    stabilized_dir = output_dir / "stabilized"
    comparison_dir = output_dir / "comparison"
    stabilized_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Reference frame (frame 0) for alignment scoring
    ref_bgr = loaded[0][1]
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY) if ref_bgr is not None else None
    ref_gray_center = _center_crop(ref_gray, 0.33) if ref_gray is not None else None

    report_frames: list[dict] = []

    for i, (fname, bgr) in enumerate(loaded):
        if bgr is None:
            report_frames.append({"frame_index": i, "skipped": True})
            continue

        h, w = bgr.shape[:2]
        corr = corr_by_idx.get(i, {"corr_dx": 0, "corr_dy": 0, "corr_da": 0})
        dx = corr.get("corr_dx", 0.0)
        dy = corr.get("corr_dy", 0.0)
        da = corr.get("corr_da", 0.0)

        # Build correction affine matrix
        center = (w / 2.0, h / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, np.degrees(da), 1.0)
        rot_mat[0, 2] += dx
        rot_mat[1, 2] += dy

        stabilized_bgr = cv2.warpAffine(
            bgr, rot_mat, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Optional crop
        if crop_ratio > 0:
            ch, cw = int(h * crop_ratio / 2), int(w * crop_ratio / 2)
            if ch > 0 and cw > 0:
                stabilized_bgr = cv2.resize(
                    stabilized_bgr[ch:h - ch, cw:w - cw], (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )

        # Save stabilized frame
        cv2.imwrite(str(stabilized_dir / fname), stabilized_bgr)

        # Generate side-by-side comparison (resized)
        raw_resized = cv2.resize(bgr, (comparison_width, int(h * comparison_width / w)))
        stab_resized = cv2.resize(stabilized_bgr, (comparison_width, int(h * comparison_width / w)))
        comparison = np.hstack([raw_resized, stab_resized])
        cv2.imwrite(str(comparison_dir / fname), comparison)

        # Compute alignment scores
        correction_mag = float(np.sqrt(dx * dx + dy * dy + (np.degrees(da)) ** 2))
        raw_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        stab_gray = cv2.cvtColor(stabilized_bgr, cv2.COLOR_BGR2GRAY)

        entry: dict = {
            "frame_index": i,
            "correction_magnitude": round(correction_mag, 4),
        }

        if ref_gray is not None:
            entry["raw_alignment_score"] = round(_alignment_score(ref_gray, raw_gray), 4)
            entry["stabilized_alignment_score"] = round(_alignment_score(ref_gray, stab_gray), 4)

            raw_center = _center_crop(raw_gray, 0.33)
            stab_center = _center_crop(stab_gray, 0.33)
            if ref_gray_center is not None:
                entry["center_crop_raw_score"] = round(
                    _alignment_score(ref_gray_center, raw_center), 4)
                entry["center_crop_stabilized_score"] = round(
                    _alignment_score(ref_gray_center, stab_center), 4)

        # Adjacent frame stability
        if i > 0:
            prev_fname, prev_bgr = loaded[i - 1]
            if prev_bgr is not None:
                prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
                prev_stab_path = stabilized_dir / prev_fname
                if prev_stab_path.exists():
                    prev_stab = cv2.imread(str(prev_stab_path))
                    if prev_stab is not None:
                        prev_stab_gray = cv2.cvtColor(prev_stab, cv2.COLOR_BGR2GRAY)
                        entry["adjacent_raw_score"] = round(
                            _alignment_score(prev_gray, raw_gray), 4)
                        entry["adjacent_stabilized_score"] = round(
                            _alignment_score(prev_stab_gray, stab_gray), 4)

        report_frames.append(entry)

    # Summary
    valid = [f for f in report_frames if not f.get("skipped") and "raw_alignment_score" in f]
    summary: dict = {}
    if valid:
        summary["avg_correction_magnitude"] = round(
            np.mean([f["correction_magnitude"] for f in valid]), 4)
        summary["avg_raw_alignment_score"] = round(
            np.mean([f["raw_alignment_score"] for f in valid]), 4)
        summary["avg_stabilized_alignment_score"] = round(
            np.mean([f["stabilized_alignment_score"] for f in valid]), 4)

        raw_avg = summary["avg_raw_alignment_score"]
        stab_avg = summary["avg_stabilized_alignment_score"]
        summary["improvement_ratio"] = round(
            1.0 - stab_avg / raw_avg if raw_avg > 0 else 0.0, 4)

        adj_raw = [f["adjacent_raw_score"] for f in valid if "adjacent_raw_score" in f]
        adj_stab = [f["adjacent_stabilized_score"] for f in valid if "adjacent_stabilized_score" in f]
        if adj_raw and adj_stab:
            ar, ast = np.mean(adj_raw), np.mean(adj_stab)
            summary["avg_adjacent_improvement"] = round(
                1.0 - ast / ar if ar > 0 else 0.0, 4)

        cc_raw = [f["center_crop_raw_score"] for f in valid if "center_crop_raw_score" in f]
        cc_stab = [f["center_crop_stabilized_score"] for f in valid if "center_crop_stabilized_score" in f]
        if cc_raw and cc_stab:
            cr, cs = np.mean(cc_raw), np.mean(cc_stab)
            summary["avg_center_crop_improvement"] = round(
                1.0 - cs / cr if cr > 0 else 0.0, 4)

    return {
        "method": "affine2d",
        "smoothing_radius": trajectory_data.get("smoothing_radius", 0),
        "smoothing_radius_actual": trajectory_data.get("smoothing_radius_actual", 0),
        "frames": report_frames,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Atomic JSON write
# ---------------------------------------------------------------------------

def _write_json_atomic(path: Path, data: dict) -> None:
    """Atomically write JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.stem, suffix=".json.tmp",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, str(path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
