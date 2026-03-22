"""Background-aware stabilization evaluation metrics.

Measures stabilization quality by focusing on background temporal consistency,
not frame-0 alignment. Handles foreground exclusion, warp border exclusion,
and illumination robustness via dual-domain (intensity + gradient) scoring.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Region masks
# ---------------------------------------------------------------------------

def _non_border_mask(gray: np.ndarray, threshold: int = 2) -> np.ndarray:
    """Pixels that are not black warp borders."""
    return gray > threshold


def _center_crop_mask(h: int, w: int, ratio: float = 0.5) -> np.ndarray:
    """Inner region mask (ratio=0.5 → inner 50%, exclude outer 25% each side)."""
    mask = np.zeros((h, w), dtype=bool)
    margin_y = int(h * (1 - ratio) / 2)
    margin_x = int(w * (1 - ratio) / 2)
    mask[margin_y:h - margin_y, margin_x:w - margin_x] = True
    return mask


def _low_flow_proxy_mask(
    flow_magnitude: np.ndarray | None, percentile: float = 75.0,
) -> np.ndarray | None:
    """Exclude high-flow (foreground) pixels. Returns None if no flow data."""
    if flow_magnitude is None:
        return None
    threshold = np.percentile(flow_magnitude, percentile)
    return flow_magnitude <= threshold


def build_valid_mask(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    center_crop_ratio: float = 0.5,
    flow_magnitude: np.ndarray | None = None,
) -> np.ndarray:
    """Build pairwise valid region mask (MVP tier)."""
    h, w = prev_gray.shape[:2]
    mask = (_non_border_mask(prev_gray) & _non_border_mask(curr_gray)
            & _center_crop_mask(h, w, center_crop_ratio))
    flow_mask = _low_flow_proxy_mask(flow_magnitude)
    if flow_mask is not None:
        mask = mask & flow_mask
    return mask


# ---------------------------------------------------------------------------
# M1: Adjacent Frame Background Stability
# ---------------------------------------------------------------------------

def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def adjacent_background_stability(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    valid_mask: np.ndarray,
) -> dict:
    """Dual-domain (intensity + gradient) stability between adjacent frames.

    Returns dict with intensity_mae, gradient_mae, combined_score.
    All scores: lower = more stable.
    """
    valid = valid_mask & (prev_gray > 2) & (curr_gray > 2)
    n_valid = int(valid.sum())

    if n_valid < 100:
        return {"intensity_mae": float("nan"), "gradient_mae": float("nan"),
                "combined_score": float("nan"), "valid_pixels": 0}

    # Intensity domain
    intensity_diff = np.abs(
        prev_gray[valid].astype(np.float32) - curr_gray[valid].astype(np.float32))
    intensity_mae = float(intensity_diff.mean())

    # Gradient domain
    prev_grad = _sobel_magnitude(prev_gray)
    curr_grad = _sobel_magnitude(curr_gray)
    grad_diff = np.abs(prev_grad[valid] - curr_grad[valid])
    gradient_mae = float(grad_diff.mean())

    # Combined (normalized)
    max_grad = max(float(prev_grad[valid].max()), float(curr_grad[valid].max()), 1.0)
    combined = 0.5 * (intensity_mae / 255.0) + 0.5 * (gradient_mae / max_grad)

    return {
        "intensity_mae": round(intensity_mae, 4),
        "gradient_mae": round(gradient_mae, 4),
        "combined_score": round(combined, 6),
        "valid_pixels": n_valid,
    }


# ---------------------------------------------------------------------------
# M2: Residual Background Motion Magnitude
# ---------------------------------------------------------------------------

def _quadrant_coverage(points: np.ndarray, h: int, w: int) -> int:
    """Count 2×2 quadrants with >= 10% of points."""
    if len(points) == 0:
        return 0
    mh, mw = h // 2, w // 2
    total = len(points)
    quads = [
        ((points[:, 0] < mw) & (points[:, 1] < mh)).sum(),
        ((points[:, 0] >= mw) & (points[:, 1] < mh)).sum(),
        ((points[:, 0] < mw) & (points[:, 1] >= mh)).sum(),
        ((points[:, 0] >= mw) & (points[:, 1] >= mh)).sum(),
    ]
    return sum(1 for q in quads if q >= total * 0.10)


def residual_background_motion(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    center_crop_ratio: float = 0.5,
    max_corners: int = 200,
    min_inlier_ratio: float = 0.3,
    min_quadrant_coverage: int = 3,
) -> dict:
    """Estimate residual global motion in center crop of frame pair.

    Returns dict with magnitude, median_displacement, reliable flag.
    """
    h, w = prev_frame.shape[:2]
    margin_y = int(h * (1 - center_crop_ratio) / 2)
    margin_x = int(w * (1 - center_crop_ratio) / 2)

    prev_crop = prev_frame[margin_y:h - margin_y, margin_x:w - margin_x]
    curr_crop = curr_frame[margin_y:h - margin_y, margin_x:w - margin_x]

    if prev_crop.ndim == 3:
        prev_gray = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_crop, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray, curr_gray = prev_crop, curr_crop

    unreliable = {
        "affine_magnitude": float("nan"), "median_displacement": float("nan"),
        "trimmed_mean_displacement": float("nan"),
        "inlier_ratio": 0.0, "quadrant_coverage": 0, "reliable": False,
    }

    ch, cw = prev_gray.shape[:2]
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners,
                                  qualityLevel=0.01, minDistance=15.0)
    if pts is None or len(pts) < 8:
        return unreliable

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None)
    good_prev = pts[status.flatten() == 1]
    good_curr = curr_pts[status.flatten() == 1]

    if len(good_prev) < 8:
        return unreliable

    transform, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr)
    if transform is None or inliers is None:
        return unreliable

    inlier_mask = inliers.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / len(good_prev)

    # Quadrant coverage on inlier points
    inlier_pts = good_prev[inlier_mask].reshape(-1, 2)
    n_quads = _quadrant_coverage(inlier_pts, ch, cw)

    reliable = (inlier_ratio >= min_inlier_ratio and n_quads >= min_quadrant_coverage)

    # Displacements
    displacements = np.linalg.norm(
        good_curr[inlier_mask].reshape(-1, 2) - good_prev[inlier_mask].reshape(-1, 2),
        axis=1)
    median_disp = float(np.median(displacements))

    # Trimmed mean (trim 10% each end)
    sorted_d = np.sort(displacements)
    trim = max(1, len(sorted_d) // 10)
    trimmed = sorted_d[trim:-trim] if len(sorted_d) > 2 * trim else sorted_d
    trimmed_mean = float(trimmed.mean()) if len(trimmed) > 0 else median_disp

    # Affine translation magnitude
    dx, dy = float(transform[0, 2]), float(transform[1, 2])
    magnitude = math.sqrt(dx * dx + dy * dy)

    return {
        "affine_magnitude": round(magnitude, 4),
        "median_displacement": round(median_disp, 4),
        "trimmed_mean_displacement": round(trimmed_mean, 4),
        "inlier_ratio": round(inlier_ratio, 4),
        "quadrant_coverage": n_quads,
        "reliable": reliable,
    }


# ---------------------------------------------------------------------------
# M3: Border Validity
# ---------------------------------------------------------------------------

def border_validity_ratio(frame: np.ndarray) -> float:
    """Fraction of non-black pixels."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return float((gray > 2).sum()) / gray.size


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def _load_frames_sorted(frame_dir: Path) -> list[tuple[str, np.ndarray | None]]:
    pngs = sorted(frame_dir.glob("frame_*.png"))
    return [(p.name, cv2.imread(str(p))) for p in pngs]


def evaluate_stabilization(
    raw_frame_dir: Path,
    stabilized_frame_dir: Path,
    crop_ratios: list[float] | None = None,
    method: str = "unknown",
) -> dict:
    """Run full evaluation comparing raw vs stabilized frame sequences.

    Returns evaluation_report dict.
    """
    if crop_ratios is None:
        crop_ratios = [0.33, 0.5, 0.66]

    raw_frames = _load_frames_sorted(raw_frame_dir)
    stab_frames = _load_frames_sorted(stabilized_frame_dir)

    if len(raw_frames) != len(stab_frames):
        logger.warning("Frame count mismatch: raw=%d, stab=%d", len(raw_frames), len(stab_frames))

    n = min(len(raw_frames), len(stab_frames))
    per_frame: list[dict] = []

    for i in range(n):
        raw_name, raw_bgr = raw_frames[i]
        stab_name, stab_bgr = stab_frames[i]

        entry: dict = {"frame_index": i}

        if raw_bgr is None or stab_bgr is None:
            entry["skipped"] = True
            per_frame.append(entry)
            continue

        # Border validity
        bv = border_validity_ratio(stab_bgr)
        entry["border_validity_ratio"] = round(bv, 4)
        entry["is_valid"] = bv >= 0.80  # permissive threshold for inclusion

        # Adjacent stability (i > 0)
        if i > 0:
            prev_raw_bgr = raw_frames[i - 1][1]
            prev_stab_bgr = stab_frames[i - 1][1]

            if prev_raw_bgr is not None and prev_stab_bgr is not None:
                raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
                stab_gray = cv2.cvtColor(stab_bgr, cv2.COLOR_BGR2GRAY)
                prev_raw_gray = cv2.cvtColor(prev_raw_bgr, cv2.COLOR_BGR2GRAY)
                prev_stab_gray = cv2.cvtColor(prev_stab_bgr, cv2.COLOR_BGR2GRAY)

                for cr in crop_ratios:
                    cr_key = f"stability_{cr}"

                    raw_mask = build_valid_mask(prev_raw_gray, raw_gray, cr)
                    stab_mask = build_valid_mask(prev_stab_gray, stab_gray, cr)

                    raw_score = adjacent_background_stability(
                        prev_raw_gray, raw_gray, raw_mask)
                    stab_score = adjacent_background_stability(
                        prev_stab_gray, stab_gray, stab_mask)

                    raw_c = raw_score["combined_score"]
                    stab_c = stab_score["combined_score"]
                    eps = 1e-6
                    if math.isnan(raw_c) or raw_c < eps:
                        improvement = float("nan")
                    else:
                        improvement = 1.0 - stab_c / raw_c

                    entry[cr_key] = {
                        "raw_intensity_mae": raw_score["intensity_mae"],
                        "stab_intensity_mae": stab_score["intensity_mae"],
                        "raw_gradient_mae": raw_score["gradient_mae"],
                        "stab_gradient_mae": stab_score["gradient_mae"],
                        "raw_combined": raw_score["combined_score"],
                        "stab_combined": stab_score["combined_score"],
                        "improvement": round(improvement, 4) if not math.isnan(improvement) else None,
                    }

                # Residual motion (M2)
                raw_motion = residual_background_motion(prev_raw_bgr, raw_bgr)
                stab_motion = residual_background_motion(prev_stab_bgr, stab_bgr)

                raw_mag = raw_motion["affine_magnitude"]
                stab_mag = stab_motion["affine_magnitude"]

                if (raw_motion["reliable"] and stab_motion["reliable"]
                        and not math.isnan(raw_mag) and raw_mag > 1e-6):
                    motion_improvement = 1.0 - stab_mag / raw_mag
                else:
                    motion_improvement = float("nan")

                entry["residual_motion"] = {
                    "raw_magnitude": raw_motion["affine_magnitude"],
                    "stab_magnitude": stab_motion["affine_magnitude"],
                    "raw_median_disp": raw_motion["median_displacement"],
                    "stab_median_disp": stab_motion["median_displacement"],
                    "improvement": round(motion_improvement, 4) if not math.isnan(motion_improvement) else None,
                    "raw_reliable": raw_motion["reliable"],
                    "stab_reliable": stab_motion["reliable"],
                }

        per_frame.append(entry)

    # Summary
    summary = _build_summary(per_frame, crop_ratios)

    return {
        "method": method,
        "metrics_version": "2.0",
        "center_crop_ratios": crop_ratios,
        "primary_crop_ratio": 0.5,
        "per_frame": per_frame,
        "summary": summary,
    }


def _build_summary(per_frame: list[dict], crop_ratios: list[float]) -> dict:
    valid = [f for f in per_frame if not f.get("skipped") and f.get("is_valid", True)]

    # Border stats
    bvs = [f["border_validity_ratio"] for f in valid if "border_validity_ratio" in f]
    border_stats = {}
    if bvs:
        border_stats = {
            "min_validity": round(min(bvs), 4),
            "p5_validity": round(float(np.percentile(bvs, 5)), 4),
            "mean_validity": round(float(np.mean(bvs)), 4),
            "flagged_at_0.80": sum(1 for v in bvs if v < 0.80),
            "flagged_at_0.85": sum(1 for v in bvs if v < 0.85),
            "flagged_at_0.90": sum(1 for v in bvs if v < 0.90),
        }

    summary: dict = {
        "num_valid_frames": len(valid),
        "border_stats": border_stats,
    }

    # Per crop-ratio stability
    for cr in crop_ratios:
        cr_key = f"stability_{cr}"
        entries = [f[cr_key] for f in valid if cr_key in f and f[cr_key].get("improvement") is not None]
        if entries:
            summary[cr_key] = {
                "avg_raw_combined": round(float(np.mean([e["raw_combined"] for e in entries])), 6),
                "avg_stab_combined": round(float(np.mean([e["stab_combined"] for e in entries])), 6),
                "avg_improvement": round(float(np.mean([e["improvement"] for e in entries])), 4),
                "direction": "lower_is_better_for_scores_higher_is_better_for_improvement",
            }

    # Residual motion
    motion_entries = [
        f["residual_motion"] for f in valid
        if "residual_motion" in f
        and f["residual_motion"].get("improvement") is not None
    ]
    if motion_entries:
        summary["residual_motion"] = {
            "avg_raw_magnitude": round(float(np.mean([e["raw_magnitude"] for e in motion_entries])), 4),
            "avg_stab_magnitude": round(float(np.mean([e["stab_magnitude"] for e in motion_entries])), 4),
            "avg_improvement": round(float(np.mean([e["improvement"] for e in motion_entries])), 4),
            "num_reliable": len(motion_entries),
            "direction": "lower_is_better_for_magnitude_higher_for_improvement",
        }

    return summary
