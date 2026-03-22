"""RAFT dense optical flow + adaptive homography fitting for frame-based stabilization.

Uses torchvision's pre-trained RAFT models to compute per-pixel flow,
then fits a global homography via RANSAC (with affine/identity fallback).
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _select_device(requested: str = "auto") -> torch.device:
    """Select best available device: auto → mps → cpu."""
    if requested == "auto":
        if torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon) device")
            return torch.device("mps")
        logger.info("MPS not available, using CPU")
        return torch.device("cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------------
# RAFT model loading
# ---------------------------------------------------------------------------

_RAFT_CACHE: dict[str, torch.nn.Module] = {}


def get_raft_model(
    name: str = "raft_small", device: torch.device | None = None,
) -> tuple[torch.nn.Module, torch.device]:
    """Load and cache a RAFT model."""
    if device is None:
        device = _select_device("auto")

    cache_key = f"{name}_{device}"
    if cache_key not in _RAFT_CACHE:
        logger.info("Loading RAFT model: %s (first run may download weights)", name)
        if name == "raft_small":
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
            model = raft_small(weights=Raft_Small_Weights.DEFAULT)
        elif name == "raft_large":
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
            model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        else:
            raise ValueError(f"Unknown RAFT model: {name}")
        model = model.to(device).eval()
        _RAFT_CACHE[cache_key] = model
    return _RAFT_CACHE[cache_key], device


# ---------------------------------------------------------------------------
# RAFT inference
# ---------------------------------------------------------------------------

def _to_raft_tensor(
    img_rgb: np.ndarray, device: torch.device, max_dim: int | None = 512,
) -> tuple[torch.Tensor, int, int]:
    """Convert RGB image to RAFT input tensor. Returns (tensor, new_w, new_h)."""
    h, w = img_rgb.shape[:2]
    if max_dim is not None and max(h, w) > max_dim:
        ratio = max_dim / max(h, w)
        new_h = int(h * ratio) // 8 * 8
        new_w = int(w * ratio) // 8 * 8
    else:
        new_h = h // 8 * 8
        new_w = w // 8 * 8

    if new_h != h or new_w != w:
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor.to(device), new_w, new_h


def estimate_raft_flow(
    prev_rgb: np.ndarray,
    curr_rgb: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    flow_resize: int | None = 512,
) -> np.ndarray:
    """Run RAFT inference, return (H, W, 2) flow in original resolution."""
    orig_h, orig_w = prev_rgb.shape[:2]

    prev_t, inf_w, inf_h = _to_raft_tensor(prev_rgb, device, flow_resize)
    curr_t, _, _ = _to_raft_tensor(curr_rgb, device, flow_resize)

    with torch.no_grad():
        try:
            flow_preds = model(prev_t, curr_t)
        except RuntimeError as e:
            if "mps" in str(device):
                logger.warning("MPS inference failed (%s), retrying on CPU", e)
                cpu = torch.device("cpu")
                model_cpu = model.to(cpu)
                flow_preds = model_cpu(prev_t.to(cpu), curr_t.to(cpu))
                model.to(device)  # move back
            else:
                raise

    # Take last (most refined) prediction
    flow = flow_preds[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()  # (inf_h, inf_w, 2)

    # Scale flow back to original resolution using exact per-axis scale
    if inf_w != orig_w or inf_h != orig_h:
        scale_x = orig_w / inf_w
        scale_y = orig_h / inf_h
        flow = cv2.resize(flow, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        flow[:, :, 0] *= scale_x
        flow[:, :, 1] *= scale_y

    return flow.astype(np.float32)


# ---------------------------------------------------------------------------
# Adaptive model fitting from dense flow
# ---------------------------------------------------------------------------

def _compute_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute image gradient magnitude (Sobel)."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _quadrant_coverage(inlier_mask: np.ndarray, min_fraction: float = 0.10) -> int:
    """Count how many of 4 quadrants have >= min_fraction of total inliers."""
    h, w = inlier_mask.shape[:2]
    total = inlier_mask.sum()
    if total == 0:
        return 0
    mh, mw = h // 2, w // 2
    quads = [
        inlier_mask[:mh, :mw],
        inlier_mask[:mh, mw:],
        inlier_mask[mh:, :mw],
        inlier_mask[mh:, mw:],
    ]
    return sum(1 for q in quads if q.sum() >= total * min_fraction)


def fit_transform_from_flow(
    flow: np.ndarray,
    prev_gray: np.ndarray | None = None,
    ransac_threshold: float = 3.0,
    min_inlier_ratio: float = 0.3,
    max_condition_number: float = 1e6,
    sample_stride: int = 8,
    gradient_threshold: float = 2.0,
) -> dict:
    """Fit best transform model to a dense flow field with adaptive fallback.

    Returns dict with transform_3x3, model_type, and diagnostics.
    """
    h, w = flow.shape[:2]

    # Sample grid points
    ys = np.arange(0, h, sample_stride)
    xs = np.arange(0, w, sample_stride)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    src_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)

    # Gradient filter (exclude textureless regions)
    if prev_gray is not None and gradient_threshold > 0:
        grad_mag = _compute_gradient_magnitude(prev_gray)
        grad_sampled = grad_mag[grid_y.ravel(), grid_x.ravel()]
        mask = grad_sampled >= gradient_threshold
        if mask.sum() < 20:
            mask = np.ones(len(src_pts), dtype=bool)  # fallback: use all
        src_pts = src_pts[mask]

    if len(src_pts) < 10:
        return _identity_result(h, w, "degenerate", "insufficient_points")

    # Compute destination points from flow
    flow_at_pts = flow[
        src_pts[:, 1].astype(int).clip(0, h - 1),
        src_pts[:, 0].astype(int).clip(0, w - 1),
    ]
    dst_pts = src_pts + flow_at_pts

    # Flow statistics
    flow_mags = np.linalg.norm(flow_at_pts, axis=1)
    flow_mean = float(np.mean(flow_mags))
    flow_p90 = float(np.percentile(flow_mags, 90))

    # Try homography first
    H, inlier_mask_H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)

    if H is not None and inlier_mask_H is not None:
        inlier_mask_flat = inlier_mask_H.ravel().astype(bool)
        inlier_count = int(inlier_mask_flat.sum())
        inlier_ratio = inlier_count / len(src_pts)
        det_H = abs(np.linalg.det(H))
        cond = float(np.linalg.cond(H)) if det_H > 1e-10 else float("inf")

        # Reprojection error
        src_h = np.hstack([src_pts[inlier_mask_flat], np.ones((inlier_count, 1))]).T
        projected = H @ src_h
        projected = projected[:2] / (projected[2:3] + 1e-12)
        reproj_err = float(np.mean(np.linalg.norm(
            projected.T - dst_pts[inlier_mask_flat], axis=1)))

        # Spatial coverage
        inlier_img = np.zeros((h, w), dtype=bool)
        inlier_pts = src_pts[inlier_mask_flat].astype(int)
        inlier_pts[:, 0] = np.clip(inlier_pts[:, 0], 0, w - 1)
        inlier_pts[:, 1] = np.clip(inlier_pts[:, 1], 0, h - 1)
        inlier_img[inlier_pts[:, 1], inlier_pts[:, 0]] = True
        n_quads = _quadrant_coverage(inlier_img)

        # Determine confidence
        confidence = _assign_confidence(
            inlier_ratio, cond, reproj_err, ransac_threshold, det_H, n_quads)

        if confidence != "degenerate":
            return {
                "transform_3x3": H.tolist(),
                "model_type": "homography",
                "inlier_count": inlier_count,
                "inlier_ratio": round(inlier_ratio, 4),
                "background_ratio": round(inlier_ratio, 4),
                "flow_magnitude_mean": round(flow_mean, 4),
                "flow_magnitude_p90": round(flow_p90, 4),
                "reprojection_error": round(reproj_err, 4),
                "condition_number": round(cond, 2),
                "confidence": confidence,
            }

    # Fallback: affine
    transform_aff, inliers_aff = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if transform_aff is not None and inliers_aff is not None:
        inlier_count = int(inliers_aff.sum())
        inlier_ratio = inlier_count / len(src_pts)
        H_aff = np.eye(3, dtype=np.float64)
        H_aff[:2, :] = transform_aff
        confidence = "low_inliers" if inlier_ratio >= 0.2 else "degenerate"

        if confidence != "degenerate":
            return {
                "transform_3x3": H_aff.tolist(),
                "model_type": "affine",
                "inlier_count": inlier_count,
                "inlier_ratio": round(inlier_ratio, 4),
                "background_ratio": round(inlier_ratio, 4),
                "flow_magnitude_mean": round(flow_mean, 4),
                "flow_magnitude_p90": round(flow_p90, 4),
                "reprojection_error": 0.0,
                "condition_number": 1.0,
                "confidence": confidence,
            }

    return _identity_result(h, w, "degenerate", "all_models_failed",
                            flow_mean=flow_mean, flow_p90=flow_p90)


def _assign_confidence(
    inlier_ratio: float, cond: float, reproj_err: float,
    ransac_thresh: float, det_H: float, n_quads: int,
) -> str:
    """Deterministic confidence assignment."""
    if (inlier_ratio < 0.2 or cond >= 1e6 or det_H < 1e-6
            or reproj_err > 2.0 * ransac_thresh):
        return "degenerate"
    base = "normal"
    if inlier_ratio < 0.5 or cond >= 1e4 or reproj_err > ransac_thresh:
        base = "low_inliers"
    # Spatial coverage downgrade
    if n_quads < 3:
        if base == "normal":
            base = "low_inliers"
        elif base == "low_inliers":
            base = "degenerate"
    return base


def _identity_result(h: int, w: int, confidence: str, reason: str, **kwargs) -> dict:
    return {
        "transform_3x3": np.eye(3).tolist(),
        "model_type": "identity",
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "background_ratio": 0.0,
        "flow_magnitude_mean": kwargs.get("flow_mean", 0.0),
        "flow_magnitude_p90": kwargs.get("flow_p90", 0.0),
        "reprojection_error": 0.0,
        "condition_number": 1.0,
        "confidence": confidence,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Full estimation pipeline
# ---------------------------------------------------------------------------

def _load_frames_sorted(frame_dir: Path) -> list[tuple[str, np.ndarray | None]]:
    pngs = sorted(frame_dir.glob("frame_*.png"))
    results = []
    for p in pngs:
        img = cv2.imread(str(p))
        if img is None:
            logger.warning("Skipping unreadable: %s", p.name)
            results.append((p.name, None))
        else:
            results.append((p.name, img))
    return results


def estimate_raft_motion(
    frame_dir: Path,
    model_name: str = "raft_small",
    device_str: str = "auto",
    flow_resize: int | None = 512,
    ransac_threshold: float = 3.0,
    manifest_fs: dict | None = None,
) -> dict:
    """Estimate motion for all frame pairs using RAFT + adaptive model fitting."""
    loaded = _load_frames_sorted(frame_dir)
    if not loaded:
        raise ValueError(f"No frames in {frame_dir}")

    model, device = get_raft_model(model_name, _select_device(device_str))

    frames_meta: list[dict] = []

    # Frame 0: identity
    entry0: dict = {
        "frame_index": 0,
        "transform_3x3": np.eye(3).tolist(),
        "model_type": "identity",
        "inlier_count": 0, "inlier_ratio": 0.0,
        "background_ratio": 1.0,
        "flow_magnitude_mean": 0.0, "flow_magnitude_p90": 0.0,
        "reprojection_error": 0.0, "condition_number": 1.0,
        "confidence": "normal",
    }
    if manifest_fs and manifest_fs.get("frames"):
        mf = manifest_fs["frames"][0]
        entry0["source_frame"] = mf.get("source_frame")
        entry0["timestamp_sec"] = mf.get("timestamp_sec")
    frames_meta.append(entry0)

    prev_bgr = loaded[0][1]
    for i in range(1, len(loaded)):
        fname, curr_bgr = loaded[i]
        if curr_bgr is None or prev_bgr is None:
            entry = {
                "frame_index": i, "skipped": True,
                "reason": "corrupt_or_missing",
            }
            frames_meta.append(entry)
            prev_bgr = curr_bgr
            continue

        prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
        curr_rgb = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)
        prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)

        flow = estimate_raft_flow(prev_rgb, curr_rgb, model, device, flow_resize)
        result = fit_transform_from_flow(
            flow, prev_gray=prev_gray, ransac_threshold=ransac_threshold)

        result["frame_index"] = i
        if manifest_fs and manifest_fs.get("frames") and i < len(manifest_fs["frames"]):
            mf = manifest_fs["frames"][i]
            result["source_frame"] = mf.get("source_frame")
            result["timestamp_sec"] = mf.get("timestamp_sec")

        frames_meta.append(result)
        prev_bgr = curr_bgr

        logger.info(
            "Frame %d: %s, inlier=%.2f, flow_mean=%.1f, confidence=%s",
            i, result.get("model_type", "?"),
            result.get("inlier_ratio", 0), result.get("flow_magnitude_mean", 0),
            result.get("confidence", "?"),
        )

    source_fps = manifest_fs.get("target_fps", 0) if manifest_fs else 0
    actual_fps = manifest_fs.get("actual_fps", source_fps) if manifest_fs else source_fps

    return {
        "method": "raft",
        "model": model_name,
        "device": str(device),
        "flow_resize": flow_resize,
        "source_fps": source_fps,
        "actual_fps": actual_fps,
        "num_frames": len(loaded),
        "runtime": {
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
        },
        "frames": frames_meta,
    }


# ---------------------------------------------------------------------------
# Cumulative transforms
# ---------------------------------------------------------------------------

def compute_cumulative_transforms_raft(motion_data: dict) -> dict:
    """Chain-multiply per-frame transforms into cumulative 3×3 matrices."""
    transforms = []
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
            transforms.append({"frame_index": 0, "matrix_3x3": np.eye(3).tolist()})
            continue

        rel = np.array(frame["transform_3x3"], dtype=np.float64)
        cumulative = rel @ cumulative
        transforms.append({
            "frame_index": frame["frame_index"],
            "matrix_3x3": [[round(float(v), 8) for v in row] for row in cumulative.tolist()],
        })

    return {
        "method": "raft",
        "transform_convention": {
            "direction": "frame_0_to_frame_n",
            "composition_order": "right_multiply",
            "point_convention": "column_vector",
            "definition": "T_0->n = T_{n-1->n} @ T_0->{n-1}",
        },
        "transforms": transforms,
    }


# ---------------------------------------------------------------------------
# Trajectory smoothing (SE(2) approximation)
# ---------------------------------------------------------------------------

def _moving_avg(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(values) < 2:
        return values.copy()
    k = radius * 2 + 1
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(padded, kernel, mode="valid")


def _decompose_homography_se2(H: np.ndarray) -> tuple[float, float, float]:
    """Decompose 3×3 homography into (tx, ty, angle)."""
    return float(H[0, 2]), float(H[1, 2]), float(math.atan2(H[1, 0], H[0, 0]))


def _reconstruct_se2(tx: float, ty: float, angle: float) -> np.ndarray:
    """Reconstruct 3×3 from SE(2) params (perspective components = 0)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1]], dtype=np.float64)


def compute_homography_trajectory(
    cumulative_transforms: dict,
    smoothing_radius: int = 5,
) -> dict:
    """Compute smoothed trajectory + per-frame H_corr from cumulative homographies."""
    transforms = cumulative_transforms["transforms"]
    valid = [t for t in transforms if not t.get("skipped")]
    n = len(valid)

    if n < 2:
        frames = []
        if n == 1:
            frames.append({
                "frame_index": 0,
                "raw_tx": 0, "raw_ty": 0, "raw_angle": 0,
                "smooth_tx": 0, "smooth_ty": 0, "smooth_angle": 0,
                "H_corr_3x3": np.eye(3).tolist(),
            })
        return {
            "method": "raft",
            "smoothing_radius": smoothing_radius,
            "smoothing_radius_actual": 0,
            "frames": frames,
        }

    actual_radius = min(smoothing_radius, n // 2)
    if actual_radius < smoothing_radius:
        logger.warning("Smoothing radius clamped %d→%d (%d frames)", smoothing_radius, actual_radius, n)

    # Decompose cumulative homographies
    txs, tys, angles = [], [], []
    for t in valid:
        H = np.array(t["matrix_3x3"], dtype=np.float64)
        tx, ty, angle = _decompose_homography_se2(H)
        txs.append(tx)
        tys.append(ty)
        angles.append(angle)

    txs = np.array(txs)
    tys = np.array(tys)
    angles = np.unwrap(np.array(angles))  # unwrap to avoid ±π jumps

    # Smooth
    s_tx = _moving_avg(txs, actual_radius)
    s_ty = _moving_avg(tys, actual_radius)
    s_angle = _moving_avg(angles, actual_radius)

    # Compute corrections
    traj_frames = []
    for idx, t in enumerate(valid):
        H_cum = np.array(t["matrix_3x3"], dtype=np.float64)
        H_smooth = _reconstruct_se2(float(s_tx[idx]), float(s_ty[idx]), float(s_angle[idx]))

        det = np.linalg.det(H_cum)
        if abs(det) < 1e-10:
            H_corr = np.eye(3, dtype=np.float64)
        else:
            H_corr = H_smooth @ np.linalg.inv(H_cum)

        traj_frames.append({
            "frame_index": t["frame_index"],
            "raw_tx": round(float(txs[idx]), 6),
            "raw_ty": round(float(tys[idx]), 6),
            "raw_angle": round(float(angles[idx]), 8),
            "smooth_tx": round(float(s_tx[idx]), 6),
            "smooth_ty": round(float(s_ty[idx]), 6),
            "smooth_angle": round(float(s_angle[idx]), 8),
            "H_corr_3x3": [[round(float(v), 8) for v in row] for row in H_corr.tolist()],
        })

    # Add skipped frames with identity correction
    skipped_indices = {t["frame_index"] for t in transforms if t.get("skipped")}
    for si in skipped_indices:
        traj_frames.append({
            "frame_index": si,
            "skipped": True,
            "H_corr_3x3": np.eye(3).tolist(),
        })
    traj_frames.sort(key=lambda x: x["frame_index"])

    return {
        "method": "raft",
        "smoothing_radius": smoothing_radius,
        "smoothing_radius_actual": actual_radius,
        "frames": traj_frames,
    }


# ---------------------------------------------------------------------------
# Stabilization
# ---------------------------------------------------------------------------

def _alignment_score(ref_gray: np.ndarray, cand_gray: np.ndarray) -> float:
    valid = (ref_gray > 5) & (cand_gray > 5)
    if valid.sum() == 0:
        return float("nan")
    return float(np.abs(
        ref_gray[valid].astype(np.float32) - cand_gray[valid].astype(np.float32)
    ).mean())


def _center_crop(img: np.ndarray, ratio: float = 0.33) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = int(w * (1 - ratio) / 2), int(h * (1 - ratio) / 2)
    return img[cy:h - cy, cx:w - cx]


def _border_validity(img_bgr: np.ndarray) -> float:
    """Fraction of non-black pixels."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float((gray > 2).sum()) / gray.size


def apply_raft_stabilization(
    frame_dir: Path,
    trajectory_data: dict,
    output_dir: Path,
    comparison_width: int = 960,
) -> dict:
    """Apply homography-based stabilization with warpPerspective."""
    loaded = _load_frames_sorted(frame_dir)
    if not loaded:
        raise ValueError(f"No frames in {frame_dir}")

    corr_by_idx = {f["frame_index"]: f for f in trajectory_data["frames"]}

    stab_dir = output_dir / "stabilized"
    comp_dir = output_dir / "comparison"
    stab_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    ref_bgr = loaded[0][1]
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY) if ref_bgr is not None else None
    ref_gray_center = _center_crop(ref_gray, 0.33) if ref_gray is not None else None

    report_frames: list[dict] = []

    for i, (fname, bgr) in enumerate(loaded):
        if bgr is None:
            report_frames.append({"frame_index": i, "skipped": True})
            continue

        h, w = bgr.shape[:2]
        corr = corr_by_idx.get(i, {})
        H_corr = np.array(corr.get("H_corr_3x3", np.eye(3).tolist()), dtype=np.float64)

        stabilized = cv2.warpPerspective(
            bgr, H_corr, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        cv2.imwrite(str(stab_dir / fname), stabilized)

        # Comparison
        rh = int(h * comparison_width / w)
        raw_r = cv2.resize(bgr, (comparison_width, rh))
        stab_r = cv2.resize(stabilized, (comparison_width, rh))
        cv2.imwrite(str(comp_dir / fname), np.hstack([raw_r, stab_r]))

        # Metrics
        raw_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        stab_gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
        corr_mag = float(np.linalg.norm(H_corr[:2, 2]))  # translation magnitude

        entry: dict = {
            "frame_index": i,
            "correction_magnitude": round(corr_mag, 4),
            "border_validity_ratio": round(_border_validity(stabilized), 4),
        }

        if ref_gray is not None:
            entry["raw_alignment_score"] = round(_alignment_score(ref_gray, raw_gray), 4)
            entry["stabilized_alignment_score"] = round(_alignment_score(ref_gray, stab_gray), 4)
            if ref_gray_center is not None:
                entry["center_crop_raw_score"] = round(
                    _alignment_score(ref_gray_center, _center_crop(raw_gray, 0.33)), 4)
                entry["center_crop_stabilized_score"] = round(
                    _alignment_score(ref_gray_center, _center_crop(stab_gray, 0.33)), 4)

        if i > 0 and loaded[i - 1][1] is not None:
            prev_gray = cv2.cvtColor(loaded[i - 1][1], cv2.COLOR_BGR2GRAY)
            prev_stab_path = stab_dir / loaded[i - 1][0]
            if prev_stab_path.exists():
                prev_stab = cv2.imread(str(prev_stab_path))
                if prev_stab is not None:
                    entry["adjacent_raw_score"] = round(
                        _alignment_score(prev_gray, raw_gray), 4)
                    entry["adjacent_stabilized_score"] = round(
                        _alignment_score(
                            cv2.cvtColor(prev_stab, cv2.COLOR_BGR2GRAY), stab_gray), 4)

        report_frames.append(entry)

    # Summary
    valid = [f for f in report_frames if not f.get("skipped") and "raw_alignment_score" in f]
    summary: dict = {}
    if valid:
        summary["avg_correction_magnitude"] = round(np.mean([f["correction_magnitude"] for f in valid]), 4)
        summary["avg_border_validity"] = round(np.mean([f["border_validity_ratio"] for f in valid]), 4)
        raw_avg = np.mean([f["raw_alignment_score"] for f in valid])
        stab_avg = np.mean([f["stabilized_alignment_score"] for f in valid])
        summary["avg_raw_alignment_score"] = round(float(raw_avg), 4)
        summary["avg_stabilized_alignment_score"] = round(float(stab_avg), 4)
        summary["improvement_ratio"] = round(float(1.0 - stab_avg / raw_avg) if raw_avg > 0 else 0, 4)

        cc_raw = [f["center_crop_raw_score"] for f in valid if "center_crop_raw_score" in f]
        cc_stab = [f["center_crop_stabilized_score"] for f in valid if "center_crop_stabilized_score" in f]
        if cc_raw and cc_stab:
            cr, cs = np.mean(cc_raw), np.mean(cc_stab)
            summary["avg_center_crop_improvement"] = round(float(1.0 - cs / cr) if cr > 0 else 0, 4)

        adj_raw = [f["adjacent_raw_score"] for f in valid if "adjacent_raw_score" in f]
        adj_stab = [f["adjacent_stabilized_score"] for f in valid if "adjacent_stabilized_score" in f]
        if adj_raw and adj_stab:
            ar, ast_ = np.mean(adj_raw), np.mean(adj_stab)
            summary["avg_adjacent_improvement"] = round(float(1.0 - ast_ / ar) if ar > 0 else 0, 4)

    return {
        "method": "raft",
        "smoothing_radius": trajectory_data.get("smoothing_radius", 0),
        "smoothing_radius_actual": trajectory_data.get("smoothing_radius_actual", 0),
        "frames": report_frames,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Atomic JSON write (shared utility)
# ---------------------------------------------------------------------------

def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
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
