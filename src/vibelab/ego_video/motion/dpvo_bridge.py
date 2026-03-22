"""DPVO integration bridge for 6DOF visual odometry stabilization.

Handles: pre-undistortion, DPVO inference (CUDA), pose parsing,
rotation-only correction, and homography-based stabilization.

DPVO inference requires CUDA + lietorch. All other functions (parse,
correct, stabilize, evaluate) run on CPU/Mac without DPVO installed.
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DPVO availability check
# ---------------------------------------------------------------------------

try:
    from dpvo.config import cfg as _dpvo_cfg
    from dpvo.dpvo import DPVO as _DPVO
    DPVO_AVAILABLE = True
except (ImportError, OSError, RuntimeError) as _import_err:
    DPVO_AVAILABLE = False
    _dpvo_cfg = None
    _DPVO = None
    logger.debug("DPVO not available: %s", _import_err)


# ---------------------------------------------------------------------------
# Pre-undistortion (runs on any machine)
# ---------------------------------------------------------------------------

_UNDISTORT_PARAMS = {
    "balance": 0.0,
    "interpolation": "INTER_LINEAR",
    "border_mode": "BORDER_CONSTANT",
}


def _get_fisheye_matrices(calibration: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract camera matrix K and distortion D from calibration dict."""
    fx = float(calibration["fx"])
    fy = float(calibration["fy"])
    cx = float(calibration["cx"])
    cy = float(calibration["cy"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = calibration.get("distortion", {})
    D = np.array([
        float(dist.get("k1", 0)), float(dist.get("k2", 0)),
        float(dist.get("k3", 0)), float(dist.get("k4", 0)),
    ], dtype=np.float64)
    return K, D


def undistort_frames(
    frame_dir: Path,
    output_dir: Path,
    calibration: dict,
) -> tuple[Path, dict]:
    """Undistort fisheye frames to pinhole. Returns (output_dir, pinhole_calibration)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    K, D = _get_fisheye_matrices(calibration)

    # Read one frame to get dimensions
    pngs = sorted(frame_dir.glob("frame_*.png"))
    if not pngs:
        raise ValueError(f"No frames found in {frame_dir}")
    sample = pngs[0]
    img = cv2.imread(str(sample))
    h, w = img.shape[:2]

    # Compute new pinhole intrinsics
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=_UNDISTORT_PARAMS["balance"])

    # Build remap (reuse for all frames)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (w, h), cv2.CV_16SC2)

    pngs = sorted(frame_dir.glob("frame_*.png"))
    for png in pngs:
        img = cv2.imread(str(png))
        if img is None:
            continue
        undistorted = cv2.remap(img, map1, map2,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(str(output_dir / png.name), undistorted)

    pinhole_calib = {
        "fx": float(K_new[0, 0]),
        "fy": float(K_new[1, 1]),
        "cx": float(K_new[0, 2]),
        "cy": float(K_new[1, 2]),
    }
    logger.info("Undistorted %d frames → %s (pinhole: fx=%.1f fy=%.1f)",
                len(pngs), output_dir, pinhole_calib["fx"], pinhole_calib["fy"])
    return output_dir, pinhole_calib


def write_dpvo_calib(calib: dict, output_path: Path) -> None:
    """Write DPVO calibration file (pinhole only, no distortion)."""
    line = f"{calib['fx']:.6f} {calib['fy']:.6f} {calib['cx']:.6f} {calib['cy']:.6f}"
    output_path.write_text(line + "\n", encoding="utf-8")


def resize_frames_for_dpvo(
    frame_dir: Path,
    output_dir: Path,
    calibration: dict,
    scale: float,
) -> tuple[Path, dict]:
    """Resize frames for DPVO inference and scale pinhole intrinsics to match."""
    if scale <= 0:
        raise ValueError(f"dpvo scale must be > 0, got {scale}")
    if abs(scale - 1.0) < 1e-8:
        return frame_dir, calibration

    output_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted(frame_dir.glob("frame_*.png"))
    if not pngs:
        raise ValueError(f"No frames found in {frame_dir}")

    for png in pngs:
        img = cv2.imread(str(png))
        if img is None:
            continue
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output_dir / png.name), resized)

    scaled_calib = {
        "fx": float(calibration["fx"]) * scale,
        "fy": float(calibration["fy"]) * scale,
        "cx": float(calibration["cx"]) * scale,
        "cy": float(calibration["cy"]) * scale,
    }
    logger.info(
        "Resized %d frames for DPVO → %s (scale=%.3f, fx=%.1f fy=%.1f)",
        len(pngs),
        output_dir,
        scale,
        scaled_calib["fx"],
        scaled_calib["fy"],
    )
    return output_dir, scaled_calib


# ---------------------------------------------------------------------------
# DPVO inference (requires CUDA)
# ---------------------------------------------------------------------------

def run_dpvo_inference(
    image_dir: Path,
    calib_path: Path,
    model_path: str = "dpvo.pth",
    config_path: str = "config/default.yaml",
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Run DPVO inference. Returns (poses Nx7, timestamps N).

    Raises ImportError if DPVO is not installed.
    """
    if not DPVO_AVAILABLE:
        raise ImportError(
            "DPVO not installed. Required: CUDA machine with lietorch. "
            "Install: git clone https://github.com/princeton-vl/DPVO.git --recursive && "
            "conda env create -f environment.yml && pip install ."
        )

    import torch
    from multiprocessing import Process, Queue
    from dpvo.stream import image_stream

    cfg = _dpvo_cfg.clone()
    if Path(config_path).exists():
        cfg.merge_from_file(config_path)

    queue: Queue = Queue(maxsize=8)
    reader = Process(target=image_stream,
                     args=(queue, str(image_dir), str(calib_path), stride, 0))
    reader.start()

    slam = None
    while True:
        t, image, intrinsics = queue.get()
        if t < 0:
            break
        image = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()
        if slam is None:
            _, H, W = image.shape
            slam = _DPVO(cfg, model_path, ht=H, wd=W, viz=False)
        slam(t, image, intrinsics)

    reader.join()
    poses, tstamps = slam.terminate()
    return poses, tstamps


# ---------------------------------------------------------------------------
# TUM trajectory parsing (runs on any machine)
# ---------------------------------------------------------------------------

def parse_tum_trajectory(trajectory_path: Path) -> list[dict]:
    """Parse TUM format trajectory file into pose dicts."""
    poses = []
    for line in trajectory_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        ts, tx, ty, tz, qx, qy, qz, qw = [float(x) for x in parts[:8]]
        R = _quaternion_to_rotation(qx, qy, qz, qw)
        poses.append({
            "timestamp": ts,
            "position": [tx, ty, tz],
            "quaternion": [qx, qy, qz, qw],
            "rotation_matrix": R.tolist(),
        })
    return poses


def _quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix."""
    # Normalize
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-10:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n

    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def poses_array_to_tum_file(poses: np.ndarray, tstamps: np.ndarray, output_path: Path) -> None:
    """Write Nx7 poses + timestamps to TUM format file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# timestamp tx ty tz qx qy qz qw"]
    for i in range(len(tstamps)):
        ts = float(tstamps[i])
        tx, ty, tz = float(poses[i, 0]), float(poses[i, 1]), float(poses[i, 2])
        qx, qy, qz, qw = float(poses[i, 3]), float(poses[i, 4]), float(poses[i, 5]), float(poses[i, 6])
        lines.append(f"{ts:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Trajectory alignment
# ---------------------------------------------------------------------------

def align_trajectory_to_frames(
    poses: list[dict],
    manifest_fs: dict | None,
    source_fps: float,
    num_frames: int,
    tolerance_factor: float = 0.5,
) -> list[dict | None]:
    """Align poses to frame indices by timestamp. Returns list of length num_frames."""
    if not poses:
        return [None] * num_frames

    # Build expected timestamps
    if manifest_fs and manifest_fs.get("frames"):
        expected_ts_abs = np.array(
            [f.get("timestamp_sec", i / source_fps) for i, f in enumerate(manifest_fs["frames"])],
            dtype=np.float64,
        )
        expected_ts = expected_ts_abs - expected_ts_abs[0]
    else:
        expected_ts = np.array([i / source_fps for i in range(num_frames)], dtype=np.float64)

    tolerance = tolerance_factor / source_fps if source_fps > 0 else 1.0
    pose_ts = np.array([p["timestamp"] for p in poses], dtype=np.float64)

    # DPVO's image_stream uses enumerate(...) timestamps (0, 1, 2, ...),
    # not seconds. Normalize those to seconds so they match the frame-set
    # manifest timestamps, which are relative to the sampled fps.
    if len(pose_ts) > 1 and source_fps > 0:
        pose_deltas = np.diff(pose_ts)
        expected_deltas = np.diff(expected_ts) if len(expected_ts) > 1 else np.array([], dtype=np.float64)
        pose_step = float(np.median(pose_deltas)) if len(pose_deltas) else 0.0
        expected_step = float(np.median(expected_deltas)) if len(expected_deltas) else (1.0 / source_fps)
        if abs(pose_step - 1.0) < 0.25 and expected_step < 0.75:
            pose_ts = pose_ts / source_fps
            logger.info(
                "Normalized DPVO timestamps from frame indices to seconds using fps=%.3f",
                source_fps,
            )

    aligned: list[dict | None] = []

    for ts in expected_ts:
        diffs = np.abs(pose_ts - ts)
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] <= tolerance:
            aligned.append(poses[best_idx])
        else:
            aligned.append(None)

    matched = sum(1 for a in aligned if a is not None)
    if matched < num_frames * 0.5:
        raise ValueError(
            f"Trajectory alignment failed: only {matched}/{num_frames} frames matched "
            f"(tolerance={tolerance:.4f}s). Check timestamps or stride."
        )

    # Quaternion sign continuity
    prev_q = None
    for a in aligned:
        if a is None:
            continue
        q = np.array(a["quaternion"])
        if prev_q is not None and np.dot(q, prev_q) < 0:
            a["quaternion"] = (-q).tolist()
            a["rotation_matrix"] = _quaternion_to_rotation(*a["quaternion"]).tolist()
        prev_q = np.array(a["quaternion"])

    # Interpolate missing frames (SLERP for quaternion, linear for position)
    aligned = _interpolate_missing(aligned)

    logger.info("Trajectory aligned: %d/%d frames matched, %d interpolated",
                matched, num_frames, num_frames - matched)
    return aligned


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions."""
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        # Very close — use linear interpolation
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    a = math.sin((1 - t) * theta) / sin_theta
    b = math.sin(t * theta) / sin_theta
    result = a * q0 + b * q1
    return result / np.linalg.norm(result)


def _interpolate_missing(aligned: list[dict | None]) -> list[dict | None]:
    """Fill None entries by interpolating between nearest valid neighbors."""
    n = len(aligned)
    result = list(aligned)

    for i in range(n):
        if result[i] is not None:
            continue
        # Find nearest valid before and after
        prev_idx = next((j for j in range(i - 1, -1, -1) if result[j] is not None), None)
        next_idx = next((j for j in range(i + 1, n) if aligned[j] is not None), None)

        if prev_idx is not None and next_idx is not None:
            t = (i - prev_idx) / (next_idx - prev_idx)
            p0 = result[prev_idx]
            p1 = aligned[next_idx]
            q_interp = _slerp(np.array(p0["quaternion"]), np.array(p1["quaternion"]), t)
            pos_interp = [(1 - t) * p0["position"][k] + t * p1["position"][k] for k in range(3)]
            ts_interp = (1 - t) * p0["timestamp"] + t * p1["timestamp"]
            R_interp = _quaternion_to_rotation(*q_interp.tolist())
            result[i] = {
                "timestamp": ts_interp,
                "position": pos_interp,
                "quaternion": q_interp.tolist(),
                "rotation_matrix": R_interp.tolist(),
                "interpolated": True,
            }
        elif prev_idx is not None:
            result[i] = {**result[prev_idx], "interpolated": True}
        elif next_idx is not None:
            result[i] = {**aligned[next_idx], "interpolated": True}

    return result


# ---------------------------------------------------------------------------
# Pose → rotation-only corrections
# ---------------------------------------------------------------------------

def _rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    rv, _ = cv2.Rodrigues(R.astype(np.float64))
    return rv.reshape(3)


def _matrix_from_rotvec(rv: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rv.astype(np.float64).reshape(3, 1))
    return R


def _moving_avg(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(values) < 2:
        return values.copy()
    k = radius * 2 + 1
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(k) / k
    return np.vstack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(values.shape[1])]).T


def poses_to_corrections(
    aligned_poses: list[dict | None],
    calibration_pinhole: dict,
    smoothing_radius: int = 5,
    max_rotation_deg: float = 30.0,
) -> list[dict]:
    """Convert aligned 6DOF poses to per-frame rotation-only H_corr.

    Returns list of dicts with H_corr_3x3, rotvec_raw, rotvec_smooth, confidence.
    """
    n = len(aligned_poses)
    K = np.array([
        [calibration_pinhole["fx"], 0, calibration_pinhole["cx"]],
        [0, calibration_pinhole["fy"], calibration_pinhole["cy"]],
        [0, 0, 1],
    ], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    # Extract rotations (relative to frame 0)
    R0 = None
    raw_rotations: list[np.ndarray | None] = []
    for pose in aligned_poses:
        if pose is None:
            raw_rotations.append(None)
            continue
        R = np.array(pose["rotation_matrix"], dtype=np.float64)
        if R0 is None:
            R0 = R.copy()
        # Relative to frame 0
        R_rel = R @ np.linalg.inv(R0)
        raw_rotations.append(R_rel)

    # Convert to rotvecs, fill missing with interpolation
    rotvecs = np.zeros((n, 3), dtype=np.float64)
    valid_mask = np.zeros(n, dtype=bool)
    for i, R in enumerate(raw_rotations):
        if R is not None:
            rotvecs[i] = _rotvec_from_matrix(R)
            valid_mask[i] = True

    # Simple fill: forward-fill then back-fill for missing
    for i in range(1, n):
        if not valid_mask[i]:
            rotvecs[i] = rotvecs[i - 1]
    for i in range(n - 2, -1, -1):
        if not valid_mask[i]:
            rotvecs[i] = rotvecs[i + 1]

    # Ensure rotvec continuity (use quaternion sign already fixed in alignment,
    # then ensure rotvec doesn't jump near ±π by checking consecutive magnitude)
    for i in range(1, n):
        if valid_mask[i]:
            diff = np.linalg.norm(rotvecs[i] - rotvecs[i - 1])
            if diff > math.pi:
                # Likely a sign flip artifact — negate rotvec
                rotvecs[i] = -rotvecs[i]

    # Smooth
    actual_radius = min(smoothing_radius, n // 2)
    if actual_radius < smoothing_radius:
        logger.warning("Smoothing radius clamped %d→%d (%d frames)", smoothing_radius, actual_radius, n)
    smooth_rotvecs = _moving_avg(rotvecs, actual_radius)

    # Compute corrections
    results: list[dict] = []
    for i in range(n):
        R_raw = _matrix_from_rotvec(rotvecs[i])
        R_smooth = _matrix_from_rotvec(smooth_rotvecs[i])
        R_corr = R_smooth @ R_raw.T
        H_corr = K @ R_corr @ K_inv

        # Confidence
        rot_magnitude = float(np.linalg.norm(rotvecs[i]))
        rot_deg = math.degrees(rot_magnitude)
        if aligned_poses[i] is None:
            confidence = "interpolated"
        elif rot_deg > max_rotation_deg:
            confidence = "high_rotation"
        else:
            confidence = "normal"

        results.append({
            "frame_index": i,
            "H_corr_3x3": [[round(float(v), 8) for v in row] for row in H_corr.tolist()],
            "rotvec_raw": [round(float(v), 8) for v in rotvecs[i]],
            "rotvec_smooth": [round(float(v), 8) for v in smooth_rotvecs[i]],
            "rotation_magnitude_deg": round(rot_deg, 4),
            "confidence": confidence,
        })

    return results


# ---------------------------------------------------------------------------
# Full estimation pipeline
# ---------------------------------------------------------------------------

def estimate_dpvo_motion(
    frame_dir: Path,
    calibration: dict,
    output_dir: Path,
    manifest_fs: dict | None = None,
    source_fps: float = 30.0,
    smoothing_radius: int = 5,
    dpvo_model_path: str = "dpvo.pth",
    dpvo_config: str = "config/default.yaml",
    stride: int = 1,
    image_scale: float = 0.5,
) -> dict:
    """Full DPVO estimation pipeline. Returns motion.json dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    num_frames = len(list(frame_dir.glob("frame_*.png")))

    # Step 1: Pre-undistort
    undist_dir = output_dir / "_undistorted"
    is_fisheye = calibration.get("model", "").lower() == "fisheye" or calibration.get("distortion")
    if is_fisheye:
        undist_dir, pinhole_calib = undistort_frames(frame_dir, undist_dir, calibration)
    else:
        undist_dir = frame_dir
        pinhole_calib = {k: calibration[k] for k in ("fx", "fy", "cx", "cy")}

    # Step 1b: Resize to reduce DPVO memory footprint.
    resized_dir = output_dir / "_dpvo_input"
    dpvo_input_dir, dpvo_calib = resize_frames_for_dpvo(
        undist_dir,
        resized_dir,
        pinhole_calib,
        image_scale,
    )

    # Step 2: Write DPVO calib
    calib_path = output_dir / "calib.txt"
    write_dpvo_calib(dpvo_calib, calib_path)

    # Step 3: Run DPVO
    logger.info(
        "Running DPVO inference on %d frames (stride=%d, image_scale=%.3f)",
        num_frames,
        stride,
        image_scale,
    )
    poses_array, tstamps = run_dpvo_inference(
        image_dir=dpvo_input_dir,
        calib_path=calib_path,
        model_path=dpvo_model_path,
        config_path=dpvo_config,
        stride=stride,
    )

    # Save TUM trajectory
    tum_path = output_dir / "trajectory.txt"
    poses_array_to_tum_file(poses_array, tstamps, tum_path)
    logger.info("Saved TUM trajectory: %s (%d poses)", tum_path, len(tstamps))

    # Step 4: Parse poses
    poses = parse_tum_trajectory(tum_path)

    # Step 5: Align to frames
    aligned = align_trajectory_to_frames(
        poses, manifest_fs, source_fps, num_frames)

    # Step 6: Compute corrections
    corrections = poses_to_corrections(
        aligned, pinhole_calib, smoothing_radius=smoothing_radius)

    # Build motion.json
    frames_meta = []
    for i, corr in enumerate(corrections):
        entry = {**corr}
        if manifest_fs and manifest_fs.get("frames") and i < len(manifest_fs["frames"]):
            mf = manifest_fs["frames"][i]
            entry["source_frame"] = mf.get("source_frame")
            entry["timestamp_sec"] = mf.get("timestamp_sec")
        if aligned[i] is not None:
            entry["position"] = aligned[i]["position"]
            entry["quaternion"] = aligned[i]["quaternion"]
        frames_meta.append(entry)

    motion_data = {
        "method": "dpvo",
        "model": dpvo_model_path,
        "stride": stride,
        "image_scale": image_scale,
        "source_fps": source_fps,
        "num_frames": num_frames,
        "num_poses": len(poses),
        "num_aligned": sum(1 for a in aligned if a is not None),
        "rotation_only": True,
        "calibration_pinhole": pinhole_calib,
        "calibration_dpvo": dpvo_calib,
        "undistort_params": _UNDISTORT_PARAMS if is_fisheye else None,
        "frames": frames_meta,
    }
    return motion_data


# ---------------------------------------------------------------------------
# Trajectory JSON (for cross-machine handoff)
# ---------------------------------------------------------------------------

def build_trajectory_json(
    corrections: list[dict],
    smoothing_radius: int,
    actual_radius: int,
    pinhole_calib: dict,
    is_fisheye: bool,
) -> dict:
    """Build trajectory.json from corrections for stabilize step."""
    return {
        "schema_version": "dpvo_v1",
        "method": "dpvo",
        "smoothing_radius": smoothing_radius,
        "smoothing_radius_actual": actual_radius,
        "rotation_only": True,
        "calibration_pinhole": pinhole_calib,
        "undistort_params": _UNDISTORT_PARAMS if is_fisheye else None,
        "frames": corrections,
    }


# ---------------------------------------------------------------------------
# Stabilization (runs on any machine from trajectory.json)
# ---------------------------------------------------------------------------

def validate_trajectory_schema(trajectory_data: dict) -> list[str]:
    """Validate trajectory JSON schema. Returns list of warnings (empty = valid)."""
    warnings = []
    if "schema_version" not in trajectory_data:
        warnings.append("Missing schema_version field")
    if "method" not in trajectory_data:
        warnings.append("Missing method field")
    frames = trajectory_data.get("frames")
    if not frames or not isinstance(frames, list):
        warnings.append("Missing or empty frames list")
        return warnings

    fallback_count = 0
    for i, f in enumerate(frames):
        if "frame_index" not in f and not f.get("skipped"):
            warnings.append(f"Frame {i}: missing frame_index")
        h_corr = f.get("H_corr_3x3")
        if h_corr is None and not f.get("skipped"):
            fallback_count += 1
        elif h_corr is not None:
            if not (isinstance(h_corr, list) and len(h_corr) == 3
                    and all(isinstance(row, list) and len(row) == 3 for row in h_corr)):
                warnings.append(f"Frame {i}: H_corr_3x3 is not 3x3")

    if fallback_count > 0:
        warnings.append(f"{fallback_count} frames missing H_corr_3x3 (will use identity)")

    return warnings


def apply_dpvo_stabilization(
    frame_dir: Path,
    trajectory_data: dict,
    output_dir: Path,
    comparison_width: int = 960,
) -> dict:
    """Apply DPVO rotation-only corrections via warpPerspective.

    Works from trajectory.json — no DPVO installation needed.
    """
    # Validate schema
    schema_warnings = validate_trajectory_schema(trajectory_data)
    for w in schema_warnings:
        logger.warning("Trajectory schema: %s", w)

    working_frame_dir = frame_dir
    calibration_pinhole = trajectory_data.get("calibration_pinhole")
    undistort_params = trajectory_data.get("undistort_params")

    if undistort_params is not None and calibration_pinhole is not None:
        undist_dir = output_dir / "_undistorted_for_stab"
        if undist_dir.is_dir():
            working_frame_dir = undist_dir
        else:
            manifest_path = frame_dir.parent / "manifest.json"
            fisheye_calib = None
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    fisheye_calib = manifest.get("calibration")
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Failed to read %s: %s", manifest_path, exc)

            if fisheye_calib:
                undistort_frames(frame_dir, undist_dir, fisheye_calib)
                working_frame_dir = undist_dir
            else:
                logger.warning(
                    "No fisheye calibration found for %s, applying DPVO correction to raw frames",
                    frame_dir,
                )

    # Reuse RAFT stabilization logic (same warpPerspective + metrics),
    # but operate in undistorted pinhole space when possible.
    from vibelab.ego_video.motion.raft_flow import apply_raft_stabilization
    return apply_raft_stabilization(
        frame_dir=working_frame_dir,
        trajectory_data=trajectory_data,
        output_dir=output_dir,
        comparison_width=comparison_width,
    )


# ---------------------------------------------------------------------------
# Atomic JSON write
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
