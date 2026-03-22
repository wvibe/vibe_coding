"""Simple moving-average video stabilization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vibelab.ego_video.io.video import read_video_info
from vibelab.ego_video.motion.global_motion import MotionEstimate, estimate_global_motion


@dataclass(slots=True)
class StabilizationTrace:
    """Frame-aligned stabilization diagnostics derived from global motion."""

    frame_indices: np.ndarray
    trajectory: np.ndarray
    smoothed_trajectory: np.ndarray
    correction: np.ndarray


@dataclass(slots=True)
class CameraRotationEstimate:
    """Per-frame relative camera rotation estimate."""

    frame_index: int
    rotation_matrix: np.ndarray
    inlier_count: int


@dataclass(slots=True)
class RotationStabilizationTrace:
    """Frame-aligned 3D camera rotation stabilization diagnostics."""

    frame_indices: np.ndarray
    cumulative_rotvecs: np.ndarray
    smoothed_rotvecs: np.ndarray
    correction_matrices: list[np.ndarray]


def _moving_average(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values
    kernel_size = radius * 2 + 1
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    smoothed = np.vstack(
        [np.convolve(padded[:, column], kernel, mode="valid") for column in range(values.shape[1])]
    ).T
    return smoothed


def _build_correction_matrix(
    dx: float,
    dy: float,
    da: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Build a warp that applies the pose correction around the image center."""

    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, np.degrees(da), 1.0)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return matrix.astype(np.float32)


def _crop_and_resize(frame: np.ndarray, crop_ratio: float) -> np.ndarray:
    """Crop a small border and resize back to suppress edge artifacts."""

    if crop_ratio <= 0.0:
        return frame

    height, width = frame.shape[:2]
    crop_x = int(width * crop_ratio / 2.0)
    crop_y = int(height * crop_ratio / 2.0)
    if crop_x <= 0 or crop_y <= 0 or crop_x * 2 >= width or crop_y * 2 >= height:
        return frame

    cropped = frame[crop_y : height - crop_y, crop_x : width - crop_x]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def _draw_source_outline(
    frame_bgr: np.ndarray,
    outline_color_bgr: tuple[int, int, int],
    outline_thickness: int,
    inset_ratio: float = 0.12,
) -> np.ndarray:
    """Draw an inset source-frame marker before warping for debugging."""

    if outline_thickness <= 0:
        return frame_bgr

    outlined = frame_bgr.copy()
    height, width = outlined.shape[:2]
    inset_x = max(int(width * inset_ratio), outline_thickness)
    inset_y = max(int(height * inset_ratio), outline_thickness)
    top_left = (inset_x, inset_y)
    bottom_right = (max(width - 1 - inset_x, inset_x), max(height - 1 - inset_y, inset_y))
    cv2.rectangle(outlined, top_left, bottom_right, outline_color_bgr, thickness=outline_thickness)
    marker_radius = max(outline_thickness, 6)
    for point in (
        top_left,
        (bottom_right[0], top_left[1]),
        bottom_right,
        (top_left[0], bottom_right[1]),
    ):
        cv2.circle(outlined, point, marker_radius, outline_color_bgr, thickness=-1)
    return outlined


def _camera_model(calibration: dict[str, Any] | Any) -> str:
    if isinstance(calibration, dict):
        model = calibration.get("model")
    else:
        model = getattr(calibration, "model", None)
    return str(model or "pinhole").lower()


def _camera_matrix_and_distortion(
    calibration: dict[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(calibration, dict):
        fx = calibration["fx"]
        fy = calibration["fy"]
        cx = calibration["cx"]
        cy = calibration["cy"]
        distortion = calibration.get("distortion", {})
    else:
        fx = calibration.fx
        fy = calibration.fy
        cx = calibration.cx
        cy = calibration.cy
        distortion = getattr(calibration, "distortion", {})

    if any(value is None for value in (fx, fy, cx, cy)):
        raise ValueError("Calibration must contain fx, fy, cx, and cy.")

    camera_matrix = np.array(
        [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    distortion_vector = np.array(
        [
            float(distortion.get("k1", 0.0)),
            float(distortion.get("k2", 0.0)),
            float(distortion.get("k3", 0.0)),
            float(distortion.get("k4", 0.0)),
        ],
        dtype=np.float64,
    )
    return camera_matrix, distortion_vector


def _undistort_points(
    points: np.ndarray,
    calibration: dict[str, Any] | Any,
) -> np.ndarray:
    camera_matrix, distortion = _camera_matrix_and_distortion(calibration)
    model = _camera_model(calibration)
    point_array = points.reshape(-1, 1, 2).astype(np.float64)
    if model == "fisheye":
        undistorted = cv2.fisheye.undistortPoints(point_array, camera_matrix, distortion)
    else:
        undistorted = cv2.undistortPoints(point_array, camera_matrix, distortion)
    return undistorted.reshape(-1, 2)


def _rotation_matrix_to_rotvec(rotation_matrix: np.ndarray) -> np.ndarray:
    rotvec, _ = cv2.Rodrigues(rotation_matrix.astype(np.float64))
    return rotvec.reshape(3).astype(np.float32)


def _rotvec_to_rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotation_matrix, _ = cv2.Rodrigues(rotvec.astype(np.float64).reshape(3, 1))
    return rotation_matrix.astype(np.float32)


def _scale_rotation_matrix(rotation_matrix: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return rotation_matrix.astype(np.float32)
    rotvec = _rotation_matrix_to_rotvec(rotation_matrix)
    return _rotvec_to_rotation_matrix(rotvec * float(scale))


def compute_stabilization_trace(
    motions: list[MotionEstimate],
    smoothing_radius: int = 15,
) -> StabilizationTrace:
    """Convert per-frame motion estimates into frame-aligned trajectory corrections."""

    transforms = np.array([[m.dx, m.dy, m.da] for m in motions], dtype=np.float32)
    if transforms.size == 0:
        raise ValueError("No frame transforms were provided.")

    trajectory = np.cumsum(transforms, axis=0)
    smoothed = _moving_average(trajectory, radius=smoothing_radius)
    correction = smoothed - trajectory

    zero = np.zeros((1, 3), dtype=np.float32)
    trajectory_full = np.vstack([zero, trajectory])
    smoothed_full = np.vstack([zero, smoothed])
    correction_full = np.vstack([zero, correction])
    frame_indices = np.arange(len(correction_full), dtype=np.int32)

    return StabilizationTrace(
        frame_indices=frame_indices,
        trajectory=trajectory_full,
        smoothed_trajectory=smoothed_full,
        correction=correction_full,
    )


def estimate_camera_rotations(
    video_path: str | Path,
    calibration: dict[str, Any] | Any,
    max_corners: int = 200,
    quality_level: float = 0.01,
    min_distance: float = 30.0,
    essential_threshold: float = 0.002,
) -> list[CameraRotationEstimate]:
    """Estimate relative camera rotations using normalized correspondences."""

    capture = cv2.VideoCapture(str(video_path))
    ok, prev_frame = capture.read()
    if not ok:
        raise FileNotFoundError(f"Failed to read first frame from {video_path}")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    estimates: list[CameraRotationEstimate] = []
    frame_index = 1

    while True:
        ok, curr_frame = capture.read()
        if not ok:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
        )
        if prev_pts is None or len(prev_pts) < 8:
            estimates.append(CameraRotationEstimate(frame_index, np.eye(3, dtype=np.float32), 0))
            prev_gray = curr_gray
            frame_index += 1
            continue

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]

        if len(good_prev) < 8 or len(good_curr) < 8:
            estimates.append(
                CameraRotationEstimate(
                    frame_index,
                    np.eye(3, dtype=np.float32),
                    int(len(good_curr)),
                )
            )
            prev_gray = curr_gray
            frame_index += 1
            continue

        prev_norm = _undistort_points(good_prev, calibration)
        curr_norm = _undistort_points(good_curr, calibration)
        essential, inliers = cv2.findEssentialMat(
            prev_norm,
            curr_norm,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=essential_threshold,
        )
        if essential is None:
            estimates.append(CameraRotationEstimate(frame_index, np.eye(3, dtype=np.float32), 0))
            prev_gray = curr_gray
            frame_index += 1
            continue

        if essential.shape[0] > 3:
            essential = essential[:3, :]

        _, rotation_matrix, _, pose_mask = cv2.recoverPose(
            essential,
            prev_norm,
            curr_norm,
            focal=1.0,
            pp=(0.0, 0.0),
        )
        inlier_count = int(np.count_nonzero(pose_mask)) if pose_mask is not None else 0
        estimates.append(
            CameraRotationEstimate(
                frame_index=frame_index,
                rotation_matrix=rotation_matrix.astype(np.float32),
                inlier_count=inlier_count,
            )
        )

        prev_gray = curr_gray
        frame_index += 1

    capture.release()
    return estimates


def compute_rotation_stabilization_trace(
    rotations: list[CameraRotationEstimate],
    smoothing_radius: int = 15,
) -> RotationStabilizationTrace:
    """Accumulate, smooth, and convert relative rotations into frame corrections."""

    if not rotations:
        raise ValueError("No relative rotations were provided.")

    cumulative_matrices: list[np.ndarray] = [np.eye(3, dtype=np.float32)]
    current = np.eye(3, dtype=np.float32)
    for estimate in rotations:
        current = estimate.rotation_matrix @ current
        cumulative_matrices.append(current.astype(np.float32))

    cumulative_rotvecs = np.vstack(
        [_rotation_matrix_to_rotvec(matrix) for matrix in cumulative_matrices]
    )
    smoothed_rotvecs = _moving_average(cumulative_rotvecs, radius=smoothing_radius)

    correction_matrices: list[np.ndarray] = []
    for raw_rotvec, smooth_rotvec in zip(cumulative_rotvecs, smoothed_rotvecs, strict=True):
        raw_matrix = _rotvec_to_rotation_matrix(raw_rotvec)
        smooth_matrix = _rotvec_to_rotation_matrix(smooth_rotvec)
        correction_matrices.append((smooth_matrix @ raw_matrix.T).astype(np.float32))

    return RotationStabilizationTrace(
        frame_indices=np.arange(len(correction_matrices), dtype=np.int32),
        cumulative_rotvecs=cumulative_rotvecs,
        smoothed_rotvecs=smoothed_rotvecs,
        correction_matrices=correction_matrices,
    )


def stabilize_frame(
    frame_rgb: np.ndarray,
    dx: float,
    dy: float,
    da: float,
    crop_ratio: float = 0.05,
    border_mode: int = cv2.BORDER_REFLECT,
    border_value: tuple[int, int, int] = (0, 0, 0),
    debug_outline: bool = False,
    outline_color_rgb: tuple[int, int, int] = (255, 0, 255),
    outline_thickness: int = 8,
    outline_inset_ratio: float = 0.12,
) -> np.ndarray:
    """Apply a single stabilization correction to one RGB frame."""

    height, width = frame_rgb.shape[:2]
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if debug_outline:
        outline_color_bgr = (
            int(outline_color_rgb[2]),
            int(outline_color_rgb[1]),
            int(outline_color_rgb[0]),
        )
        frame_bgr = _draw_source_outline(
            frame_bgr,
            outline_color_bgr,
            outline_thickness,
            inset_ratio=outline_inset_ratio,
        )
    transform = _build_correction_matrix(dx, dy, da, width, height)
    stabilized_bgr = cv2.warpAffine(
        frame_bgr,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=border_value,
    )
    stabilized_bgr = _crop_and_resize(stabilized_bgr, crop_ratio=crop_ratio)
    return cv2.cvtColor(stabilized_bgr, cv2.COLOR_BGR2RGB)


def stabilize_frame_3d_rotation(
    frame_rgb: np.ndarray,
    calibration: dict[str, Any] | Any,
    correction_matrix: np.ndarray,
    crop_ratio: float = 0.05,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: tuple[int, int, int] = (0, 0, 0),
    debug_outline: bool = False,
    outline_color_rgb: tuple[int, int, int] = (255, 0, 255),
    outline_thickness: int = 8,
    outline_inset_ratio: float = 0.12,
) -> np.ndarray:
    """Apply a rotation-only 3D camera compensation using camera intrinsics."""

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if debug_outline:
        outline_color_bgr = (
            int(outline_color_rgb[2]),
            int(outline_color_rgb[1]),
            int(outline_color_rgb[0]),
        )
        frame_bgr = _draw_source_outline(
            frame_bgr,
            outline_color_bgr,
            outline_thickness,
            inset_ratio=outline_inset_ratio,
        )

    height, width = frame_bgr.shape[:2]
    camera_matrix, distortion = _camera_matrix_and_distortion(calibration)
    model = _camera_model(calibration)

    if model == "fisheye":
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            distortion,
            correction_matrix.astype(np.float64),
            camera_matrix,
            (width, height),
            cv2.CV_16SC2,
        )
    else:
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix,
            distortion,
            correction_matrix.astype(np.float64),
            camera_matrix,
            (width, height),
            cv2.CV_16SC2,
        )

    stabilized_bgr = cv2.remap(
        frame_bgr,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=border_value,
    )
    stabilized_bgr = _crop_and_resize(stabilized_bgr, crop_ratio=crop_ratio)
    return cv2.cvtColor(stabilized_bgr, cv2.COLOR_BGR2RGB)


def stabilize_video(
    video_path: str | Path,
    output_path: str | Path,
    smoothing_radius: int = 15,
    crop_ratio: float = 0.05,
    correction_sign: float = 1.0,
    border_mode: int = cv2.BORDER_REFLECT,
    border_value: tuple[int, int, int] = (0, 0, 0),
    debug_outline: bool = False,
    outline_color_rgb: tuple[int, int, int] = (255, 0, 255),
    outline_thickness: int = 8,
    outline_inset_ratio: float = 0.12,
) -> Path:
    """Estimate frame motion, smooth the camera path, and warp frames back."""

    motions = estimate_global_motion(video_path)
    info = read_video_info(video_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    trace = compute_stabilization_trace(motions, smoothing_radius=smoothing_radius)

    capture = cv2.VideoCapture(str(video_path))
    ok, first_frame = capture.read()
    if not ok:
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    fps = info.fps if info.fps > 0 else 30.0
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (info.width, info.height),
    )

    frame_idx = 0
    frame = first_frame
    while ok and frame_idx < len(trace.correction):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dx, dy, da = trace.correction[frame_idx]
        stabilized_rgb = stabilize_frame(
            frame_rgb=frame_rgb,
            dx=float(correction_sign * dx),
            dy=float(correction_sign * dy),
            da=float(correction_sign * da),
            crop_ratio=crop_ratio,
            border_mode=border_mode,
            border_value=border_value,
            debug_outline=debug_outline,
            outline_color_rgb=outline_color_rgb,
            outline_thickness=outline_thickness,
            outline_inset_ratio=outline_inset_ratio,
        )
        writer.write(cv2.cvtColor(stabilized_rgb, cv2.COLOR_RGB2BGR))
        ok, frame = capture.read()
        frame_idx += 1

    capture.release()
    writer.release()
    return output


def stabilize_video_3d_rotation(
    video_path: str | Path,
    output_path: str | Path,
    calibration: dict[str, Any] | Any,
    smoothing_radius: int = 15,
    crop_ratio: float = 0.05,
    correction_sign: float = 1.0,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: tuple[int, int, int] = (0, 0, 0),
    debug_outline: bool = False,
    outline_color_rgb: tuple[int, int, int] = (255, 0, 255),
    outline_thickness: int = 8,
    outline_inset_ratio: float = 0.12,
) -> Path:
    """Stabilize a video by estimating and smoothing the accumulated 3D camera rotation."""

    rotations = estimate_camera_rotations(video_path, calibration=calibration)
    trace = compute_rotation_stabilization_trace(rotations, smoothing_radius=smoothing_radius)
    info = read_video_info(video_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    ok, first_frame = capture.read()
    if not ok:
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    fps = info.fps if info.fps > 0 else 30.0
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (info.width, info.height),
    )

    frame_idx = 0
    frame = first_frame
    while ok and frame_idx < len(trace.correction_matrices):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        correction_matrix = _scale_rotation_matrix(
            trace.correction_matrices[frame_idx],
            scale=correction_sign,
        )
        stabilized_rgb = stabilize_frame_3d_rotation(
            frame_rgb=frame_rgb,
            calibration=calibration,
            correction_matrix=correction_matrix,
            crop_ratio=crop_ratio,
            border_mode=border_mode,
            border_value=border_value,
            debug_outline=debug_outline,
            outline_color_rgb=outline_color_rgb,
            outline_thickness=outline_thickness,
            outline_inset_ratio=outline_inset_ratio,
        )
        writer.write(cv2.cvtColor(stabilized_rgb, cv2.COLOR_RGB2BGR))
        ok, frame = capture.read()
        frame_idx += 1

    capture.release()
    writer.release()
    return output
