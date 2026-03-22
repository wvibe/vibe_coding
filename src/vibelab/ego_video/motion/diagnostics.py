"""Diagnostics helpers for ego-video motion and stabilization experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from vibelab.ego_video.io.video import read_video_info, sample_frames_by_index
from vibelab.ego_video.motion.global_motion import estimate_global_motion
from vibelab.ego_video.motion.stabilize import (
    compute_rotation_stabilization_trace,
    compute_stabilization_trace,
    estimate_camera_rotations,
    stabilize_frame,
    stabilize_frame_3d_rotation,
)

FrameCompareModel = Literal["affine2d", "rotation3d"]


@dataclass(slots=True)
class FrameCompensationDiagnostic:
    """One row of frame-level stabilization diagnostics."""

    frame_index: int
    time_sec: float
    reference_display: np.ndarray
    raw_display: np.ndarray
    identity_display: np.ndarray
    plus_display: np.ndarray
    minus_display: np.ndarray
    raw_score: float
    identity_score: float
    plus_score: float
    minus_score: float
    plus_label: str
    minus_label: str


def draw_debug_outline(
    frame_rgb: np.ndarray,
    color_rgb: tuple[int, int, int] = (255, 0, 255),
    thickness: int = 24,
    inset_ratio: float = 0.12,
) -> np.ndarray:
    """Draw a thick inset rectangle plus corner markers for warp diagnostics."""

    if thickness <= 0:
        return frame_rgb

    outlined = frame_rgb.copy()
    height, width = outlined.shape[:2]
    inset_x = max(int(width * inset_ratio), thickness)
    inset_y = max(int(height * inset_ratio), thickness)
    top_left = (inset_x, inset_y)
    bottom_right = (max(width - 1 - inset_x, inset_x), max(height - 1 - inset_y, inset_y))
    cv2.rectangle(outlined, top_left, bottom_right, color_rgb, thickness=thickness)
    marker_radius = max(thickness, 6)
    for point in (
        top_left,
        (bottom_right[0], top_left[1]),
        bottom_right,
        (top_left[0], bottom_right[1]),
    ):
        cv2.circle(outlined, point, marker_radius, color_rgb, thickness=-1)
    return outlined


def alignment_score(reference_rgb: np.ndarray, candidate_rgb: np.ndarray) -> float:
    """Compute a simple mean absolute grayscale error on valid pixels only."""

    ref_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
    cand_gray = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2GRAY)
    valid = (ref_gray > 5) & (cand_gray > 5)
    if valid.sum() == 0:
        return float("nan")
    diff = np.abs(ref_gray[valid].astype(np.float32) - cand_gray[valid].astype(np.float32))
    return float(diff.mean())


def scale_rotation_matrix(matrix: np.ndarray, sign: float) -> np.ndarray:
    """Flip or preserve a rotation correction by scaling its Rodrigues vector."""

    rotvec, _ = cv2.Rodrigues(matrix.astype(np.float64))
    scaled_matrix, _ = cv2.Rodrigues(rotvec * float(sign))
    return scaled_matrix.astype(np.float32)


def build_frame_compensation_diagnostics(
    video_path: str | Path,
    frame_indices: list[int],
    reference_frame_index: int = 0,
    model: FrameCompareModel = "rotation3d",
    calibration: dict[str, Any] | Any | None = None,
    smoothing_radius: int = 15,
    crop_ratio: float = 0.0,
    border_mode: int = cv2.BORDER_CONSTANT,
    outline_color_rgb: tuple[int, int, int] = (255, 0, 255),
    outline_thickness: int = 24,
    outline_inset_ratio: float = 0.12,
    max_corners: int = 200,
    quality_level: float = 0.01,
    min_distance: float = 30.0,
) -> list[FrameCompensationDiagnostic]:
    """Build per-frame raw/identity/+corr/-corr views for notebook inspection."""

    if model == "rotation3d" and calibration is None:
        raise ValueError("Calibration is required when model='rotation3d'.")

    info = read_video_info(video_path)
    clean_indices = sorted({int(idx) for idx in frame_indices if 0 <= int(idx) < info.frame_count})
    if not clean_indices:
        return []

    reference_frame = sample_frames_by_index(video_path, [reference_frame_index])[0]
    raw_frames = sample_frames_by_index(video_path, clean_indices)
    reference_display = draw_debug_outline(
        reference_frame,
        color_rgb=outline_color_rgb,
        thickness=outline_thickness,
        inset_ratio=outline_inset_ratio,
    )

    if model == "rotation3d":
        relative_rotations = estimate_camera_rotations(
            video_path,
            calibration=calibration,
            max_corners=max_corners,
            quality_level=quality_level,
            min_distance=min_distance,
        )
        rotation_trace = compute_rotation_stabilization_trace(
            relative_rotations,
            smoothing_radius=smoothing_radius,
        )

        def identity_view(frame_rgb: np.ndarray, with_outline: bool) -> np.ndarray:
            return stabilize_frame_3d_rotation(
                frame_rgb=frame_rgb,
                calibration=calibration,
                correction_matrix=np.eye(3, dtype=np.float32),
                crop_ratio=crop_ratio,
                border_mode=border_mode,
                debug_outline=with_outline,
                outline_color_rgb=outline_color_rgb,
                outline_thickness=outline_thickness,
                outline_inset_ratio=outline_inset_ratio,
            )

        reference_identity = identity_view(reference_frame, with_outline=False)
    else:
        motions = estimate_global_motion(
            video_path,
            max_corners=max_corners,
            quality_level=quality_level,
            min_distance=min_distance,
        )
        trace = compute_stabilization_trace(motions, smoothing_radius=smoothing_radius)
        reference_identity = reference_frame

    diagnostics: list[FrameCompensationDiagnostic] = []
    for raw_frame, frame_index in zip(raw_frames, clean_indices, strict=True):
        raw_display = draw_debug_outline(
            raw_frame,
            color_rgb=outline_color_rgb,
            thickness=outline_thickness,
            inset_ratio=outline_inset_ratio,
        )
        time_sec = frame_index / info.fps if info.fps > 0 else 0.0

        if model == "rotation3d":
            correction_matrix = rotation_trace.correction_matrices[frame_index]
            identity_plain = identity_view(raw_frame, with_outline=False)
            identity_display = identity_view(raw_frame, with_outline=True)

            plus_matrix = scale_rotation_matrix(correction_matrix, +1.0)
            minus_matrix = scale_rotation_matrix(correction_matrix, -1.0)
            plus_plain = stabilize_frame_3d_rotation(
                frame_rgb=raw_frame,
                calibration=calibration,
                correction_matrix=plus_matrix,
                crop_ratio=crop_ratio,
                border_mode=border_mode,
                debug_outline=False,
            )
            plus_display = stabilize_frame_3d_rotation(
                frame_rgb=raw_frame,
                calibration=calibration,
                correction_matrix=plus_matrix,
                crop_ratio=crop_ratio,
                border_mode=border_mode,
                debug_outline=True,
                outline_color_rgb=outline_color_rgb,
                outline_thickness=outline_thickness,
                outline_inset_ratio=outline_inset_ratio,
            )
            minus_plain = stabilize_frame_3d_rotation(
                frame_rgb=raw_frame,
                calibration=calibration,
                correction_matrix=minus_matrix,
                crop_ratio=crop_ratio,
                border_mode=border_mode,
                debug_outline=False,
            )
            minus_display = stabilize_frame_3d_rotation(
                frame_rgb=raw_frame,
                calibration=calibration,
                correction_matrix=minus_matrix,
                crop_ratio=crop_ratio,
                border_mode=border_mode,
                debug_outline=True,
                outline_color_rgb=outline_color_rgb,
                outline_thickness=outline_thickness,
                outline_inset_ratio=outline_inset_ratio,
            )
            plus_rotvec, _ = cv2.Rodrigues(plus_matrix.astype(np.float64))
            minus_rotvec, _ = cv2.Rodrigues(minus_matrix.astype(np.float64))
            plus_deg = np.degrees(plus_rotvec.reshape(3))
            minus_deg = np.degrees(minus_rotvec.reshape(3))
            plus_deg_text = ", ".join(f"{value:.2f}" for value in plus_deg)
            minus_deg_text = ", ".join(f"{value:.2f}" for value in minus_deg)
            plus_label = f"apply (+corr) | rotvec_deg=({plus_deg_text})"
            minus_label = f"apply (-corr) | rotvec_deg=({minus_deg_text})"
        else:
            correction_dx, correction_dy, correction_da = trace.correction[frame_index]
            identity_plain = raw_frame
            identity_display = raw_display
            plus_plain = stabilize_frame(
                frame_rgb=raw_frame,
                dx=float(correction_dx),
                dy=float(correction_dy),
                da=float(correction_da),
                crop_ratio=crop_ratio,
                border_mode=border_mode,
                debug_outline=False,
            )
            plus_display = stabilize_frame(
                frame_rgb=raw_frame,
                dx=float(correction_dx),
                dy=float(correction_dy),
                da=float(correction_da),
                crop_ratio=crop_ratio,
                border_mode=border_mode,
                debug_outline=True,
                outline_color_rgb=outline_color_rgb,
                outline_thickness=outline_thickness,
                outline_inset_ratio=outline_inset_ratio,
            )
            minus_plain = stabilize_frame(
                frame_rgb=raw_frame,
                dx=float(-correction_dx),
                dy=float(-correction_dy),
                da=float(-correction_da),
                crop_ratio=crop_ratio,
                border_mode=border_mode,
                debug_outline=False,
            )
            minus_display = stabilize_frame(
                frame_rgb=raw_frame,
                dx=float(-correction_dx),
                dy=float(-correction_dy),
                da=float(-correction_da),
                crop_ratio=crop_ratio,
                border_mode=border_mode,
                debug_outline=True,
                outline_color_rgb=outline_color_rgb,
                outline_thickness=outline_thickness,
                outline_inset_ratio=outline_inset_ratio,
            )
            plus_label = (
                f"apply (+corr) | dx={correction_dx:.2f}, dy={correction_dy:.2f}, "
                f"dtheta={np.degrees(correction_da):.2f} deg"
            )
            minus_label = (
                f"apply (-corr) | dx={-correction_dx:.2f}, dy={-correction_dy:.2f}, "
                f"dtheta={np.degrees(-correction_da):.2f} deg"
            )

        diagnostics.append(
            FrameCompensationDiagnostic(
                frame_index=frame_index,
                time_sec=time_sec,
                reference_display=reference_display,
                raw_display=raw_display,
                identity_display=identity_display,
                plus_display=plus_display,
                minus_display=minus_display,
                raw_score=alignment_score(reference_frame, raw_frame),
                identity_score=alignment_score(reference_identity, identity_plain),
                plus_score=alignment_score(reference_identity, plus_plain),
                minus_score=alignment_score(reference_identity, minus_plain),
                plus_label=plus_label,
                minus_label=minus_label,
            )
        )

    return diagnostics
