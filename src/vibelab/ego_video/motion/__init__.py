"""Motion estimation baselines for ego-centric video."""

from vibelab.ego_video.motion.diagnostics import (
    FrameCompensationDiagnostic,
    alignment_score,
    build_frame_compensation_diagnostics,
    draw_debug_outline,
    scale_rotation_matrix,
)
from vibelab.ego_video.motion.global_motion import MotionEstimate, estimate_global_motion
from vibelab.ego_video.motion.slam_wrapper import SlamWrapper
from vibelab.ego_video.motion.stabilize import (
    CameraRotationEstimate,
    RotationStabilizationTrace,
    StabilizationTrace,
    compute_rotation_stabilization_trace,
    compute_stabilization_trace,
    estimate_camera_rotations,
    stabilize_frame,
    stabilize_frame_3d_rotation,
    stabilize_video,
    stabilize_video_3d_rotation,
)

__all__ = [
    "CameraRotationEstimate",
    "FrameCompensationDiagnostic",
    "MotionEstimate",
    "RotationStabilizationTrace",
    "SlamWrapper",
    "alignment_score",
    "build_frame_compensation_diagnostics",
    "compute_rotation_stabilization_trace",
    "StabilizationTrace",
    "compute_stabilization_trace",
    "draw_debug_outline",
    "estimate_camera_rotations",
    "estimate_global_motion",
    "scale_rotation_matrix",
    "stabilize_frame_3d_rotation",
    "stabilize_frame",
    "stabilize_video_3d_rotation",
    "stabilize_video",
]
