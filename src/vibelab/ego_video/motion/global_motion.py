"""Classical global motion estimation based on sparse feature tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class MotionEstimate:
    """Per-frame 2D global motion estimate."""

    frame_index: int
    dx: float
    dy: float
    da: float
    inlier_count: int


def estimate_global_motion(
    video_path: str | Path,
    max_corners: int = 200,
    quality_level: float = 0.01,
    min_distance: float = 30.0,
) -> list[MotionEstimate]:
    """Estimate frame-to-frame similarity motion from tracked corners."""

    capture = cv2.VideoCapture(str(video_path))
    ok, prev_frame = capture.read()
    if not ok:
        raise FileNotFoundError(f"Failed to read first frame from {video_path}")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    estimates: list[MotionEstimate] = []
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
        if prev_pts is None or len(prev_pts) < 4:
            estimates.append(MotionEstimate(frame_index, 0.0, 0.0, 0.0, 0))
            prev_gray = curr_gray
            frame_index += 1
            continue

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]

        if len(good_prev) < 4 or len(good_curr) < 4:
            estimates.append(MotionEstimate(frame_index, 0.0, 0.0, 0.0, int(len(good_curr))))
            prev_gray = curr_gray
            frame_index += 1
            continue

        transform, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr)
        if transform is None:
            estimates.append(MotionEstimate(frame_index, 0.0, 0.0, 0.0, 0))
        else:
            dx = float(transform[0, 2])
            dy = float(transform[1, 2])
            da = float(np.arctan2(transform[1, 0], transform[0, 0]))
            inlier_count = int(inliers.sum()) if inliers is not None else int(len(good_curr))
            estimates.append(MotionEstimate(frame_index, dx, dy, da, inlier_count))

        prev_gray = curr_gray
        frame_index += 1

    capture.release()
    return estimates
