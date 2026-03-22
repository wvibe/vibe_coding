"""Small, dependency-light video IO utilities built on OpenCV."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class VideoInfo:
    """Basic metadata about a local video file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return 0.0 if self.fps <= 0 else self.frame_count / self.fps


def read_video_info(video_path: str | Path) -> VideoInfo:
    """Read width, height, fps, and frame count from a local video."""

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return VideoInfo(path=path, width=width, height=height, fps=fps, frame_count=frame_count)


def sample_frames(
    video_path: str | Path,
    start_frame: int = 0,
    max_frames: int | None = None,
    stride: int = 1,
) -> list[np.ndarray]:
    """Load a small list of RGB frames into memory."""

    if stride <= 0:
        raise ValueError("stride must be positive")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    frames: list[np.ndarray] = []
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        if (frame_index - start_frame) % stride == 0:
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if max_frames is not None and len(frames) >= max_frames:
                break
        frame_index += 1

    capture.release()
    return frames


def sample_frames_by_index(
    video_path: str | Path,
    frame_indices: list[int],
) -> list[np.ndarray]:
    """Load specific RGB frames by absolute frame index."""

    if not frame_indices:
        return []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    frames: list[np.ndarray] = []
    for frame_index in frame_indices:
        if frame_index < 0:
            raise ValueError("frame indices must be non-negative")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = capture.read()
        if not ok:
            raise ValueError(f"Failed to read frame {frame_index} from {video_path}")
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    capture.release()
    return frames


def undistort_fisheye_frame(
    frame_rgb: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    balance: float = 0.0,
) -> np.ndarray:
    """Undistort a fisheye RGB frame using OpenCV's fisheye model."""

    height, width = frame_rgb.shape[:2]
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = np.array([k1, k2, k3, k4], dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix,
        distortion,
        (width, height),
        identity,
        balance=balance,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        identity,
        new_camera_matrix,
        (width, height),
        cv2.CV_16SC2,
    )
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    undistorted_bgr = cv2.remap(
        frame_bgr,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    start_frame: int = 0,
    max_frames: int | None = None,
    stride: int = 1,
    prefix: str = "frame",
) -> int:
    """Extract frames from a video and save them as PNG files."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for index, frame_rgb in enumerate(
        sample_frames(video_path, start_frame=start_frame, max_frames=max_frames, stride=stride)
    ):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_path = output / f"{prefix}_{index:05d}.png"
        cv2.imwrite(str(frame_path), frame_bgr)
        saved_count += 1
    return saved_count


def export_subclip(
    video_path: str | Path,
    output_path: str | Path,
    start_frame: int = 0,
    num_frames: int = 150,
) -> Path:
    """Export a short contiguous clip using the source video's native geometry."""

    info = read_video_info(video_path)
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fps = info.fps if info.fps > 0 else 30.0
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (info.width, info.height),
    )

    written = 0
    while written < num_frames:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    capture.release()
    writer.release()
    return output


def make_browser_preview(
    video_path: str | Path,
    output_path: str | Path | None = None,
    max_width: int = 960,
    overwrite: bool = False,
) -> Path:
    """Transcode a source video into a browser-friendly H.264 preview with ffmpeg."""

    source = Path(video_path)
    if output_path is None:
        output = source.with_name(f"{source.stem}.browser_preview.mp4")
    else:
        output = Path(output_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        return output

    scale_filter = (
        f"scale='min({max_width},iw)':-2"
        if max_width > 0
        else "scale=iw:ih"
    )
    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-vf",
        scale_filter,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-preset",
        "veryfast",
        str(output),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to generate browser preview.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return output
