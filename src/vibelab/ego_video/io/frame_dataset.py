"""Multi-rate frame extraction from video files.

Extracts frame sequences at multiple target frame rates from a single source
video, producing structured directories of PNG images with a manifest that
records full provenance (source frame indices, timestamps, calibration).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from vibelab.ego_video.io.video import read_video_info

logger = logging.getLogger(__name__)


def compute_frame_indices(
    start_frame: int,
    source_fps: float,
    target_fps: float,
    num_frames: int,
    total_frames: int,
) -> list[int]:
    """Compute source frame indices for a target fps using time-grid sampling.

    For each output frame *i* (0 .. num_frames-1)::

        target_time = start_time + i / target_fps
        source_frame = int(target_time * source_fps + 0.5)

    This avoids drift from integer stride rounding and handles non-divisible
    fps ratios (e.g., 29.97 source, 10 target).

    Raises:
        ValueError: If *target_fps* <= 0 or > *source_fps*.
    """
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if target_fps > source_fps:
        raise ValueError(
            f"target_fps ({target_fps}) exceeds source_fps ({source_fps}); "
            "upsampling is not supported"
        )
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    start_time = start_frame / source_fps
    indices: list[int] = []
    for i in range(num_frames):
        t = start_time + i / target_fps
        frame_idx = int(t * source_fps + 0.5)  # deterministic half-up rounding
        if frame_idx >= total_frames:
            break
        indices.append(frame_idx)
    return indices


def _fps_dir_name(fps: float) -> str:
    """Return directory name for a target fps (e.g., ``10fps``, ``7.5fps``)."""
    if fps == int(fps):
        return f"{int(fps)}fps"
    return f"{fps}fps"


def _validate_fps_list(fps_list: list[float], source_fps: float) -> None:
    """Validate the target fps list."""
    if not fps_list:
        raise ValueError("target_fps_list must not be empty")
    seen: set[float] = set()
    for fps in fps_list:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if fps > source_fps:
            raise ValueError(
                f"target fps {fps} exceeds source fps {source_fps}; "
                "upsampling is not supported"
            )
        if fps in seen:
            raise ValueError(f"duplicate fps value: {fps}")
        seen.add(fps)


def _compute_derived_stride(indices: list[int]) -> int:
    """Compute the most common frame-index delta (informational)."""
    if len(indices) < 2:
        return 1
    deltas = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
    most_common, _ = Counter(deltas).most_common(1)[0]
    return most_common


def _compute_actual_fps(indices: list[int], source_fps: float) -> float:
    """Compute actual fps from frame indices."""
    if len(indices) < 2:
        return source_fps
    time_span = (indices[-1] - indices[0]) / source_fps
    if time_span <= 0:
        return source_fps
    return (len(indices) - 1) / time_span


def extract_multi_rate_frames(
    video_path: Path,
    output_dir: Path,
    start_sec: float,
    num_frames: int,
    target_fps_list: list[float],
    calibration: dict | None = None,
    source_metadata: dict | None = None,
    strict: bool = True,
) -> Path:
    """Extract frame sequences at multiple rates from a video.

    Args:
        video_path: Source video file.
        output_dir: Output directory for frame sets.
        start_sec: Start time in seconds.
        num_frames: Number of frames to extract per rate.
        target_fps_list: List of target frame rates.
        calibration: Optional fisheye calibration dict.
        source_metadata: Optional source metadata dict.
        strict: If True, raise when requested frames exceed available.
                If False, extract what's available and log warning.

    Returns:
        Path to the written manifest.json.

    Raises:
        FileNotFoundError: If video file does not exist.
        ValueError: On invalid parameters or (strict mode) insufficient frames.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    info = read_video_info(video_path)
    source_fps = info.fps if info.fps > 0 else 30.0

    _validate_fps_list(target_fps_list, source_fps)

    start_frame = int(start_sec * source_fps + 0.5)
    if start_frame >= info.frame_count:
        raise ValueError(
            f"start_sec {start_sec} maps to frame {start_frame}, "
            f"but video only has {info.frame_count} frames"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video once, read all needed frames
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    try:
        frame_sets_manifest: list[dict] = []

        for target_fps in target_fps_list:
            indices = compute_frame_indices(
                start_frame=start_frame,
                source_fps=source_fps,
                target_fps=target_fps,
                num_frames=num_frames,
                total_frames=info.frame_count,
            )

            if len(indices) < num_frames:
                msg = (
                    f"Requested {num_frames} frames at {target_fps}fps but only "
                    f"{len(indices)} available from frame {start_frame}"
                )
                if strict:
                    raise ValueError(msg)
                logger.warning(msg)

            # Create output directory for this rate (clean stale files)
            dir_name = _fps_dir_name(target_fps)
            rate_dir = output_dir / dir_name
            if rate_dir.exists():
                for old_png in rate_dir.glob("frame_*.png"):
                    old_png.unlink()
            rate_dir.mkdir(parents=True, exist_ok=True)

            # Extract and save frames
            frames_meta: list[dict] = []
            for out_idx, src_frame in enumerate(indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, src_frame)
                ok, frame_bgr = capture.read()
                if not ok:
                    msg = f"Failed to read frame {src_frame} from {video_path}"
                    if strict:
                        raise ValueError(msg)
                    logger.warning(msg)
                    break

                filename = f"frame_{out_idx:05d}.png"
                write_ok = cv2.imwrite(str(rate_dir / filename), frame_bgr)
                if not write_ok:
                    raise IOError(
                        f"Failed to write frame to {rate_dir / filename}"
                    )
                frames_meta.append(
                    {
                        "filename": filename,
                        "source_frame": src_frame,
                        "timestamp_sec": round(src_frame / source_fps, 6),
                    }
                )

            actual_count = len(frames_meta)
            stride = _compute_derived_stride(indices[:actual_count])
            actual_fps = _compute_actual_fps(indices[:actual_count], source_fps)
            actual_indices = indices[:actual_count]

            if actual_count >= 2:
                actual_duration = (actual_indices[-1] - actual_indices[0]) / source_fps
            else:
                actual_duration = 0.0

            frame_sets_manifest.append(
                {
                    "target_fps": target_fps,
                    "actual_fps": round(actual_fps, 4),
                    "stride": stride,
                    "requested_num_frames": num_frames,
                    "actual_num_frames": actual_count,
                    "actual_duration_sec": round(actual_duration, 6),
                    "directory": dir_name,
                    "source_frame_indices": actual_indices,
                    "frames": frames_meta,
                }
            )

            logger.info(
                "Extracted %d/%d frames at %sfps → %s",
                actual_count,
                num_frames,
                target_fps,
                rate_dir,
            )

    finally:
        capture.release()

    # Build manifest — prefer relative path, fall back to filename only
    try:
        rel_video = video_path.relative_to(output_dir)
    except ValueError:
        rel_video = Path(video_path.name)

    manifest = {
        "source": {
            "video_path": str(rel_video),
            "video_path_absolute": str(video_path.resolve()),
            **(source_metadata or {}),
        },
        "calibration": calibration,
        "extraction": {
            "start_sec": start_sec,
            "source_fps": source_fps,
            "start_frame": start_frame,
            "requested_num_frames": num_frames,
        },
        "frame_sets": frame_sets_manifest,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Atomic write
    manifest_path = output_dir / "manifest.json"
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(output_dir), prefix="manifest", suffix=".json.tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_path, str(manifest_path))
    except BaseException:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info("Manifest written to %s", manifest_path)
    return manifest_path


def load_frame_set_manifest(manifest_path: Path | str) -> dict:
    """Load and return a frame-set manifest."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_frame_set(manifest_path: Path | str) -> dict:
    """Inspect a frame set directory and return a summary dict.

    Verifies that files on disk match the manifest and reports any
    discrepancies.
    """
    path = Path(manifest_path)
    manifest = load_frame_set_manifest(path)
    base_dir = path.parent

    summary = {
        "manifest_path": str(path),
        "source": manifest.get("source", {}),
        "calibration_present": manifest.get("calibration") is not None,
        "extraction": manifest.get("extraction", {}),
        "frame_sets": [],
        "warnings": [],
    }

    for fs in manifest.get("frame_sets", []):
        dir_path = base_dir / fs["directory"]
        expected_files = {f["filename"] for f in fs.get("frames", [])}
        actual_files = set()
        if dir_path.is_dir():
            actual_files = {f.name for f in dir_path.iterdir() if f.suffix == ".png"}

        missing = expected_files - actual_files
        extra = actual_files - expected_files

        fs_summary = {
            "target_fps": fs["target_fps"],
            "directory": fs["directory"],
            "expected_frames": len(expected_files),
            "actual_frames": len(actual_files),
            "ok": len(missing) == 0 and len(extra) == 0,
        }

        if missing:
            fs_summary["missing"] = sorted(missing)
            summary["warnings"].append(
                f"{fs['directory']}: {len(missing)} missing files"
            )
        if extra:
            fs_summary["extra"] = sorted(extra)
            summary["warnings"].append(
                f"{fs['directory']}: {len(extra)} unexpected files"
            )

        # Read dimensions from first frame if available
        if actual_files:
            first_file = dir_path / sorted(actual_files)[0]
            img = cv2.imread(str(first_file))
            if img is not None:
                h, w = img.shape[:2]
                fs_summary["resolution"] = f"{w}x{h}"

        summary["frame_sets"].append(fs_summary)

    return summary
