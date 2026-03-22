"""Unit tests for vibelab.ego_video.io.frame_dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vibelab.ego_video.io.frame_dataset import (
    _fps_dir_name,
    _validate_fps_list,
    compute_frame_indices,
    extract_multi_rate_frames,
    inspect_frame_set,
    load_frame_set_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_video(path: Path, num_frames: int = 100, fps: float = 30.0) -> Path:
    """Create a small test video with numbered frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 160, 120
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(num_frames):
        frame = np.full((h, w, 3), fill_value=(i * 2) % 256, dtype=np.uint8)
        # Stamp frame number into pixel [0,0] channels for identification
        frame[0, 0, 0] = i % 256
        frame[0, 0, 1] = (i >> 8) % 256
        frame[0, 0, 2] = 0
        writer.write(frame)
    writer.release()
    return path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# compute_frame_indices
# ---------------------------------------------------------------------------

class TestComputeFrameIndices:
    def test_exact_divisor(self) -> None:
        """30fps source, 10fps target → indices spaced by 3."""
        indices = compute_frame_indices(
            start_frame=0, source_fps=30.0, target_fps=10.0,
            num_frames=5, total_frames=1000,
        )
        assert indices == [0, 3, 6, 9, 12]

    def test_exact_divisor_with_offset(self) -> None:
        """With start_frame=300, indices should start at 300."""
        indices = compute_frame_indices(
            start_frame=300, source_fps=30.0, target_fps=10.0,
            num_frames=5, total_frames=1000,
        )
        assert indices == [300, 303, 306, 309, 312]

    def test_non_divisor(self) -> None:
        """29.97fps source, 10fps target → correct time-grid spacing."""
        indices = compute_frame_indices(
            start_frame=0, source_fps=29.97, target_fps=10.0,
            num_frames=5, total_frames=1000,
        )
        # Expected: t=0 → frame 0, t=0.1 → frame 3, t=0.2 → frame 6, etc.
        assert len(indices) == 5
        # Verify spacing is approximately 3 (29.97/10 ≈ 2.997)
        for i in range(1, len(indices)):
            delta = indices[i] - indices[i - 1]
            assert delta in (2, 3, 4), f"Unexpected delta {delta} at index {i}"

    def test_target_exceeds_source_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds source_fps"):
            compute_frame_indices(
                start_frame=0, source_fps=30.0, target_fps=60.0,
                num_frames=5, total_frames=1000,
            )

    def test_target_equals_source(self) -> None:
        """target_fps == source_fps → stride 1."""
        indices = compute_frame_indices(
            start_frame=0, source_fps=30.0, target_fps=30.0,
            num_frames=5, total_frames=1000,
        )
        assert indices == [0, 1, 2, 3, 4]

    def test_zero_target_fps_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_frame_indices(
                start_frame=0, source_fps=30.0, target_fps=0.0,
                num_frames=5, total_frames=1000,
            )

    def test_negative_target_fps_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_frame_indices(
                start_frame=0, source_fps=30.0, target_fps=-5.0,
                num_frames=5, total_frames=1000,
            )

    def test_truncated_by_total_frames(self) -> None:
        """Returns fewer indices when video is shorter than requested."""
        indices = compute_frame_indices(
            start_frame=95, source_fps=30.0, target_fps=10.0,
            num_frames=10, total_frames=100,
        )
        assert len(indices) < 10
        assert all(idx < 100 for idx in indices)

    def test_first_frame_always_start(self) -> None:
        """First index should always be start_frame."""
        for target in [30.0, 10.0, 3.0, 1.0]:
            indices = compute_frame_indices(
                start_frame=42, source_fps=30.0, target_fps=target,
                num_frames=5, total_frames=1000,
            )
            assert indices[0] == 42


# ---------------------------------------------------------------------------
# fps validation
# ---------------------------------------------------------------------------

class TestFpsValidation:
    def test_valid(self) -> None:
        _validate_fps_list([30.0, 10.0, 3.0], source_fps=30.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _validate_fps_list([], source_fps=30.0)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _validate_fps_list([0.0], source_fps=30.0)

    def test_exceeds_source_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            _validate_fps_list([60.0], source_fps=30.0)

    def test_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            _validate_fps_list([10.0, 10.0], source_fps=30.0)


class TestFpsDirName:
    def test_integer(self) -> None:
        assert _fps_dir_name(30.0) == "30fps"
        assert _fps_dir_name(10.0) == "10fps"

    def test_float(self) -> None:
        assert _fps_dir_name(7.5) == "7.5fps"


# ---------------------------------------------------------------------------
# extract_multi_rate_frames (integration-style with synthetic video)
# ---------------------------------------------------------------------------

class TestExtractMultiRateFrames:
    @pytest.fixture()
    def test_video(self, tmp_path: Path) -> Path:
        return _make_test_video(tmp_path / "test.mp4", num_frames=300, fps=30.0)

    def test_basic_extraction(self, test_video: Path, tmp_path: Path) -> None:
        """Extract 10 frames at 30,10,3 fps."""
        out = tmp_path / "output"
        manifest_path = extract_multi_rate_frames(
            video_path=test_video,
            output_dir=out,
            start_sec=0.0,
            num_frames=10,
            target_fps_list=[30.0, 10.0, 3.0],
        )

        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())

        assert len(manifest["frame_sets"]) == 3
        for fs in manifest["frame_sets"]:
            assert fs["actual_num_frames"] == 10
            assert len(fs["frames"]) == 10
            assert len(fs["source_frame_indices"]) == 10

            # Verify files on disk
            rate_dir = out / fs["directory"]
            pngs = list(rate_dir.glob("frame_*.png"))
            assert len(pngs) == 10

    def test_manifest_schema(self, test_video: Path, tmp_path: Path) -> None:
        """Verify manifest has all required fields."""
        out = tmp_path / "output"
        manifest_path = extract_multi_rate_frames(
            video_path=test_video,
            output_dir=out,
            start_sec=1.0,
            num_frames=5,
            target_fps_list=[10.0],
            calibration={"model": "fisheye", "fx": 100.0},
        )
        manifest = json.loads(manifest_path.read_text())

        # Top-level keys
        assert "source" in manifest
        assert "calibration" in manifest
        assert "extraction" in manifest
        assert "frame_sets" in manifest
        assert "created_at" in manifest

        # Extraction
        assert manifest["extraction"]["start_sec"] == 1.0
        assert manifest["extraction"]["requested_num_frames"] == 5

        # Calibration preserved
        assert manifest["calibration"]["fx"] == 100.0

        # Frame set detail
        fs = manifest["frame_sets"][0]
        assert "target_fps" in fs
        assert "actual_fps" in fs
        assert "stride" in fs
        assert "requested_num_frames" in fs
        assert "actual_num_frames" in fs
        assert "source_frame_indices" in fs
        assert "frames" in fs
        assert all("source_frame" in f for f in fs["frames"])
        assert all("timestamp_sec" in f for f in fs["frames"])

    def test_all_rates_same_first_frame(self, test_video: Path, tmp_path: Path) -> None:
        """frame_00000.png should be identical across all rates."""
        out = tmp_path / "output"
        extract_multi_rate_frames(
            video_path=test_video,
            output_dir=out,
            start_sec=0.0,
            num_frames=10,
            target_fps_list=[30.0, 10.0, 3.0],
        )

        hashes = set()
        for rate in ["30fps", "10fps", "3fps"]:
            h = _sha256_file(out / rate / "frame_00000.png")
            hashes.add(h)

        assert len(hashes) == 1, f"First frame differs across rates: {hashes}"

    def test_strict_mode_raises(self, test_video: Path, tmp_path: Path) -> None:
        """Requesting more frames than available should raise in strict mode."""
        out = tmp_path / "output"
        with pytest.raises(ValueError, match="only.*available"):
            extract_multi_rate_frames(
                video_path=test_video,
                output_dir=out,
                start_sec=9.5,  # near end of 10s video
                num_frames=50,
                target_fps_list=[3.0],
                strict=True,
            )

    def test_allow_partial(self, test_video: Path, tmp_path: Path) -> None:
        """Allow-partial should save what's available."""
        out = tmp_path / "output"
        manifest_path = extract_multi_rate_frames(
            video_path=test_video,
            output_dir=out,
            start_sec=9.5,
            num_frames=50,
            target_fps_list=[3.0],
            strict=False,
        )
        manifest = json.loads(manifest_path.read_text())
        fs = manifest["frame_sets"][0]
        assert fs["requested_num_frames"] == 50
        assert fs["actual_num_frames"] < 50
        assert fs["actual_num_frames"] > 0

    def test_null_calibration(self, test_video: Path, tmp_path: Path) -> None:
        """Extraction works without calibration."""
        out = tmp_path / "output"
        manifest_path = extract_multi_rate_frames(
            video_path=test_video,
            output_dir=out,
            start_sec=0.0,
            num_frames=5,
            target_fps_list=[10.0],
            calibration=None,
        )
        manifest = json.loads(manifest_path.read_text())
        assert manifest["calibration"] is None

    def test_manifest_atomic_write(self, test_video: Path, tmp_path: Path) -> None:
        """No .tmp file should remain after successful extraction."""
        out = tmp_path / "output"
        extract_multi_rate_frames(
            video_path=test_video,
            output_dir=out,
            start_sec=0.0,
            num_frames=5,
            target_fps_list=[10.0],
        )
        tmp_files = list(out.glob("*.json.tmp"))
        assert len(tmp_files) == 0

    def test_start_sec_offset(self, test_video: Path, tmp_path: Path) -> None:
        """Frames extracted with start_sec > 0 should start at correct frame."""
        out = tmp_path / "output"
        manifest_path = extract_multi_rate_frames(
            video_path=test_video,
            output_dir=out,
            start_sec=2.0,
            num_frames=5,
            target_fps_list=[30.0],
        )
        manifest = json.loads(manifest_path.read_text())
        fs = manifest["frame_sets"][0]
        # At 30fps, 2.0s = frame 60
        assert fs["source_frame_indices"][0] == 60

    def test_video_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_multi_rate_frames(
                video_path=tmp_path / "nonexistent.mp4",
                output_dir=tmp_path / "output",
                start_sec=0.0,
                num_frames=5,
                target_fps_list=[10.0],
            )


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

class TestInspectFrameSet:
    def test_inspect_valid(self, tmp_path: Path) -> None:
        video = _make_test_video(tmp_path / "test.mp4", num_frames=100, fps=30.0)
        out = tmp_path / "output"
        manifest_path = extract_multi_rate_frames(
            video_path=video,
            output_dir=out,
            start_sec=0.0,
            num_frames=5,
            target_fps_list=[30.0, 10.0],
        )

        summary = inspect_frame_set(manifest_path)
        assert len(summary["frame_sets"]) == 2
        assert all(fs["ok"] for fs in summary["frame_sets"])
        assert len(summary["warnings"]) == 0

    def test_inspect_missing_files(self, tmp_path: Path) -> None:
        video = _make_test_video(tmp_path / "test.mp4", num_frames=100, fps=30.0)
        out = tmp_path / "output"
        manifest_path = extract_multi_rate_frames(
            video_path=video,
            output_dir=out,
            start_sec=0.0,
            num_frames=5,
            target_fps_list=[10.0],
        )

        # Delete a frame file
        (out / "10fps" / "frame_00002.png").unlink()

        summary = inspect_frame_set(manifest_path)
        fs = summary["frame_sets"][0]
        assert not fs["ok"]
        assert "frame_00002.png" in fs["missing"]
