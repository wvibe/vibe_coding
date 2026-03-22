"""Tests for dpvo_bridge.py — Mac-side only (no CUDA/DPVO required)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from vibelab.ego_video.motion.dpvo_bridge import (
    _quaternion_to_rotation,
    align_trajectory_to_frames,
    apply_dpvo_stabilization,
    build_trajectory_json,
    parse_tum_trajectory,
    poses_to_corrections,
    undistort_frames,
    write_dpvo_calib,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_frames(path: Path, n: int = 5) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.full((120, 160, 3), fill_value=(i * 30) % 256, dtype=np.uint8)
        cv2.imwrite(str(path / f"frame_{i:05d}.png"), img)
    return path


def _sample_calibration() -> dict:
    return {
        "model": "fisheye",
        "fx": 500.0, "fy": 500.0, "cx": 80.0, "cy": 60.0,
        "distortion": {"k1": -0.01, "k2": 0.001, "k3": 0.0, "k4": 0.0},
        "image_width": 160, "image_height": 120,
    }


def _pinhole_calib() -> dict:
    return {"fx": 500.0, "fy": 500.0, "cx": 80.0, "cy": 60.0}


def _sample_tum_content() -> str:
    lines = ["# timestamp tx ty tz qx qy qz qw"]
    for i in range(5):
        t = i * 0.033333
        # Small rotation around z-axis
        angle = i * 0.01
        qw = math.cos(angle / 2)
        qz = math.sin(angle / 2)
        lines.append(f"{t:.6f} {i*0.001:.6f} 0.000000 0.000000 0.000000 0.000000 {qz:.6f} {qw:.6f}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Quaternion
# ---------------------------------------------------------------------------

class TestQuaternion:
    def test_identity(self) -> None:
        R = _quaternion_to_rotation(0, 0, 0, 1)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_90_deg_z(self) -> None:
        """90° rotation around z-axis: qw=cos(45°), qz=sin(45°)."""
        a = math.pi / 2
        R = _quaternion_to_rotation(0, 0, math.sin(a / 2), math.cos(a / 2))
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_normalization(self) -> None:
        """Non-unit quaternion should be normalized."""
        R = _quaternion_to_rotation(0, 0, 0, 2.0)  # scale 2
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


# ---------------------------------------------------------------------------
# TUM parsing
# ---------------------------------------------------------------------------

class TestParseTum:
    def test_parse_basic(self, tmp_path: Path) -> None:
        tum_file = tmp_path / "traj.txt"
        tum_file.write_text(_sample_tum_content())
        poses = parse_tum_trajectory(tum_file)
        assert len(poses) == 5
        assert "timestamp" in poses[0]
        assert "position" in poses[0]
        assert "quaternion" in poses[0]
        assert "rotation_matrix" in poses[0]
        assert len(poses[0]["rotation_matrix"]) == 3

    def test_skips_comments(self, tmp_path: Path) -> None:
        content = "# comment\n0.0 0 0 0 0 0 0 1\n# another\n0.1 0 0 0 0 0 0 1\n"
        tum_file = tmp_path / "traj.txt"
        tum_file.write_text(content)
        poses = parse_tum_trajectory(tum_file)
        assert len(poses) == 2


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_write_pinhole(self, tmp_path: Path) -> None:
        calib = _pinhole_calib()
        path = tmp_path / "calib.txt"
        write_dpvo_calib(calib, path)
        content = path.read_text()
        assert "500.000000" in content
        assert "80.000000" in content
        # No k params
        parts = content.strip().split()
        assert len(parts) == 4


# ---------------------------------------------------------------------------
# Undistortion
# ---------------------------------------------------------------------------

class TestUndistort:
    def test_undistort_produces_frames(self, tmp_path: Path) -> None:
        frame_dir = _make_test_frames(tmp_path / "raw", n=3)
        calib = _sample_calibration()
        out_dir = tmp_path / "undistorted"
        out_dir, pinhole = undistort_frames(frame_dir, out_dir, calib)
        assert out_dir.is_dir()
        assert len(list(out_dir.glob("frame_*.png"))) == 3
        assert "fx" in pinhole
        assert "fy" in pinhole


# ---------------------------------------------------------------------------
# Trajectory alignment
# ---------------------------------------------------------------------------

class TestAlignment:
    def test_basic_alignment(self, tmp_path: Path) -> None:
        tum_file = tmp_path / "traj.txt"
        tum_file.write_text(_sample_tum_content())
        poses = parse_tum_trajectory(tum_file)
        aligned = align_trajectory_to_frames(poses, None, 30.0, 5)
        assert len(aligned) == 5
        matched = sum(1 for a in aligned if a is not None)
        assert matched >= 3  # at least 60%

    def test_too_short_raises(self) -> None:
        poses = [{"timestamp": 0.0, "quaternion": [0, 0, 0, 1],
                  "rotation_matrix": np.eye(3).tolist(), "position": [0, 0, 0]}]
        with pytest.raises(ValueError, match="alignment failed"):
            align_trajectory_to_frames(poses, None, 30.0, 10)

    def test_quaternion_sign_continuity(self) -> None:
        """Flipped quaternion should be corrected."""
        poses = [
            {"timestamp": 0.0, "quaternion": [0, 0, 0, 1],
             "rotation_matrix": np.eye(3).tolist(), "position": [0, 0, 0]},
            {"timestamp": 0.033, "quaternion": [0, 0, 0, -1],  # flipped sign
             "rotation_matrix": np.eye(3).tolist(), "position": [0, 0, 0]},
            {"timestamp": 0.066, "quaternion": [0, 0, 0, 1],
             "rotation_matrix": np.eye(3).tolist(), "position": [0, 0, 0]},
        ]
        aligned = align_trajectory_to_frames(poses, None, 30.0, 3)
        # After continuity fix, all qw should have same sign
        qws = [a["quaternion"][3] for a in aligned if a is not None]
        signs = [1 if q >= 0 else -1 for q in qws]
        assert len(set(signs)) == 1  # all same sign


# ---------------------------------------------------------------------------
# Poses to corrections
# ---------------------------------------------------------------------------

class TestCorrections:
    def test_static_gives_identity(self) -> None:
        """Static poses → identity corrections."""
        poses = [
            {"timestamp": i * 0.033, "quaternion": [0, 0, 0, 1],
             "rotation_matrix": np.eye(3).tolist(), "position": [0, 0, 0]}
            for i in range(5)
        ]
        corrections = poses_to_corrections(poses, _pinhole_calib())
        for c in corrections:
            H = np.array(c["H_corr_3x3"])
            np.testing.assert_allclose(H, np.eye(3), atol=1e-6)

    def test_rotation_produces_nonidentity(self) -> None:
        """Rotating poses → non-identity H_corr."""
        poses = []
        for i in range(10):
            angle = i * 0.05  # increasing rotation
            qz = math.sin(angle / 2)
            qw = math.cos(angle / 2)
            R = _quaternion_to_rotation(0, 0, qz, qw)
            poses.append({
                "timestamp": i * 0.033,
                "quaternion": [0, 0, qz, qw],
                "rotation_matrix": R.tolist(),
                "position": [0, 0, 0],
            })
        corrections = poses_to_corrections(poses, _pinhole_calib(), smoothing_radius=3)
        # At least some non-identity corrections
        non_identity = sum(
            1 for c in corrections
            if not np.allclose(np.array(c["H_corr_3x3"]), np.eye(3), atol=1e-4)
        )
        assert non_identity > 0

    def test_missing_pose_interpolated(self) -> None:
        """None poses should be filled and marked."""
        aligned = [
            {"timestamp": 0, "quaternion": [0, 0, 0, 1],
             "rotation_matrix": np.eye(3).tolist(), "position": [0, 0, 0]},
            None,  # missing
            {"timestamp": 0.066, "quaternion": [0, 0, 0, 1],
             "rotation_matrix": np.eye(3).tolist(), "position": [0, 0, 0]},
        ]
        corrections = poses_to_corrections(aligned, _pinhole_calib())
        assert len(corrections) == 3
        assert corrections[1]["confidence"] == "interpolated"


# ---------------------------------------------------------------------------
# DPVO not installed
# ---------------------------------------------------------------------------

class TestDpvoNotInstalled:
    def test_import_error(self) -> None:
        from vibelab.ego_video.motion.dpvo_bridge import DPVO_AVAILABLE, run_dpvo_inference
        if DPVO_AVAILABLE:
            pytest.skip("DPVO is installed")
        with pytest.raises(ImportError, match="DPVO not installed"):
            run_dpvo_inference(Path("."), Path("."))


# ---------------------------------------------------------------------------
# Trajectory JSON
# ---------------------------------------------------------------------------

class TestTrajectoryJson:
    def test_schema_version(self) -> None:
        corrections = [{"frame_index": 0, "H_corr_3x3": np.eye(3).tolist()}]
        tj = build_trajectory_json(corrections, 5, 5, _pinhole_calib(), True)
        assert tj["schema_version"] == "dpvo_v1"
        assert tj["method"] == "dpvo"
        assert "frames" in tj

    def test_stabilize_from_json(self, tmp_path: Path) -> None:
        """Stabilize works from trajectory JSON without DPVO."""
        frame_dir = _make_test_frames(tmp_path / "frames", n=3)
        corrections = [
            {"frame_index": i, "H_corr_3x3": np.eye(3).tolist()}
            for i in range(3)
        ]
        tj = build_trajectory_json(corrections, 5, 2, _pinhole_calib(), False)
        output_dir = tmp_path / "output"
        report = apply_dpvo_stabilization(frame_dir, tj, output_dir)
        assert (output_dir / "stabilized").is_dir()
        assert (output_dir / "comparison").is_dir()
        assert "summary" in report
