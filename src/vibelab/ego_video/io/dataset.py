"""Dataset manifest helpers for small-sample ego-video experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CameraCalibration:
    """Minimal camera metadata used by the MVP pipeline."""

    model: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    distortion: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EgoVideoSample:
    """A single local sample entry for the dataset subset manifest."""

    sample_id: str
    video_path: Path
    calibration: CameraCalibration | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_calibration(raw: dict[str, Any] | None) -> CameraCalibration | None:
    if not raw:
        return None

    distortion = raw.get("distortion") or raw.get("fisheye") or {}
    if not distortion:
        distortion = {
            key: raw[key]
            for key in ("k1", "k2", "k3", "k4")
            if key in raw and raw[key] is not None
        }
    metadata = {
        key: value
        for key, value in raw.items()
        if key
        not in {
            "model",
            "image_width",
            "image_height",
            "fx",
            "fy",
            "cx",
            "cy",
            "distortion",
            "fisheye",
            "k1",
            "k2",
            "k3",
            "k4",
        }
    }

    return CameraCalibration(
        model=raw.get("model"),
        image_width=raw.get("image_width"),
        image_height=raw.get("image_height"),
        fx=raw.get("fx"),
        fy=raw.get("fy"),
        cx=raw.get("cx"),
        cy=raw.get("cy"),
        distortion=distortion if isinstance(distortion, dict) else {},
        metadata=metadata,
    )


def load_sample_manifest(manifest_path: str | Path) -> list[EgoVideoSample]:
    """Load a small local manifest that points to downloaded sample videos."""

    manifest = Path(manifest_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    root = manifest.parent
    samples: list[EgoVideoSample] = []

    for item in payload.get("samples", []):
        video_path = Path(item["video_path"])
        if not video_path.is_absolute():
            video_path = root / video_path
        samples.append(
            EgoVideoSample(
                sample_id=item["sample_id"],
                video_path=video_path,
                calibration=_parse_calibration(item.get("calibration")),
                metadata=item.get("metadata", {}),
            )
        )

    return samples
