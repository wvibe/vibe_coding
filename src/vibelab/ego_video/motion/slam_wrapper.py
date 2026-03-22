"""Placeholder interface for future SLAM integrations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SlamWrapper:
    """Thin placeholder for later ORB-SLAM3 or DROID-SLAM adapters."""

    name: str = "placeholder"

    def run(self, video_path: str | Path) -> None:
        raise NotImplementedError(
            "Complex SLAM backends are intentionally out of scope for the MVP stage."
        )
