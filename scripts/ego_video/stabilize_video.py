#!/usr/bin/env python3
"""Run a simple global-motion-based stabilization baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from vibelab.ego_video.motion.stabilize import stabilize_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stabilize a video with a simple baseline.")
    parser.add_argument("--video-path", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output-path", type=Path, required=True, help="Stabilized video output.")
    parser.add_argument(
        "--smoothing-radius",
        type=int,
        default=15,
        help="Temporal smoothing radius for camera trajectory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = stabilize_video(
        video_path=args.video_path,
        output_path=args.output_path,
        smoothing_radius=args.smoothing_radius,
    )
    print(f"Saved stabilized video to {output}")


if __name__ == "__main__":
    main()
