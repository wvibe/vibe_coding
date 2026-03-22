#!/usr/bin/env python3
"""Estimate simple frame-to-frame camera motion from a local clip."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from vibelab.ego_video.motion.global_motion import estimate_global_motion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate global motion from a clip.")
    parser.add_argument("--video-path", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output-csv", type=Path, required=True, help="CSV path for motion table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motions = estimate_global_motion(args.video_path)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "dx", "dy", "da", "inlier_count"])
        for motion in motions:
            writer.writerow(
                [motion.frame_index, motion.dx, motion.dy, motion.da, motion.inlier_count]
            )

    print(f"Wrote {len(motions)} motion estimates to {args.output_csv}")


if __name__ == "__main__":
    main()
