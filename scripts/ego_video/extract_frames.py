#!/usr/bin/env python3
"""Extract RGB frames from a local video into a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from vibelab.ego_video.io.video import extract_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from a local video.")
    parser.add_argument("--video-path", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PNG frames.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index to export.")
    parser.add_argument("--max-frames", type=int, default=60, help="Maximum number of frames.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = extract_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        stride=args.stride,
    )
    print(f"Saved {saved} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
