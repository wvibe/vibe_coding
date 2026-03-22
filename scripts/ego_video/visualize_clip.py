#!/usr/bin/env python3
"""Create a contact sheet to inspect an ego-centric clip quickly."""

from __future__ import annotations

import argparse
from pathlib import Path

from vibelab.ego_video.viz.clip import render_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a contact sheet from a clip.")
    parser.add_argument("--video-path", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output-path", type=Path, required=True, help="PNG output path.")
    parser.add_argument("--max-frames", type=int, default=12, help="Number of frames to sample.")
    parser.add_argument("--stride", type=int, default=10, help="Stride between sampled frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_contact_sheet(
        video_path=args.video_path,
        output_path=args.output_path,
        max_frames=args.max_frames,
        stride=args.stride,
    )
    print(f"Saved contact sheet to {output}")


if __name__ == "__main__":
    main()
