"""Visualization helpers for quick clip inspection."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from vibelab.ego_video.io.video import sample_frames


def render_contact_sheet(
    video_path: str | Path,
    output_path: str | Path,
    max_frames: int = 12,
    stride: int = 10,
) -> Path:
    """Render a simple contact sheet from uniformly sampled frames."""

    frames = sample_frames(video_path, max_frames=max_frames, stride=stride)
    if not frames:
        raise ValueError(f"No frames sampled from {video_path}")

    columns = min(4, len(frames))
    rows = (len(frames) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, frame in zip(axes, frames, strict=False):
        axis.imshow(frame)
        axis.axis("off")
    for axis in axes[len(frames) :]:
        axis.axis("off")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return output
