#!/usr/bin/env python3
"""CLI for downloading ego-video samples and extracting multi-rate frame sets.

Subcommands
-----------
download  — Fetch a sample video + intrinsics from HuggingFace Hub.
extract   — Extract multi-rate frame sequences from a local video.
inspect   — Summarize and validate an extracted frame set.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tarfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_frames")


# ---------------------------------------------------------------------------
# download subcommand
# ---------------------------------------------------------------------------

def cmd_download(args: argparse.Namespace) -> None:
    """Download a sample video from HuggingFace Hub via direct tar extraction."""
    from huggingface_hub import hf_hub_download

    if args.max_samples <= 0:
        logger.error("--max-samples must be positive")
        sys.exit(1)
    if args.shard_index < 0:
        logger.error("--shard-index must be non-negative")
        sys.exit(1)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize worker path: accept both "factory_001/worker_001" and
    # "factory_001/workers/worker_001".
    worker_path = args.worker
    if "/workers/" not in worker_path:
        parts = worker_path.split("/")
        if len(parts) == 2:
            worker_path = f"{parts[0]}/workers/{parts[1]}"

    factory_id = worker_path.split("/")[0]
    worker_id = worker_path.split("/")[-1]

    # Build shard filename
    factory_short = factory_id.replace("_", "")
    worker_short = worker_id.replace("_", "")
    shard_name = f"{factory_short}_{worker_short}_part{args.shard_index:02d}.tar"
    shard_remote = f"{worker_path}/{shard_name}"

    logger.info("Downloading intrinsics: %s/intrinsics.json", worker_path)
    intrinsics_local = hf_hub_download(
        repo_id=args.repo_id,
        filename=f"{worker_path}/intrinsics.json",
        repo_type="dataset",
        local_dir=str(output_dir),
    )
    logger.info("Intrinsics saved: %s", intrinsics_local)

    # Load and normalize intrinsics
    intrinsics_raw = json.loads(Path(intrinsics_local).read_text(encoding="utf-8"))
    calibration = _normalize_intrinsics(intrinsics_raw)

    logger.info("Downloading tar shard: %s", shard_remote)
    tar_local = hf_hub_download(
        repo_id=args.repo_id,
        filename=shard_remote,
        repo_type="dataset",
        local_dir=str(output_dir / "cache"),
    )
    logger.info("Tar shard saved: %s", tar_local)

    # Extract videos and metadata from tar
    videos_dir = output_dir / "videos"
    metadata_dir = output_dir / "metadata"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    count = 0
    with tarfile.open(tar_local, "r") as tf:
        members = tf.getnames()
        mp4s = sorted(m for m in members if m.endswith(".mp4"))

        for mp4_name in mp4s:
            if count >= args.max_samples:
                break

            # Safe extraction: reject path traversal attacks
            _safe_tar_extract(tf, mp4_name, videos_dir)
            video_path = videos_dir / mp4_name

            # Extract matching JSON metadata if present
            json_name = mp4_name.replace(".mp4", ".json")
            meta = {}
            if json_name in members:
                _safe_tar_extract(tf, json_name, metadata_dir)
                meta_path = metadata_dir / json_name
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))

            sample_id = Path(mp4_name).stem
            samples.append(
                {
                    "sample_id": sample_id,
                    "video_path": str(video_path.relative_to(output_dir)),
                    "calibration": calibration,
                    "metadata": {
                        "factory_id": factory_id,
                        "worker_id": worker_id,
                        "repo_id": args.repo_id,
                        **meta,
                    },
                }
            )
            count += 1
            logger.info("Extracted sample %d: %s", count, sample_id)

    if not samples:
        logger.error("No mp4 files found in tar shard %s", shard_remote)
        sys.exit(1)

    manifest = {"samples": samples}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Download manifest written: %s (%d samples)", manifest_path, len(samples))


def _safe_tar_extract(tf: tarfile.TarFile, member_name: str, dest_dir: Path) -> None:
    """Extract a single tar member safely, rejecting path traversal."""
    resolved = (dest_dir / member_name).resolve()
    if not resolved.is_relative_to(dest_dir.resolve()):
        raise ValueError(
            f"Tar member {member_name!r} would extract outside {dest_dir}: {resolved}"
        )
    member = tf.getmember(member_name)
    if member.issym() or member.islnk():
        raise ValueError(f"Tar member {member_name!r} is a symlink/hardlink, skipping for safety")
    tf.extract(member, path=str(dest_dir))


def _normalize_intrinsics(raw: dict) -> dict:
    """Normalize raw intrinsics JSON into standard calibration dict."""
    distortion = {}
    for key in ("k1", "k2", "k3", "k4"):
        if key in raw and raw[key] is not None:
            distortion[key] = raw[key]

    return {
        "model": raw.get("model"),
        "image_width": raw.get("image_width"),
        "image_height": raw.get("image_height"),
        "fx": raw.get("fx"),
        "fy": raw.get("fy"),
        "cx": raw.get("cx"),
        "cy": raw.get("cy"),
        "distortion": distortion,
    }


# ---------------------------------------------------------------------------
# extract subcommand
# ---------------------------------------------------------------------------

def cmd_extract(args: argparse.Namespace) -> None:
    """Extract multi-rate frame sequences from a local video."""
    from vibelab.ego_video.io.frame_dataset import extract_multi_rate_frames

    video_path = Path(args.video).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # Parse fps list with validation
    raw_parts = [x.strip() for x in args.fps.split(",") if x.strip()]
    if not raw_parts:
        logger.error("--fps must contain at least one valid fps value")
        sys.exit(1)
    try:
        fps_list = [float(x) for x in raw_parts]
    except ValueError as e:
        logger.error("Invalid --fps value: %s", e)
        sys.exit(1)

    # Load calibration if provided
    calibration = None
    calibration_warnings: list[str] = []
    if args.calibration:
        cal_path = Path(args.calibration).expanduser().resolve()
        if cal_path.exists():
            try:
                raw = json.loads(cal_path.read_text(encoding="utf-8"))
                calibration = _normalize_intrinsics(raw)
            except Exception as e:
                calibration_warnings.append(f"Failed to parse calibration: {e}")
                logger.warning("Calibration parse error: %s", e)
        else:
            calibration_warnings.append(f"Calibration file not found: {cal_path}")
            logger.warning("Calibration file not found: %s", cal_path)

    # Build source metadata
    source_metadata = {}
    if args.sample_id:
        source_metadata["sample_id"] = args.sample_id
    if calibration_warnings:
        source_metadata["calibration_warnings"] = calibration_warnings

    manifest_path = extract_multi_rate_frames(
        video_path=video_path,
        output_dir=output_dir,
        start_sec=args.start_sec,
        num_frames=args.num_frames,
        target_fps_list=fps_list,
        calibration=calibration,
        source_metadata=source_metadata,
        strict=not args.allow_partial,
    )

    logger.info("Frame extraction complete: %s", manifest_path)


# ---------------------------------------------------------------------------
# inspect subcommand
# ---------------------------------------------------------------------------

def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect and validate an extracted frame set."""
    from vibelab.ego_video.io.frame_dataset import inspect_frame_set

    manifest_path = Path(args.frame_set).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"

    summary = inspect_frame_set(manifest_path)

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    # Human-readable output
    src = summary.get("source", {})
    ext = summary.get("extraction", {})
    print(f"Source: {src.get('video_path_absolute', src.get('video_path', 'N/A'))}")
    print(f"Calibration: {'present' if summary.get('calibration_present') else 'absent'}")
    print(f"Start: {ext.get('start_sec', 0)}s  |  Source FPS: {ext.get('source_fps', 'N/A')}")
    print(f"Requested frames per rate: {ext.get('requested_num_frames', 'N/A')}")
    print()

    for fs in summary.get("frame_sets", []):
        status = "✅" if fs.get("ok") else "❌"
        res = fs.get("resolution", "?")
        print(
            f"  {status} {fs['directory']:>10s}: "
            f"{fs['actual_frames']}/{fs['expected_frames']} frames  "
            f"({res})"
        )
        if fs.get("missing"):
            print(f"     Missing: {', '.join(fs['missing'][:5])}...")
        if fs.get("extra"):
            print(f"     Extra: {', '.join(fs['extra'][:5])}...")

    if summary.get("warnings"):
        print()
        for w in summary["warnings"]:
            print(f"  ⚠️  {w}")

    # Contact sheet
    if args.contact_sheet:
        _generate_contact_sheet(manifest_path.parent, summary, args.contact_sheet)


def _generate_contact_sheet(
    base_dir: Path,
    summary: dict,
    output_path: str,
) -> None:
    """Generate a thumbnail grid from the first few frames of each rate."""
    import cv2
    import numpy as np

    frame_sets = summary.get("frame_sets", [])
    if not frame_sets:
        logger.warning("No frame sets to create contact sheet")
        return

    thumb_w, thumb_h = 320, 180
    max_cols = 5
    rows_data: list[list[np.ndarray]] = []

    for fs in frame_sets:
        dir_path = base_dir / fs["directory"]
        if not dir_path.is_dir():
            continue
        pngs = sorted(dir_path.glob("frame_*.png"))[:max_cols]
        row: list[np.ndarray] = []
        for png in pngs:
            img = cv2.imread(str(png))
            if img is not None:
                row.append(cv2.resize(img, (thumb_w, thumb_h)))
        if row:
            # Pad row to max_cols
            while len(row) < max_cols:
                row.append(np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8))
            rows_data.append(row)

    if not rows_data:
        logger.warning("No frames found for contact sheet")
        return

    grid = np.vstack([np.hstack(row) for row in rows_data])
    out = Path(output_path).expanduser().resolve()
    cv2.imwrite(str(out), grid)
    logger.info("Contact sheet saved: %s", out)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare_frames",
        description="Download ego-video samples and extract multi-rate frame sets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- download ---
    dl = sub.add_parser("download", help="Download sample video from HuggingFace Hub.")
    dl.add_argument(
        "--repo-id",
        default="builddotai/Egocentric-10K",
        help="HF dataset repo id (default: %(default)s).",
    )
    dl.add_argument(
        "--worker",
        required=True,
        help="Worker path, e.g. 'factory_001/worker_001' or 'factory_001/workers/worker_001'.",
    )
    dl.add_argument("--shard-index", type=int, default=0, help="Tar shard index (default: 0).")
    dl.add_argument("--max-samples", type=int, default=1, help="Max videos to extract from shard.")
    dl.add_argument("--output-dir", required=True, help="Output directory.")
    dl.set_defaults(func=cmd_download)

    # --- extract ---
    ex = sub.add_parser("extract", help="Extract multi-rate frame sequences from a video.")
    ex.add_argument("--video", required=True, help="Path to source video file.")
    ex.add_argument("--start-sec", type=float, default=0.0, help="Start time in seconds.")
    ex.add_argument("--num-frames", type=int, default=20, help="Frames per rate (default: 20).")
    ex.add_argument(
        "--fps",
        default="30,10,3",
        help="Comma-separated target fps values (default: '30,10,3').",
    )
    ex.add_argument("--output-dir", required=True, help="Output directory for frame sets.")
    ex.add_argument("--calibration", help="Path to intrinsics.json (optional).")
    ex.add_argument("--sample-id", help="Sample identifier for metadata.")
    ex.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow partial extraction when frames are insufficient.",
    )
    ex.set_defaults(func=cmd_extract)

    # --- inspect ---
    ins = sub.add_parser("inspect", help="Inspect and validate a frame set.")
    ins.add_argument(
        "--frame-set",
        required=True,
        help="Path to frame set directory or manifest.json.",
    )
    ins.add_argument("--json", action="store_true", help="Output as JSON.")
    ins.add_argument("--contact-sheet", help="Save thumbnail grid to this path.")
    ins.set_defaults(func=cmd_inspect)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
