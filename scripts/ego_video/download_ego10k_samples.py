#!/usr/bin/env python3
"""Download a small local subset of ego-centric videos from Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a tiny Ego10K sample subset.")
    parser.add_argument("--repo-id", type=str, required=True, help="HF dataset repo id.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Local output directory for downloaded files and manifest.",
    )
    parser.add_argument(
        "--include",
        type=str,
        default=".mp4",
        help="Only keep remote files containing this substring.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=3,
        help="Maximum number of files to download for the MVP subset.",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="dataset",
        choices=["dataset", "model", "space"],
        help="HF repo type.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=args.repo_id, repo_type=args.repo_type)
    matched_files = [path for path in repo_files if args.include in path][: args.max_files]

    if not matched_files:
        raise ValueError(
            f"No remote files matched include pattern {args.include!r} in repo {args.repo_id!r}."
        )

    samples = []
    for index, remote_path in enumerate(matched_files):
        local_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=remote_path,
            repo_type=args.repo_type,
            local_dir=args.output_dir,
            local_dir_use_symlinks=False,
        )
        samples.append(
            {
                "sample_id": f"sample_{index:03d}",
                "video_path": str(Path(local_path).relative_to(args.output_dir)),
                "metadata": {"remote_path": remote_path, "repo_id": args.repo_id},
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": samples}, indent=2), encoding="utf-8")
    print(f"Downloaded {len(samples)} files into {args.output_dir}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
