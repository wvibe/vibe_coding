"""Helpers for downloading a very small local subset from Egocentric-10K."""

from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from typing import Any

from datasets import Features, Value, load_dataset
from huggingface_hub import hf_hub_download

EGO10K_FEATURES = Features(
    {
        "mp4": Value("binary"),
        "json": {
            "factory_id": Value("string"),
            "worker_id": Value("string"),
            "video_index": Value("int64"),
            "duration_sec": Value("float64"),
            "width": Value("int64"),
            "height": Value("int64"),
            "fps": Value("float64"),
            "size_bytes": Value("int64"),
            "codec": Value("string"),
        },
        "__key__": Value("string"),
        "__url__": Value("string"),
    }
)


def _normalize_intrinsics(raw: dict[str, Any]) -> dict[str, Any]:
    distortion = {
        key: raw[key]
        for key in ("k1", "k2", "k3", "k4")
        if key in raw and raw[key] is not None
    }
    return {
        "model": raw.get("model"),
        "image_width": raw.get("image_width"),
        "image_height": raw.get("image_height"),
        "fx": raw.get("fx"),
        "fy": raw.get("fy"),
        "cx": raw.get("cx"),
        "cy": raw.get("cy"),
        "distortion": distortion,
        "metadata": {
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
                "k1",
                "k2",
                "k3",
                "k4",
            }
        },
    }


def download_worker_intrinsics(
    factory_id: str,
    worker_id: str,
    output_dir: str | Path,
    repo_id: str = "builddotai/Egocentric-10K",
    token: str | None = None,
) -> dict[str, Any]:
    """Download and normalize the intrinsics for one worker."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"{factory_id}/workers/{worker_id}/intrinsics.json"
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
        local_dir=output_path,
    )
    raw = json.loads(Path(local_path).read_text(encoding="utf-8"))
    normalized = _normalize_intrinsics(raw)
    normalized_path = output_path / "intrinsics.normalized.json"
    normalized_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def download_small_sample_set(
    output_dir: str | Path,
    data_files: list[str],
    max_samples: int = 3,
    repo_id: str = "builddotai/Egocentric-10K",
    token: str | None = None,
) -> Path:
    """Stream a few examples from one or more tar shards and save them locally."""

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    output_path = Path(output_dir)
    videos_dir = output_path / "videos"
    metadata_dir = output_path / "metadata"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    stream = load_dataset(
        repo_id,
        data_files=data_files,
        streaming=True,
        features=EGO10K_FEATURES,
        split="train",
        token=token,
    )

    samples: list[dict[str, Any]] = []
    calibration_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for example in islice(stream, max_samples):
        key = example["__key__"]
        meta = dict(example["json"])
        factory_id = meta["factory_id"]
        worker_id = meta["worker_id"]
        worker_key = (factory_id, worker_id)

        if worker_key not in calibration_cache:
            calibration_cache[worker_key] = download_worker_intrinsics(
                factory_id=factory_id,
                worker_id=worker_id,
                output_dir=output_path,
                repo_id=repo_id,
                token=token,
            )

        video_relpath = Path("videos") / f"{key}.mp4"
        metadata_relpath = Path("metadata") / f"{key}.json"
        (output_path / video_relpath).write_bytes(example["mp4"])
        (output_path / metadata_relpath).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        samples.append(
            {
                "sample_id": key,
                "video_path": str(video_relpath),
                "calibration": calibration_cache[worker_key],
                "metadata": {
                    **meta,
                    "source_url": example["__url__"],
                    "metadata_path": str(metadata_relpath),
                },
            }
        )

    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": samples}, indent=2), encoding="utf-8")
    return manifest_path
