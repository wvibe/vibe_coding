# Frame Dataset Pipeline — Design Doc

## Part 1 — PRD

### Persona
Wayne (ML engineer) + Collie (AI coding assistant) working on ego-video stabilization research.

### Problem
The current workflow relies on Jupyter notebooks for data download, clip extraction, and frame inspection. Notebooks are problematic for AI-assisted coding: opaque state, cell ordering dependencies, hard to diff/review. We need a CLI-first pipeline that produces static image sets for downstream algorithm experiments.

### User Stories

**US-1: Download a sample video**
```
$ python scripts/ego_video/prepare_frames.py download \
    --repo-id builddotai/Egocentric-10K \
    --worker factory_001/worker_001 \
    --max-samples 1 \
    --output-dir ~/Data/datasets/ego_video/builddotai/ego10k_samples
```
→ Downloads 1 video + intrinsics from HF, writes `manifest.json`.

**US-2: Extract multi-rate frame sequences from a video**
```
$ python scripts/ego_video/prepare_frames.py extract \
    --video ~/Data/.../videos/factory001_worker001_00000.mp4 \
    --start-sec 10 \
    --num-frames 20 \
    --fps 30,10,3 \
    --output-dir ~/Data/datasets/ego_video/builddotai/frame_sets/clip_001 \
    --calibration ~/Data/.../intrinsics.json
```
→ Produces:
```
clip_001/
  manifest.json          # source video, time range, calibration, per-set metadata
  30fps/
    frame_00000.png      # 20 consecutive frames at native 30fps (every frame)
    frame_00001.png
    ...
    frame_00019.png
  10fps/
    frame_00000.png      # 20 frames at 10fps (every 3rd frame from 30fps source)
    ...
    frame_00019.png
  3fps/
    frame_00000.png      # 20 frames at 3fps (every 10th frame from 30fps source)
    ...
    frame_00019.png
```

**US-3: Quick inspect / sanity check**
```
$ python scripts/ego_video/prepare_frames.py inspect \
    --frame-set ~/Data/.../frame_sets/clip_001 \
    --json
```
→ Default: human-readable summary. `--json`: structured JSON output.
→ Optionally `--contact-sheet output.png` saves a thumbnail grid.

### Output Format Contract

**manifest.json** (per frame set):
```json
{
  "source": {
    "video_path": "relative/path/to/video.mp4",
    "video_path_absolute": "/absolute/path/to/video.mp4",
    "repo_id": "builddotai/Egocentric-10K",
    "factory_id": "factory_001",
    "worker_id": "worker_001",
    "sample_id": "factory001_worker001_00000"
  },
  "calibration": {
    "model": "fisheye",
    "image_width": 1920,
    "image_height": 1080,
    "fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0,
    "distortion": {"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0}
  },
  "extraction": {
    "start_sec": 10.0,
    "source_fps": 30.0,
    "start_frame": 300,
    "requested_num_frames": 20
  },
  "frame_sets": [
    {
      "target_fps": 30,
      "actual_fps": 30.0,
      "stride": 1,
      "requested_num_frames": 20,
      "actual_num_frames": 20,
      "actual_duration_sec": 0.633,
      "directory": "30fps",
      "source_frame_indices": [300, 301, 302, "..."],
      "frames": [
        {"filename": "frame_00000.png", "source_frame": 300, "timestamp_sec": 10.0},
        {"filename": "frame_00001.png", "source_frame": 301, "timestamp_sec": 10.033}
      ]
    },
    {
      "target_fps": 10,
      "actual_fps": 10.0,
      "stride": 3,
      "requested_num_frames": 20,
      "actual_num_frames": 20,
      "actual_duration_sec": 1.9,
      "directory": "10fps",
      "source_frame_indices": [300, 303, 306, "..."],
      "frames": [
        {"filename": "frame_00000.png", "source_frame": 300, "timestamp_sec": 10.0},
        {"filename": "frame_00001.png", "source_frame": 303, "timestamp_sec": 10.1}
      ]
    }
  ],
  "created_at": "2026-03-22T12:00:00Z"
}
```

### Command Namespace
Single entry point: `scripts/ego_video/prepare_frames.py` with subcommands:
- `download` — fetch sample from HF
- `extract` — cut frames from a local video
- `inspect` — summarize a frame set

### Not-Yet-Implemented (future)
- Batch extraction across multiple clips / workers
- Undistorted frame extraction (apply fisheye correction before saving)
- Automatic "interesting segment" detection (high-motion windows)

---

## Part 2 — Technical Design

### Architecture

```
prepare_frames.py (CLI, argparse subcommands)
    │
    ├── download  →  vibelab.ego_video.io.hf_samples (existing, minor adapt)
    ├── extract   →  vibelab.ego_video.io.video (existing) + new frame_dataset.py
    └── inspect   →  new, reads manifest.json + counts files
```

### New Module: `src/vibelab/ego_video/io/frame_dataset.py`

Core function:
```python
def extract_multi_rate_frames(
    video_path: Path,
    output_dir: Path,
    start_sec: float,
    num_frames: int,
    target_fps_list: list[float],
    calibration: dict | None = None,
    source_metadata: dict | None = None,
    strict: bool = True,
) -> Path:
    """Extract frame sequences at multiple rates, return manifest path.
    
    Args:
        video_path: Source video file.
        output_dir: Output directory for frame sets.
        start_sec: Start time in seconds.
        num_frames: Number of frames to extract per rate.
        target_fps_list: List of target frame rates (must be > 0, <= source fps).
        calibration: Optional fisheye calibration dict. None is valid (stored as null).
        source_metadata: Optional source metadata dict.
        strict: If True, raise error when requested frames exceed available.
                If False, extract what's available and log warning.
    """
```

**`--num-frames` is the primary contract. `--duration-sec` is removed.**

Rationale: The user's requirement is "extract N consecutive frames at each rate." Duration is a derived quantity that differs per rate (20 frames at 30fps = 0.67s; 20 frames at 3fps = 6.67s). Making `num_frames` authoritative avoids ambiguity. Duration is computed and stored in manifest as `actual_duration_sec` per frame set.

### Frame Index Computation (time-grid based)

Instead of simple stride rounding, use timestamp-based nearest-frame selection:

```python
def compute_frame_indices(
    start_frame: int,
    source_fps: float,
    target_fps: float,
    num_frames: int,
    total_frames: int,
) -> list[int]:
    """Compute source frame indices for a target fps using time-grid sampling.
    
    For each output frame i (0..num_frames-1):
        target_time = start_time + i / target_fps
        source_frame = round(target_time * source_fps)
    
    This avoids drift from integer stride rounding and handles
    non-divisible fps ratios (e.g., 29.97 source, 10 target).
    """
    start_time = start_frame / source_fps
    indices = []
    for i in range(num_frames):
        t = start_time + i / target_fps
        frame_idx = round(t * source_fps)
        if frame_idx >= total_frames:
            break
        indices.append(frame_idx)
    return indices
```

This gives:
- **Exact temporal spacing**: each frame is exactly `1/target_fps` apart in time
- **No drift**: works correctly for 29.97fps sources
- **Stride is derived, not assumed**: `stride` in manifest is computed as the most common frame-index delta (informational only)
- **actual_fps**: computed from actual frame indices and source fps

### Validation Rules for `--fps`
- Each value must be > 0
- Each value must be <= source fps (error if target_fps > source_fps)
- No duplicates allowed
- Float values accepted; directory naming uses `{fps}fps` with decimals stripped if integer (e.g., `10fps`, `7.5fps`)

### Data Flow

```
HuggingFace Hub
    │ (direct hf_hub_download for tar + intrinsics)
    ▼
~/Data/datasets/ego_video/builddotai/ego10k_samples/
    ├── videos/factory001_worker001_00000.mp4
    ├── metadata/factory001_worker001_00000.json
    ├── factory_001/workers/worker_001/intrinsics.json
    └── manifest.json
    │
    │ (extract subcommand)
    ▼
~/Data/datasets/ego_video/builddotai/frame_sets/
    └── clip_001/
        ├── manifest.json
        ├── 30fps/frame_00000.png ... frame_00019.png
        ├── 10fps/frame_00000.png ... frame_00019.png
        └── 3fps/frame_00000.png ... frame_00019.png
```

### Download Subcommand

**Primary path**: Direct `hf_hub_download` for the tar shard + `intrinsics.json`, then extract video + metadata JSON from the tar. This avoids the `load_dataset` gated-repo token issues entirely.

```python
def download_sample_direct(
    repo_id: str,
    worker_path: str,       # e.g., "factory_001/workers/worker_001"
    shard_index: int,        # e.g., 0 for part00.tar
    output_dir: Path,
    max_samples: int = 1,
) -> Path:
    """Download tar shard via hf_hub_download, extract videos, return manifest path."""
```

Token handling: `hf_hub_download` uses the stored HF token automatically (from `huggingface-cli login`). No explicit token parameter needed.

**Fallback**: `load_dataset` streaming (existing `download_small_sample_set`), passing `token=True` for gated repo auth.

### Calibration Handling

- `--calibration` flag accepts path to `intrinsics.json` (optional)
- If omitted, manifest stores `"calibration": null`
- If file is malformed/partial, store what's parseable with a `"calibration_warnings": [...]` field
- Downstream tools must handle `null` calibration gracefully

### Manifest Atomicity

Write manifest to `manifest.json.tmp` first, then `os.replace()` to `manifest.json`. This prevents half-written manifests on interruption.

### Key Design Decisions

1. **PNG over JPEG**: Lossless. These are small sets (20 frames × 3 rates = 60 files per clip). Disk cost is negligible.

2. **Frame naming is sequential within each rate**: `frame_00000.png` through `frame_00019.png`. Manifest records full mapping back to absolute source frame indices and timestamps.

3. **Time-grid sampling over integer stride**: Guarantees correct temporal spacing for any fps ratio. Stride in manifest is informational/derived.

4. **All rates start from the same frame**: `frame_00000` at all rates is the same physical source frame. Critical for algorithm comparison across time scales.

5. **Calibration carried forward**: The manifest includes full fisheye calibration from intrinsics.json so downstream tools don't need to re-fetch it.

6. **`num_frames` is authoritative**: No `--duration-sec` flag. Duration varies per rate and is computed in manifest.

7. **Strict mode default**: By default, fail if requested frames can't all be extracted. `--allow-partial` flag enables graceful degradation with warnings.

8. **Relative paths in manifest**: `source.video_path` is relative to manifest location. `source.video_path_absolute` added for convenience but not relied upon.

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| HF gated access fails | Can't download | Primary path uses hf_hub_download (stored token); fallback to load_dataset with token=True |
| Source video fps ≠ 30 | Wrong frame spacing | Time-grid sampling; actual_fps computed from real indices |
| target_fps > source_fps | Upsampling (nonsensical) | Validate and reject at CLI level |
| Requested frames exceed available | Incomplete set | Strict mode (default) fails; --allow-partial saves what's available with clear manifest metadata |
| Large PNGs (1920×1080) | ~5MB each, 60 per clip = 300MB | Acceptable for research. Add optional --max-width resize later if needed |
| Interrupted write | Corrupt manifest | Atomic write via tmp + os.replace() |
| Missing/malformed intrinsics | No calibration | Store null with warnings; don't block extraction |

---

## Part 3 — Test Plan

### PRD Acceptance Tests (manual verification)

| Test | Steps | Expected |
|------|-------|----------|
| T-1: Download works | Run `download` subcommand | Video + intrinsics + manifest in output dir |
| T-2: Extract produces correct structure | Run `extract` with fps=30,10,3 num-frames=20 | 3 directories, 20 PNGs each, valid manifest |
| T-3: Frame alignment | Compare frame_00000.png across all 3 rates (pixel hash) | Identical bytes |
| T-4: Temporal spacing | At 10fps rate, verify source_frame_indices in manifest differ by exactly 3 for 30fps source | Indices: [300, 303, 306, ...] |
| T-5: Inspect works | Run `inspect` on a frame set, both human and --json modes | Summary matches actual files; JSON parses cleanly |
| T-6: Strict mode | Request 20 frames near end of video where only 12 remain, strict=True | Error raised with clear message |
| T-7: Partial mode | Same as T-6 but with --allow-partial | Saves 12 frames, manifest shows requested=20 actual=12, warning logged |

### Unit Tests (`tests/ego_video/test_frame_dataset.py`)

1. `test_compute_frame_indices_exact_divisor` — 30fps source, 10fps target → indices spaced by 3
2. `test_compute_frame_indices_non_divisor` — 29.97fps source, 10fps target → correct time-grid indices
3. `test_compute_frame_indices_target_exceeds_source` — raises ValueError
4. `test_manifest_schema_completeness` — all required fields present, source_frame_indices populated
5. `test_frame_count_matches_manifest` — file count == actual_num_frames in manifest
6. `test_all_rates_same_first_frame` — pixel-compare frame_00000 across rates (SHA256)
7. `test_strict_mode_raises` — insufficient frames → error
8. `test_allow_partial_succeeds` — insufficient frames → partial with metadata
9. `test_fps_validation` — reject 0, negative, duplicates, > source fps
10. `test_manifest_atomic_write` — manifest.json.tmp does not remain after success
11. `test_null_calibration` — extraction works without calibration, manifest has null

### Integration Test

- End-to-end: download 1 sample → extract frames → inspect → verify structure
- Requires HF access; mark as `@pytest.mark.integration`

---

## Part 4 — Rollout & Status

### Implementation Order
1. `src/vibelab/ego_video/io/frame_dataset.py` — core extraction logic + compute_frame_indices
2. `scripts/ego_video/prepare_frames.py` — CLI entry point (download + extract + inspect)
3. Unit tests
4. Manual end-to-end verification

### Status
- [x] Design round 1 reviewed (gpt-5.3-codex)
- [ ] Design round 2 review
- [ ] Wayne approval
- [ ] Implementation
- [ ] Tests passing
- [ ] End-to-end verified with real Ego10K data

---

## Review Notes

### Round 1 — Reviewer: gpt-5.3-codex

**Major (4) — all addressed:**
1. ❌→✅ `duration_sec` vs `num_frames` ambiguity: **Removed `--duration-sec`**. `num_frames` is now the sole authoritative parameter. Duration is computed per rate in manifest.
2. ❌→✅ Stride rounding drift for non-divisible fps: **Replaced with time-grid sampling** (`t = start_time + i / target_fps`, then nearest frame). Stride is derived/informational only.
3. ❌→✅ Manifest missing source frame indices: **Added `source_frame_indices` list and per-frame `source_frame` + `timestamp_sec`** to each frame_set entry.
4. ❌→✅ Partial extraction policy: **Added `--strict` (default) vs `--allow-partial`**. Manifest now includes `requested_num_frames` vs `actual_num_frames`.

**Minor (6) — all addressed:**
1. Token handling: switched to `hf_hub_download` primary path (uses stored token automatically), no explicit token param.
2. fps input validation: defined rules (>0, <=source, no dupes, float accepted, directory naming convention).
3. Calibration missing: explicit null handling with `calibration_warnings` field.
4. Inspect output: added `--json` flag for machine-readable output.
5. Manifest atomicity: tmp file + `os.replace()`.
6. Download fallback: designed concrete `download_sample_direct()` function with tar extraction.

**Nit (3) — all addressed:**
1. Manifest paths: primary path is now relative; absolute is supplementary.
2. Duration precision: `actual_duration_sec` computed from actual frame indices, not rounded.
3. T-4 rephrased: now asserts on frame indices, not visual judgment.
