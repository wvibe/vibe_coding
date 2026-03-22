# Motion Analysis CLI Pipeline — Design Doc

## Part 1 — PRD

### Persona
Wayne (ML engineer) + Collie (AI coding assistant) working on ego-video stabilization research.

### Problem
The current motion estimation and stabilization workflow lives in notebook `02_motion_baseline.ipynb`. It relies on interactive cells, inline plots, and manual parameter tweaking. This makes it impossible for AI-assisted coding to run, inspect, and iterate on the stabilization pipeline reliably.

We need a CLI-first pipeline that:
1. Runs motion estimation on extracted frame sets (from `prepare_frames.py`)
2. Produces structured output: per-frame motion data, cumulative transforms, stabilized frames
3. Enables before/after comparison without notebooks
4. Works with the three frame-rate datasets (30fps, 10fps, 3fps) already prepared

### User Stories

**US-1: Estimate frame-to-frame motion on a frame set (affine2d)**
```
$ python scripts/ego_video/analyze_motion.py estimate \
    --frame-set ~/Data/.../frame_sets/clip_001 \
    --fps 30 \
    --method affine2d \
    --output-dir ~/Data/.../frame_sets/clip_001/analysis/30fps_affine2d
```
→ Produces:
```
analysis/30fps_affine2d/
  motion.json              # per-frame motion (method-specific payload)
  transforms.json          # per-frame cumulative 3×3 transform matrices
  trajectory.json          # raw + smoothed cumulative path + corrections
  motion_summary.json      # aggregate stats
```

**US-2: Estimate motion using rotation3d method (with calibration)**
```
$ python scripts/ego_video/analyze_motion.py estimate \
    --frame-set ~/Data/.../frame_sets/clip_001 \
    --fps 10 \
    --method rotation3d \
    --output-dir ~/Data/.../frame_sets/clip_001/analysis/10fps_rotation3d
```
→ Same top-level file structure, but method-specific per-frame payload:
- `affine2d` frames store `{dx, dy, da, scale, inlier_count, matrix_2x3}`
- `rotation3d` frames store `{rotvec_rel, rotation_matrix_3x3, inlier_count}`

**US-3: Apply stabilization to frame set and produce before/after comparison**
```
$ python scripts/ego_video/analyze_motion.py stabilize \
    --frame-set ~/Data/.../frame_sets/clip_001 \
    --fps 30 \
    --method affine2d \
    --smoothing-radius 5 \
    --output-dir ~/Data/.../frame_sets/clip_001/analysis/30fps_affine2d \
    --comparison-width 960
```
→ Produces:
```
analysis/30fps_affine2d/
  stabilized/
    frame_00000.png ... frame_00019.png   # stabilized frames
  comparison/
    frame_00000.png ... frame_00019.png   # side-by-side raw|stabilized
  stabilization_report.json               # per-frame metrics + summary
```

**US-4: Inspect analysis results**
```
$ python scripts/ego_video/analyze_motion.py report \
    --analysis-dir ~/Data/.../frame_sets/clip_001/analysis/30fps_affine2d \
    --json
```
→ Summary of motion estimates, stabilization quality, before/after comparison.

### Output Format Contracts

**motion.json** — per-frame motion estimates (method-specific payloads):

For `affine2d`:
```json
{
  "method": "affine2d",
  "source_fps": 30,
  "actual_fps": 30.0,
  "num_frames": 20,
  "frames": [
    {
      "frame_index": 0,
      "source_frame": 300,
      "timestamp_sec": 10.0,
      "dx": 0.0, "dy": 0.0, "da": 0.0, "scale": 1.0,
      "inlier_count": 0,
      "matrix_2x3": [[1,0,0],[0,1,0]]
    },
    {
      "frame_index": 1,
      "source_frame": 301,
      "timestamp_sec": 10.033,
      "dx": 1.23, "dy": -0.45, "da": 0.002, "scale": 0.9998,
      "inlier_count": 142,
      "matrix_2x3": [[0.9998, -0.002, 1.23], [0.002, 0.9998, -0.45]]
    }
  ]
}
```

For `rotation3d`:
```json
{
  "method": "rotation3d",
  "source_fps": 10,
  "actual_fps": 10.0,
  "num_frames": 20,
  "frames": [
    {
      "frame_index": 0,
      "source_frame": 300,
      "timestamp_sec": 10.0,
      "rotvec_rel": [0.0, 0.0, 0.0],
      "rotation_matrix_3x3": [[1,0,0],[0,1,0],[0,0,1]],
      "inlier_count": 0,
      "confidence": "normal"
    },
    {
      "frame_index": 1,
      "source_frame": 303,
      "timestamp_sec": 10.1,
      "rotvec_rel": [0.001, -0.003, 0.002],
      "rotation_matrix_3x3": [[0.999, -0.002, 0.001], ...],
      "inlier_count": 87,
      "confidence": "normal"
    }
  ]
}
```

`confidence` field values: `"normal"` | `"low_inliers"` | `"degenerate"` (pure translation / insufficient parallax → identity fallback).

**transforms.json** — per-frame cumulative 3×3 transform matrices:
```json
{
  "method": "affine2d",
  "transform_convention": {
    "direction": "frame_0_to_frame_n",
    "composition_order": "right_multiply",
    "point_convention": "column_vector",
    "definition": "T_0→n = T_{n-1→n} * T_{n-2→n-1} * ... * T_{0→1}. Applies to column-vector points: p_n = T_0→n * p_0"
  },
  "transforms": [
    {
      "frame_index": 0,
      "matrix_3x3": [[1,0,0],[0,1,0],[0,0,1]]
    },
    {
      "frame_index": 1,
      "matrix_3x3": [[0.9998, -0.002, 1.23], [0.002, 0.9998, -0.45], [0, 0, 1]]
    }
  ]
}
```

For `affine2d`: the full 2×3 matrix from `estimateAffinePartial2D` (including scale) is embedded into a 3×3 homogeneous matrix and chain-multiplied. Scale is preserved, not discarded.

For `rotation3d`: the 3×3 rotation from `recoverPose` is chain-multiplied as `R_cum_n = R_{n-1→n} @ R_cum_{n-1}`. Same right-multiply convention.

**stabilization_report.json** — before/after quality metrics:
```json
{
  "method": "affine2d",
  "smoothing_radius": 5,
  "smoothing_radius_actual": 5,
  "frames": [
    {
      "frame_index": 0,
      "correction_magnitude": 0.0,
      "raw_alignment_score": 12.3,
      "stabilized_alignment_score": 8.1,
      "adjacent_raw_score": 5.2,
      "adjacent_stabilized_score": 3.1,
      "center_crop_raw_score": 10.1,
      "center_crop_stabilized_score": 6.5
    }
  ],
  "summary": {
    "avg_correction_magnitude": 1.5,
    "avg_raw_alignment_score": 15.2,
    "avg_stabilized_alignment_score": 10.8,
    "improvement_ratio": 0.29,
    "avg_adjacent_improvement": 0.35,
    "avg_center_crop_improvement": 0.31
  }
}
```

### Command Namespace
Single entry point: `scripts/ego_video/analyze_motion.py` with subcommands:
- `estimate` — run motion estimation on a frame set
- `stabilize` — apply stabilization and produce comparison frames
- `report` — summarize analysis results

### Not-Yet-Implemented (future)
- VO/SLAM integration (ORB-SLAM3, DROID-SLAM)
- Foreground mask / hand rejection
- Multi-clip batch analysis
- Motion heatmap visualization

---

## Part 2 — Technical Design

### Architecture

```
analyze_motion.py (CLI, argparse subcommands)
    │
    ├── estimate   →  new motion_analysis.py module
    ├── stabilize  →  reuses existing stabilize.py + new comparison logic
    └── report     →  reads output JSONs, computes summary
```

### New Module: `src/vibelab/ego_video/motion/motion_analysis.py`

This module operates on **frame sets** (directories of PNG images from `prepare_frames.py`), not videos.

#### Core Functions

```python
def estimate_frame_motion(
    frame_dir: Path,
    method: str = "affine2d",
    calibration: dict | None = None,
    max_corners: int = 200,
    quality_level: float = 0.01,
    min_distance: float = 30.0,
    manifest: dict | None = None,
) -> dict:
    """Estimate frame-to-frame motion on sequential PNG frames.
    
    Frame ordering is driven by manifest if provided (using source_frame
    indices for metadata), falling back to lexicographic filename sort.
    
    Returns dict matching method-specific motion.json schema.
    """
```

**Logic (affine2d):**
1. Load frames in manifest order (or sorted filename order)
2. For each consecutive pair (frame_i, frame_{i+1}):
   - Convert to grayscale
   - Detect corners with `goodFeaturesToTrack`
   - Track with `calcOpticalFlowPyrLK`
   - Fit global similarity with `estimateAffinePartial2D`
   - Store full 2×3 matrix + extracted (dx, dy, da, scale)
3. Frame 0 gets identity (zero motion, scale=1.0, identity matrix)
4. If estimation fails (< 4 inliers): store identity with `inlier_count=0`

**Logic (rotation3d):**
1. Same corner detection + optical flow tracking
2. Undistort matched points using calibration (fisheye model)
3. Estimate essential matrix with `findEssentialMat`
4. Recover rotation with `recoverPose`
5. Store 3×3 rotation matrix + Rodrigues vector per frame pair
6. **Degenerate cases**: if `findEssentialMat` fails or inlier count < threshold, store identity rotation with `confidence: "degenerate"`; if inliers are low but estimation succeeds, mark `confidence: "low_inliers"`

```python
def compute_cumulative_transforms(
    motion_data: dict,
) -> dict:
    """Accumulate per-frame motions into cumulative 3×3 transforms.
    
    For affine2d: embed 2×3 into 3×3 homogeneous, right-multiply chain.
    For rotation3d: right-multiply 3×3 rotations.
    
    Convention: T_0→n maps points from frame 0 space to frame n space.
    Composition: T_0→n = T_{n-1→n} @ T_0→{n-1}
    
    Returns dict matching transforms.json schema (with convention metadata).
    """
```

```python
def compute_trajectory(
    motion_data: dict,
    smoothing_radius: int = 5,
    num_frames: int | None = None,
) -> dict:
    """Compute raw and smoothed cumulative trajectory + corrections.
    
    For affine2d: trajectory in (cum_dx, cum_dy, cum_da) space, 
    smoothed with moving average, correction = smoothed - raw.
    
    For rotation3d: trajectory in Rodrigues vector space (3D).
    Cumulative rotvecs smoothed with moving average.
    Correction rotation: R_corr = R_smooth @ R_raw^T
    (This is the proper SO(3) group composition, not matrix subtraction.)
    
    Smoothing radius is clamped to min(radius, num_frames // 2) with warning.
    
    Returns dict with raw/smoothed/correction per frame.
    """
```

```python
def apply_stabilization(
    frame_dir: Path,
    motion_data: dict,
    trajectory_data: dict,
    output_dir: Path,
    method: str = "affine2d",
    calibration: dict | None = None,
    crop_ratio: float = 0.0,
    comparison_width: int = 960,
) -> dict:
    """Apply stabilization corrections and produce comparison images.
    
    For each frame:
    1. Load raw frame from frame_dir
    2. Apply correction:
       - affine2d: build correction warp matrix, cv2.warpAffine
       - rotation3d: use R_corr with fisheye-aware rectification map
    3. Save stabilized frame to output_dir/stabilized/
    4. Generate side-by-side [raw | stabilized] comparison (resized to comparison_width per side)
    5. Compute alignment scores:
       - vs reference frame (frame 0): full frame + center crop
       - vs adjacent frame: measures temporal stability
    
    Returns dict matching stabilization_report.json schema.
    """
```

### Edge Case Handling

**Very short sequences (N < 2):**
- `estimate`: returns motion.json with only frame 0 (identity). No pairs to estimate.
- `stabilize`: copies frame 0 to stabilized/ unchanged. Report shows zero correction.
- `report`: valid JSON with single entry.

**Missing / corrupt PNG in sequence:**
- If a PNG fails to load (`cv2.imread` returns None): skip it, log warning, record as `{"frame_index": N, "skipped": true, "reason": "corrupt_or_missing"}` in motion.json.
- Downstream stabilization treats skipped frames as identity (no correction applied).

**Manifest-driven ordering:**
- If manifest.json exists in the parent frame set, read `source_frame_indices` and `frames[].timestamp_sec` to populate output metadata.
- If no manifest, use lexicographic sort and infer frame indices from filenames.

### Mathematical Specification

**Affine2d cumulative transform:**
```
T_{i→i+1} = estimateAffinePartial2D(pts_i, pts_{i+1})  # 2×3 matrix
T_{i→i+1}_3x3 = [[a, -b, tx], [b, a, ty], [0, 0, 1]]  # embed to 3×3

T_0→n = T_{n-1→n} @ T_{n-2→n-1} @ ... @ T_{0→1}  # right-multiply chain

Decomposition: dx = tx, dy = ty, da = atan2(b, a), scale = sqrt(a² + b²)
```

**Rotation3d cumulative rotation:**
```
R_{i→i+1} = recoverPose(E, pts_i_undistorted, pts_{i+1}_undistorted)

R_cum_n = R_{n-1→n} @ R_cum_{n-1}    (R_cum_0 = I)

Rodrigues: rotvec_cum_n = cv2.Rodrigues(R_cum_n)
```

**Smoothing and correction (rotation3d):**
```
rotvec_cum = [cv2.Rodrigues(R_cum_i) for each frame]
rotvec_smooth = moving_average(rotvec_cum, radius)

R_smooth_i = cv2.Rodrigues(rotvec_smooth_i)
R_raw_i = R_cum_i

R_correction_i = R_smooth_i @ R_raw_i^T    # proper SO(3) group operation
```

This matches the existing implementation in `stabilize.py:compute_rotation_stabilization_trace()`.

### Key Design Decisions

1. **Operates on frame images, not video**: Decouples from video I/O; enables CLI iteration on static image sets at different frame rates.

2. **Full estimated matrix preserved**: For `affine2d`, the complete 2×3 matrix from `estimateAffinePartial2D` is stored (including scale). The decomposed `(dx, dy, da, scale)` are convenience fields derived from the stored matrix.

3. **Method-specific payloads**: `affine2d` and `rotation3d` have different per-frame fields in motion.json. This avoids forcing fake values into shared fields.

4. **Explicit transform conventions**: `transforms.json` includes `transform_convention` metadata (direction, composition order, point convention) to prevent silent consumer bugs.

5. **SO(3) correction via group composition**: For rotation3d, correction is computed as `R_smooth @ R_raw^T`, not by subtracting rotvecs. Smoothing happens in rotvec space (small-angle tangent space approximation), which is consistent with existing code and valid for the typical magnitude of ego-video rotations.

6. **Multiple alignment metrics**: Report includes reference-frame alignment, adjacent-frame stability, and center-crop scores. Center-crop metrics are more trustworthy per the stabilization design doc findings.

7. **Side-by-side comparison images**: Primary visual diagnostic. `--comparison-width` controls resize (default 960px per side = 1920 total).

### Data Flow

```
frame_sets/clip_001/
    ├── manifest.json
    ├── 30fps/frame_00000.png ... frame_00019.png
    ├── 10fps/...
    └── 3fps/...
         │
         │ (analyze_motion.py estimate)
         ▼
    analysis/
    ├── 30fps_affine2d/
    │   ├── motion.json
    │   ├── transforms.json
    │   ├── trajectory.json
    │   └── motion_summary.json
    ├── 10fps_rotation3d/
    │   └── ...
         │
         │ (analyze_motion.py stabilize)
         ▼
    └── 30fps_affine2d/
        ├── stabilized/frame_00000.png ...
        ├── comparison/frame_00000.png ...
        └── stabilization_report.json
```

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Too few features at low fps (3fps) | Degenerate estimates | confidence field flags degenerate cases; identity fallback |
| rotation3d requires calibration | CLI error if missing | Validate at CLI level with clear error message |
| Scale in affine not meaningful for ego-video | Confusing output | Store full matrix for correctness; decomposed scale is informational |
| Smoothing radius too large for 20 frames | Edge artifacts | Clamp to `min(radius, num_frames // 2)`, warn if clamped |
| Missing/corrupt PNGs | Broken pipeline | Skip + log + identity fallback; reflected in output JSON |
| Rotvec smoothing invalid for large rotations | Wrong correction | Documented as small-angle approximation; sufficient for frame-to-frame ego-motion |

---

## Part 3 — Test Plan

### PRD Acceptance Tests

| Test | Steps | Expected |
|------|-------|----------|
| T-1: Estimate affine2d | Run on 30fps frame set | motion.json with 20 entries, valid dx/dy/da/scale + matrix_2x3 |
| T-2: Estimate rotation3d | Run on 10fps with calibration | motion.json with rotvec_rel + rotation_matrix_3x3 |
| T-3: Cumulative transforms | Check transforms.json | Frame 0 = identity; convention metadata present |
| T-4: Transform chain correctness | Manual verify T_0→2 = T_{1→2} @ T_{0→1} | Products match |
| T-5: Stabilize produces output | Run stabilize on 30fps affine2d | stabilized/ + comparison/ dirs with 20 PNGs each |
| T-6: Alignment metrics | Check stabilization_report.json | All three score types present (reference, adjacent, center-crop) |
| T-7: Report works | Run report --json on analysis dir | Valid JSON matching schema |
| T-8: Missing calibration error | Run rotation3d without --calibration | Clear error, non-zero exit |
| T-9: Short sequence (N=1) | Run on single-frame set | Valid output with identity, no crash |
| T-10: Degenerate handling | Frame pair with no features | confidence="degenerate", identity matrix |

### Unit Tests (`tests/ego_video/test_motion_analysis.py`)

1. `test_estimate_affine2d_synthetic_shift` — horizontally shifted frames → expected dx, scale≈1.0
2. `test_estimate_affine2d_stores_full_matrix` — matrix_2x3 round-trips through 3×3 embedding
3. `test_estimate_rotation3d_requires_calibration` — raises without calibration
4. `test_rotation3d_degenerate_fallback` — blank frames → identity + confidence="degenerate"
5. `test_cumulative_transforms_identity_start` — first entry is identity matrix
6. `test_cumulative_transforms_chain_product` — T_0→2 == T_1→2 @ T_0→1
7. `test_transforms_convention_metadata` — direction, composition_order, point_convention present
8. `test_stabilize_produces_files` — correct output structure
9. `test_comparison_image_dimensions` — width = 2 × comparison_width
10. `test_alignment_score_identical_frames` — score = 0 for same frame
11. `test_smoothing_radius_clamping` — warns when radius > num_frames/2
12. `test_single_frame_sequence` — N=1 → valid output, no crash
13. `test_corrupt_png_skipped` — missing file → skipped entry in motion.json

---

## Part 4 — Rollout & Status

### Implementation Order
1. `src/vibelab/ego_video/motion/motion_analysis.py` — core frame-based analysis
2. `scripts/ego_video/analyze_motion.py` — CLI entry point
3. Unit tests
4. Manual end-to-end on real Ego10K frame sets

### Status
- [x] Design round 1 reviewed (gpt-5.3-codex)
- [ ] Design round 2 review
- [ ] Wayne approval
- [ ] Implementation
- [ ] Tests passing
- [ ] End-to-end verified

---

## Review Notes

### Round 1 — Reviewer: gpt-5.3-codex

**Major (4) — all addressed:**
1. ❌→✅ motion.json schema inconsistent for rotation3d: **Made method-specific payloads** — affine2d stores (dx,dy,da,scale,matrix_2x3), rotation3d stores (rotvec_rel, rotation_matrix_3x3, confidence).
2. ❌→✅ Affine math lossy (scale discarded): **Full 2×3 matrix now preserved** in motion.json and used for cumulative transform chain. Decomposed (dx,dy,da,scale) are convenience fields.
3. ❌→✅ transforms.json lacks convention metadata: **Added `transform_convention` object** with direction, composition_order, point_convention, and definition string.
4. ❌→✅ Rotation stabilization needs explicit SO(3) handling: **Added full mathematical specification** — `R_corr = R_smooth @ R_raw^T` (group composition, not matrix subtraction). Rotvec smoothing documented as small-angle tangent-space approximation.

**Minor (6) — all addressed:**
1. Manifest-driven ordering: specified manifest as primary ordering source with filename fallback.
2. source_fps vs actual_fps: both recorded in motion.json.
3. Short sequences (N<2): explicit behavior defined for estimate/stabilize/report.
4. Missing/corrupt PNG: skip + log + identity fallback with skipped entry in output.
5. Alignment metrics: added adjacent-frame stability and center-crop scores alongside reference-frame score.
6. Rotation3d degenerate cases: confidence field with "normal"/"low_inliers"/"degenerate" values.

**Nit (3) — all addressed:**
1. Output dir naming: standardized to `{fps}_{method}` convention.
2. US-2 wording: clarified "same top-level files, method-specific frame payload".
3. comparison-width: surfaced as CLI arg `--comparison-width` with default 960.
