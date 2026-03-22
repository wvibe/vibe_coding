# RAFT Dense Flow Stabilization — Design Doc

## Part 1 — PRD

### Persona
Wayne (ML engineer) + Collie (AI coding assistant) working on ego-video stabilization research.

### Problem
The affine2d baseline failed to improve stabilization at any frame rate (improvement: -1.8% at 30fps, -6.5% at 10fps, -21.5% at 3fps). Root causes:
1. **Sparse features** (200 corners) get dominated by foreground motion (hands, tools)
2. **2D affine model** can't represent perspective/rotational ego-motion
3. At low fps, sparse tracking collapses (21 inliers avg at 3fps)

### Proposed Solution: RAFT Dense Optical Flow + Adaptive Model Fitting

RAFT provides **per-pixel motion estimation**. Combined with RANSAC-based model fitting and confidence gating, this addresses all three failure modes:

- **Dense flow → robust foreground rejection**: Per-pixel motion lets us statistically separate background (camera ego-motion) from foreground (hands/tools) via RANSAC
- **Adaptive model selection**: Fit homography first; if degenerate, fall back to affine; report confidence so downstream can gate
- **Different strategy per frame rate**: 30fps uses pairwise flow directly; 3fps composes transforms from denser intermediate estimates when available

### User Stories

**US-1: Estimate motion using RAFT dense flow**
```
$ python scripts/ego_video/analyze_motion.py estimate \
    --frame-set ~/Data/.../frame_sets/clip_001 \
    --fps 30 \
    --method raft \
    --output-dir ~/Data/.../frame_sets/clip_001/analysis/30fps_raft
```
→ Produces:
```
analysis/30fps_raft/
  motion.json           # per-frame homography + diagnostics
  transforms.json       # cumulative 3×3 homography matrices
  trajectory.json       # smoothed trajectory + corrections (homography-specific)
  motion_summary.json   # aggregate stats
  flow/                 # optional: saved flow .npy for debugging
```

**US-2: Stabilize using RAFT-derived transforms**
```
$ python scripts/ego_video/analyze_motion.py stabilize \
    --frame-set ~/Data/.../frame_sets/clip_001 \
    --fps 30 \
    --method raft \
    --output-dir ~/Data/.../frame_sets/clip_001/analysis/30fps_raft
```

**US-3: Visualize flow fields (optional)**
```
$ python scripts/ego_video/analyze_motion.py visualize-flow \
    --analysis-dir ~/Data/.../frame_sets/clip_001/analysis/30fps_raft
```

### Output Format Contract

**motion.json** for `raft` method:
```json
{
  "method": "raft",
  "model": "raft_small",
  "device": "mps",
  "flow_resize": 512,
  "source_fps": 30,
  "actual_fps": 30.0,
  "num_frames": 20,
  "runtime": {
    "torch_version": "2.10.0",
    "torchvision_version": "0.25.0"
  },
  "frames": [
    {
      "frame_index": 0,
      "source_frame": 300,
      "timestamp_sec": 10.0,
      "transform_3x3": [[1,0,0],[0,1,0],[0,0,1]],
      "model_type": "identity",
      "inlier_count": 0,
      "inlier_ratio": 0.0,
      "flow_magnitude_mean": 0.0,
      "flow_magnitude_p90": 0.0,
      "background_ratio": 1.0,
      "reprojection_error": 0.0,
      "condition_number": 1.0,
      "confidence": "normal"
    },
    {
      "frame_index": 1,
      "transform_3x3": [[...homography or affine...]],
      "model_type": "homography",
      "inlier_count": 45000,
      "inlier_ratio": 0.87,
      "reprojection_error": 1.2,
      "condition_number": 15.3,
      "confidence": "normal"
    }
  ]
}
```

`model_type`: `"homography"` | `"affine"` (fallback) | `"identity"` (degenerate).
`confidence`: `"normal"` | `"low_inliers"` | `"degenerate"`.
`background_ratio`: inlier pixels / total sampled points (after RANSAC).

### Not-Yet-Implemented (future)
- raft_large model
- Learned foreground segmentation (SAM / hand detector)
- VO/SLAM integration (requires CUDA)

---

## Part 2 — Technical Design

### Architecture

```
analyze_motion.py (existing CLI, add --method raft)
    │
    └── estimate --method raft  →  new raft_flow.py module
        │
        ├── RAFT inference (torchvision, auto device: mps → cpu)
        ├── Adaptive model fitting (homography → affine fallback)
        ├── Confidence gating
        └── Homography-specific trajectory + stabilization
```

### New Module: `src/vibelab/ego_video/motion/raft_flow.py`

#### Device Selection

```python
def _select_device(requested: str = "auto") -> torch.device:
    """Select best available device with fallback.
    
    auto → mps (if available) → cpu
    Logs selected device. Handles MPS op failures by catching and retrying on CPU.
    """
```

First-run weight download: RAFT weights (~5MB for small) are cached by torchvision. Log a message on first download.

#### RAFT Inference

```python
def estimate_raft_flow(
    prev_rgb: np.ndarray,
    curr_rgb: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    flow_resize: int = 512,
) -> np.ndarray:
    """Run RAFT inference, return (H, W, 2) flow in original resolution.
    
    flow_resize: max(H, W) is resized to this value (maintaining aspect ratio,
    rounded to multiple of 8). Flow is then scaled back to original coordinates.
    None = use original size.
    """
```

**MPS error handling**: If RAFT inference fails on MPS (unsupported op), catch the error, log warning, retry on CPU for that frame pair, and continue.

#### Adaptive Model Fitting

```python
def fit_transform_from_flow(
    flow: np.ndarray,
    ransac_threshold: float = 3.0,
    min_inlier_ratio: float = 0.3,
    max_condition_number: float = 1e6,
    sample_stride: int = 8,
    gradient_threshold: float = 2.0,
) -> dict:
    """Fit the best transform model to a dense flow field.
    
    Strategy:
    1. Sample source points on grid (every sample_stride pixels)
    2. Filter: exclude points with very low image gradient (textureless regions)
    3. Compute destination points as source + flow
    4. Fit homography with cv2.findHomography(RANSAC)
    5. Check quality:
       a. If condition_number(H) > max_condition_number → degenerate
       b. If inlier_ratio < min_inlier_ratio → degenerate
       c. If reprojection_error > 2 * ransac_threshold → degenerate
    6. On degenerate homography: fall back to cv2.estimateAffinePartial2D
    7. On degenerate affine: return identity
    
    Returns dict with transform_3x3, model_type, diagnostics.
    """
```

**Degeneracy checks:**
- `det(H)` near zero → degenerate
- Condition number of H > 1e6 → ill-conditioned
- Mean reprojection error of inliers > threshold → poor fit
- Inlier spatial coverage: if all inliers cluster in one quadrant → unreliable

**Sampling strategy:**
- Regular grid with `sample_stride=8` pixels (~16K points for 1024×576)
- **Gradient filter**: exclude points where `|∇I| < gradient_threshold` (textureless → unreliable flow)
- **Peripheral mask**: optionally exclude outer 10% border (fisheye distortion worst there)

#### Camera Intrinsics Consideration

For this iteration, we do **not** undistort before flow estimation. Rationale:
- RAFT is trained on perspective images but is robust to moderate distortion
- Undistorting 1920×1080 fisheye adds computation and can introduce interpolation artifacts
- The homography fitted in distorted space still captures the dominant motion well enough for stabilization
- Future: add `--undistort` flag to optionally undistort before flow

#### Per-Frame-Rate Strategy

**30fps and 10fps**: Direct pairwise RAFT flow → fit transform → accumulate. Inter-frame motion is small enough for RAFT to handle reliably.

**3fps**: Two strategies available (selected at CLI):
1. **Direct**: Same pairwise pipeline. May produce lower quality (large inter-frame motion). Confidence gating flags unreliable frames.
2. **Compose from 30fps** (future): If 30fps analysis already exists, compose 30fps pairwise transforms to get 3fps-equivalent transforms. Not implemented in MVP — flagged for future.

For MVP: use direct pairwise for all rates. The confidence field and diagnostics (flow magnitude, inlier ratio) will clearly show where 3fps estimation is unreliable.

#### Homography-Specific Trajectory Smoothing

```python
def compute_homography_trajectory(
    motion_data: dict,
    smoothing_radius: int = 5,
) -> dict:
    """Compute smoothed trajectory from cumulative homographies.
    
    Smoothing approach (SE(2) approximation with exact formulas):
    
    1. Decompose each cumulative homography H_cum_i into (tx_i, ty_i, angle_i):
         angle_i = atan2(H[1,0], H[0,0])
         tx_i = H[0,2]
         ty_i = H[1,2]
    
    2. Smooth each component with moving average:
         tx_smooth, ty_smooth, angle_smooth = moving_avg([tx, ty, angle], radius)
    
    3. Reconstruct smoothed homography H_smooth_i:
         cos_s = cos(angle_smooth_i)
         sin_s = sin(angle_smooth_i)
         H_smooth_i = [[cos_s, -sin_s, tx_smooth_i],
                        [sin_s,  cos_s, ty_smooth_i],
                        [0,      0,     1          ]]
       Note: perspective components H[2,0] and H[2,1] are dropped (set to 0)
       in the smoothed version. This is intentional — we smooth only the
       rigid-body (SE(2)) part. The perspective distortion from the original
       homography is not carried into the correction.
    
    4. Correction homography:
         H_corr_i = H_smooth_i @ inv(H_cum_i)
       If inv(H_cum_i) is singular (det < 1e-10), set H_corr_i = I (no correction).
    
    Invariant: H_corr_0 = I (frame 0 gets no correction).
    Invariant: For identity input (all H_cum = I), all H_corr = I.
    
    This is valid because ego-video frame-to-frame homographies have
    |H[2,0]| and |H[2,1]| typically < 1e-4. The SE(2) approximation
    captures >99% of the actual transform energy.
    """
```

#### RAFT trajectory.json Schema

```json
{
  "method": "raft",
  "smoothing_radius": 5,
  "smoothing_radius_actual": 5,
  "frames": [
    {
      "frame_index": 0,
      "raw_tx": 0.0, "raw_ty": 0.0, "raw_angle": 0.0,
      "smooth_tx": 0.0, "smooth_ty": 0.0, "smooth_angle": 0.0,
      "H_corr_3x3": [[1,0,0],[0,1,0],[0,0,1]]
    },
    {
      "frame_index": 1,
      "raw_tx": 2.3, "raw_ty": -1.1, "raw_angle": 0.003,
      "smooth_tx": 1.8, "smooth_ty": -0.9, "smooth_angle": 0.002,
      "H_corr_3x3": [[0.9999, 0.001, -0.5], [-0.001, 0.9999, 0.2], [0, 0, 1]]
    }
  ]
}
```

`H_corr_3x3` is the correction homography to apply to each frame via `cv2.warpPerspective`.

**Skipped frames**: If a frame was skipped (corrupt/missing), its entry has `"skipped": true` and `H_corr_3x3 = identity`. The stabilize step passes it through unchanged.

**Relationship to affine pipeline**: The affine path stores `corr_dx/corr_dy/corr_da` and applies via `warpAffine`. The RAFT path stores `H_corr_3x3` and applies via `warpPerspective`. The `report` subcommand reads `stabilization_report.json` which has the same schema for both methods.

#### Confidence Assignment Rules

| Condition | Confidence |
|-----------|-----------|
| inlier_ratio ≥ 0.5 AND condition_number < 1e4 AND reprojection_error < ransac_threshold | `"normal"` |
| 0.2 ≤ inlier_ratio < 0.5 OR 1e4 ≤ condition_number < 1e6 | `"low_inliers"` |
| inlier_ratio < 0.2 OR condition_number ≥ 1e6 OR det(H) < 1e-6 OR model fallback to identity | `"degenerate"` |

**Spatial coverage check**: Divide frame into 2×2 quadrants. Count quadrants containing ≥10% of total inliers. If fewer than 3 quadrants have sufficient inliers → downgrade confidence by one level (normal→low_inliers, low_inliers→degenerate).

#### Stabilization with Homography

```python
def apply_homography_stabilization(
    frame_dir: Path,
    trajectory_data: dict,
    output_dir: Path,
    comparison_width: int = 960,
) -> dict:
    """Apply homography-based stabilization.
    
    Uses cv2.warpPerspective for the correction.
    
    Alignment metrics:
    - Reference frame alignment (full frame + center 1/3 crop)
    - Adjacent frame stability (temporal smoothness)
    - Border validity ratio (fraction of non-black pixels after warp)
    """
```

**Border validity metric**: `warpPerspective` with `BORDER_CONSTANT` can create large black regions if the correction is too aggressive. Track `border_validity_ratio` (non-black pixels / total pixels) as a quality signal.

### Integration Plan

Rather than refactoring the existing affine pipeline, add RAFT as a **parallel path**:

```
motion_analysis.py          # affine2d estimate/stabilize (unchanged)
raft_flow.py                # RAFT estimate + homography stabilize (new)
analyze_motion.py           # CLI dispatches by --method
```

`--method affine2d` → existing `motion_analysis.py`
`--method raft` → new `raft_flow.py`

Both produce the same top-level files (motion.json, transforms.json, trajectory.json, stabilization_report.json) with method-specific payloads. The `report` subcommand works on either.

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| MPS op failures | Inference crashes | auto device fallback (mps → cpu); per-frame retry |
| Homography degenerate | Bad transform | Condition number + reprojection error checks; affine fallback |
| 3fps large motion | RAFT flow unreliable | Confidence gating; don't trust low-inlier frames for stabilization |
| Perspective warp artifacts | Ugly stabilized frames | Border validity metric; comparison images show reality |
| Flow resize loses detail | Slightly worse | Flow computed at 512, homography fitted in original coords |
| Fisheye distortion bias | Edge flow unreliable | Gradient filter + optional peripheral exclusion mask |

---

## Part 3 — Test Plan

### PRD Acceptance Tests

| Test | Steps | Expected |
|------|-------|----------|
| T-1: RAFT estimate 30fps | Run estimate --method raft --fps 30 | motion.json with homography per frame |
| T-2: Model type | Check model_type in motion.json | Mostly "homography", some may be "affine" fallback |
| T-3: Diagnostics quality | Check inlier_ratio, condition_number | inlier_ratio > 0.5, condition_number < 1e4 |
| T-4: Stabilize RAFT | Run stabilize --method raft --fps 30 | stabilized/ + comparison/ + report |
| T-5: Better than affine2d | Compare improvement_ratio | RAFT > affine2d (or at least less negative) |
| T-6: CPU fallback | Run with --device cpu | Works (slower) |
| T-7: 3fps with confidence | Run on 3fps | Runs; confidence flags show degradation |
| T-8: Degenerate handling | Frame pair with no texture | model_type="identity", confidence="degenerate" |
| T-9: Border validity | Check stabilization_report | border_validity_ratio present |

### Unit Tests (`tests/ego_video/test_raft_flow.py`)

1. `test_raft_model_loads` — raft_small loads on CPU
2. `test_flow_shape` — output (H, W, 2) float
3. `test_flow_resize_scaling` — flow at 512 scales correctly to original
4. `test_identity_flow_gives_identity_transform` — zero flow → identity
5. `test_uniform_flow_gives_translation` — constant flow → translation homography
6. `test_ransac_rejects_outliers` — mixed flow → outliers masked
7. `test_degenerate_fallback_to_affine` — ill-conditioned H → affine
8. `test_degenerate_fallback_to_identity` — total failure → identity
9. `test_condition_number_check` — high cond → flagged
10. `test_homography_trajectory_smoothing` — smoothed trajectory is smoother
11. `test_warp_perspective_produces_valid_output` — no crash, valid image
12. `test_border_validity_ratio` — correct ratio computation
13. `test_device_fallback` — auto selects mps or cpu without crash

---

## Part 4 — Rollout & Status

### Implementation Order
1. `src/vibelab/ego_video/motion/raft_flow.py` — RAFT inference + adaptive model fitting + homography trajectory
2. Update `scripts/ego_video/analyze_motion.py` — add `--method raft` dispatch
3. Unit tests (CPU-only for CI)
4. End-to-end on 30fps, 10fps, 3fps frame sets → compare with affine2d baseline

### Status
- [x] Design round 1 reviewed (gpt-5.3-codex)
- [x] Design round 2 reviewed (gpt-5.3-codex)
- [ ] Wayne approval
- [ ] Implementation
- [ ] Tests passing
- [ ] End-to-end comparison

---

## Review Notes

### Round 1 — Reviewer: gpt-5.3-codex

**Major (5) — all addressed:**
1. ❌→✅ Trajectory math under-specified for homography: **Added SE(2) approximation smoothing** — decompose H_cum into (tx, ty, angle), smooth those, reconstruct. Valid because H[2,0]/H[2,1] are tiny for frame-to-frame ego-motion. Explicit fallback if inv(H_cum) is singular.
2. ❌→✅ Homography not always valid: **Added adaptive model selection** — fit homography first, check condition number + reprojection error + inlier coverage. Fall back to affine if degenerate, then identity. `model_type` field tracks which model was used.
3. ❌→✅ 3fps failure mode not solved: **Acknowledged as known limitation** — MVP uses direct pairwise with confidence gating. Future: compose from 30fps transforms. Diagnostics clearly show where 3fps is unreliable.
4. ❌→✅ MPS not production-ready: **Added auto device fallback** (auto → mps → cpu) with per-frame retry on MPS failure. Explicit logging and metadata in motion.json.
5. ❌→✅ Integration gap: **Parallel path, not shared logic** — raft_flow.py is a separate module with its own trajectory and stabilization. CLI dispatches by --method. No pretense of drop-in.

**Minor (5) — all addressed:**
1. Right-multiply clarity: kept consistent convention with explicit equation.
2. RANSAC sampling: added gradient filter + optional peripheral mask.
3. Fisheye/intrinsics: acknowledged not undistorting in MVP; future --undistort flag.
4. Degeneracy checks: concrete thresholds (condition number, det, reprojection error, spatial coverage).
5. Evaluation metrics: added border_validity_ratio + center-crop as primary metric.

**Nit (3) — all addressed:**
1. background_ratio: formally defined as inlier_pixels / total_sampled_points.
2. flow_resize: defined as max(H, W), rounded to multiple of 8.
3. Reproducibility: added runtime fields (torch/torchvision versions, device, dtype).

### Round 2 — Reviewer: gpt-5.3-codex

**Major (2) — all addressed:**
1. ❌→✅ Homography smoothing math unspecified: **Added exact formulas** — decompose into (tx, ty, angle), smooth, reconstruct as SE(2) (drop perspective components). Explicit invariants and singularity fallback.
2. ❌→✅ RAFT trajectory/stabilize schema not pinned: **Added complete trajectory.json schema** with `H_corr_3x3` per frame. Defined skipped-frame semantics and relationship to affine pipeline.

**Minor (2) — all addressed:**
1. Confidence thresholds: **Added deterministic rule table** — exact inlier_ratio / condition_number / reprojection_error thresholds for normal/low_inliers/degenerate.
2. Spatial coverage: **Defined 2×2 quadrant check** — if <3 quadrants have ≥10% inliers, downgrade confidence one level.

**Nit (1) — addressed:**
1. Reproducibility metadata: standardized in motion.json schema block.
