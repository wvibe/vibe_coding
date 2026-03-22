# Stabilization Evaluation Metrics — Design Doc

## Part 1 — PRD

### Persona
Wayne (ML engineer) + Collie (AI coding assistant) working on ego-video stabilization research.

### Problem
Current stabilization metrics report negative improvement (-1.8% to -36.4%) even when the stabilization pipeline is working correctly. Root cause: the primary metric (per-frame L1 alignment against frame 0) is fundamentally wrong for ego-video scenes where:

1. **The scene is changing** — worker's hands move, tools rotate, workpieces shift
2. **Foreground motion dominates pixel differences** — hands/tools cover 20-40% of the frame
3. **Stabilization introduces valid black borders** — `warpPerspective` creates black regions that inflate error

We need metrics that answer: **"Is the background more stable after correction?"**

### Metric Design Principles

1. **Only measure what stabilization can fix** — background camera motion, not foreground objects
2. **Only measure where data is valid** — exclude black borders, exclude unreliable periphery
3. **Measure temporal smoothness, not frame-0 similarity** — adjacent-frame consistency, not identity
4. **Be robust to illumination/exposure changes** — use gradient-domain metrics alongside intensity
5. **Report per-frame + aggregate** — identify which frames benefit vs degrade

---

## Metric Intuition & Real-World Examples

### Reading the numbers

All metrics follow one convention:
- **Scores** (MAE, magnitude): **lower = more stable = better**
- **Improvement ratios**: **positive = stabilization helped**, negative = made it worse
- **Improvement formula**: `improvement = 1 - stabilized / raw` (with eps guard)

### M1 Example: Adjacent Stability (RAFT @ 10fps, real data)

```
raw_combined = 0.0838    # raw frames: background changes this much between frames
stab_combined = 0.0701   # stabilized: background changes less

improvement = 1 - 0.0701/0.0838 = +16.8%  ✅
```
→ After stabilization, consecutive frames' backgrounds differ 16.8% less.

**Counter-example** (RAFT @ 30fps):
```
raw_combined = 0.0477    # already very small motion at 30fps
stab_combined = 0.0535   # homography overcorrection made it worse

improvement = 1 - 0.0535/0.0477 = -12.2%  ❌
```
→ The perspective warp introduced more variation than it removed.

### M2 Example: Residual Motion (RAFT @ 3fps, real data)

```
raw_magnitude = 28.31 px    # background moves ~28 pixels between raw frames
stab_magnitude = 6.06 px    # after stabilization, only ~6 pixels of residual motion

improvement = 1 - 6.06/28.31 = +78.6%  ✅
```
→ Camera-induced background motion reduced by nearly 80%.

**Counter-example** (affine2d @ 10fps):
```
raw_magnitude = 13.80 px
stab_magnitude = 20.40 px   # correction went the wrong direction

improvement = 1 - 20.40/13.80 = -47.8%  ❌
```
→ Sparse tracking failed at this frame rate; the correction increased background motion.

### M3 Example: Border Validity

```
border_validity = 0.97   # 97% real pixels, 3% black border from warp  ✅ normal
border_validity = 0.72   # 28% black border  ❌ correction too aggressive, frame flagged
```

### Full Results Table (2026-03-22)

| Method | FPS | M1 Stability | M2 Motion | Interpretation |
|--------|-----|:---:|:---:|------|
| affine2d | 30 | **+10.5%** ✅ | **+22.8%** ✅ | Small motion, simple model works |
| affine2d | 10 | -6.7% ❌ | -81.4% ❌ | Large motion, sparse tracking collapse |
| affine2d | 3 | -16.0% ❌ | +9.6% | Tracking mostly failed |
| RAFT | 30 | -16.0% ❌ | -6.6% ❌ | Small motion, homography overfits |
| RAFT | 10 | **+16.8%** ✅ | **+58.5%** ✅ | Medium motion, dense flow advantage |
| RAFT | 3 | **+7.8%** ✅ | **+80.5%** ✅ | Large motion, only dense flow survives |

**Key insight**: Method complexity should match motion complexity. Affine2d (4 DOF) is right-sized for small motion; RAFT homography (8 DOF) is needed for large motion but overfits on small motion. An adaptive strategy that selects per-frame based on flow magnitude is the natural next step.

---

## Part 2 — Technical Design

### Valid Region Mask (Two-Tier)

All metrics are computed within a valid region mask. Two implementation tiers:

**MVP (Tier 1):**
```
valid_mask = non_border_mask ∩ center_crop_mask ∩ low_flow_proxy_mask
```

- `non_border_mask`: pixel gray > 2 in BOTH frames being compared (pairwise intersection)
- `center_crop_mask`: inner region (configurable, default 50% = exclude outer 25% each side)
- `low_flow_proxy_mask`: exclude pixels where raw optical flow magnitude > P75 of the frame pair's flow field. This uses the RAFT flow from estimation (stored as reference) and acts as a coarse foreground exclusion.

**V2 (Tier 2, future):**
```
valid_mask = non_border_mask ∩ center_crop_mask ∩ ransac_inlier_mask
```

- `ransac_inlier_mask`: per-pixel mask from RAFT + RANSAC homography fitting, cached during estimation. More precise foreground exclusion but requires storing per-frame masks (~200KB per frame).

**Flow source for low_flow_proxy**: The raw (pre-stabilization) RAFT flow field between frame i-1 and frame i. This is already computed during estimation. For MVP, we re-run RAFT on the raw frame pair (or cache flow magnitudes in motion.json). Since we're already running RAFT for estimation, adding `flow_magnitude_at_grid` to motion.json is cheap.

### Metric Definitions

#### M1: Adjacent Frame Background Stability (primary)

**Dual-domain**: compute in both intensity and gradient domains to be robust to illumination changes.

```python
def adjacent_background_stability(
    prev_frame: np.ndarray,   # grayscale
    curr_frame: np.ndarray,   # grayscale
    valid_mask: np.ndarray,   # pairwise intersection mask
) -> dict:
    """Returns {intensity_mae, gradient_mae, combined_score}."""
    
    # Intensity domain: MAE on valid pixels
    valid = valid_mask & (prev_frame > 2) & (curr_frame > 2)
    if valid.sum() < 100:
        return {"intensity_mae": nan, "gradient_mae": nan, "combined_score": nan}
    
    intensity_mae = |prev[valid] - curr[valid]|.mean()
    
    # Gradient domain: Sobel magnitude MAE (illumination-robust)
    prev_grad = sobel_magnitude(prev_frame)
    curr_grad = sobel_magnitude(curr_frame)
    gradient_mae = |prev_grad[valid] - curr_grad[valid]|.mean()
    
    # Combined: equal weight, both normalized to [0,1]
    # max_grad = per-frame max(prev_grad[valid].max(), curr_grad[valid].max(), 1.0)
    # This is per-frame to avoid cross-video comparability issues
    max_grad = max(prev_grad[valid].max(), curr_grad[valid].max(), 1.0)
    combined = 0.5 * (intensity_mae / 255.0) + 0.5 * (gradient_mae / max_grad)
    
    return {intensity_mae, gradient_mae, combined_score}
```

Computed for both raw and stabilized frame pairs. Improvement = `1 - stab/raw` (with eps=1e-6 when raw ≈ 0; return NaN if raw < eps).

Report at **three crop ratios**: 0.33, 0.50, 0.66 — default primary is 0.50.

#### M2: Residual Background Motion Magnitude (diagnostic)

**Robust estimation**: Uses sparse tracking but with explicit anti-foreground measures.

```python
def residual_background_motion(
    prev_stab: np.ndarray,
    curr_stab: np.ndarray,
    center_crop_ratio: float = 0.5,
    max_corners: int = 200,
    min_inlier_ratio: float = 0.3,
    min_quadrant_coverage: int = 3,
) -> dict:
    """Estimate residual global motion in stabilized center crop.
    
    Steps:
    1. Crop to center region
    2. Detect corners + LK optical flow
    3. Fit affine with RANSAC
    4. Validity checks:
       a. inlier_ratio >= min_inlier_ratio
       b. inliers cover >= min_quadrant_coverage quadrants (2x2)
       If checks fail: return {magnitude: NaN, reliable: False}
    5. Compute robust statistics:
       - median_displacement (median of individual point displacements)
       - trimmed_mean_displacement (trim top/bottom 10%)
       - affine_translation_magnitude (sqrt(dx² + dy²) from fitted transform)
    
    Returns: {
        affine_magnitude, median_displacement, trimmed_mean_displacement,
        inlier_ratio, quadrant_coverage, reliable: bool
    }
    """
```

**Same sampling with coordinate mapping**: 
1. Detect corners on raw frame i (center crop region)
2. For raw: track from raw frame i to raw frame i+1 using LK flow
3. For stabilized: map the same initial corners through the stabilization homography (`H_corr_i`) to get their positions in stabilized frame i, then track from stabilized i to stabilized i+1
4. Filter: only use corners that land within the non-border valid region of the stabilized frame (gray > 2)
5. This ensures both raw and stabilized measurements start from geometrically corresponding points, eliminating selection bias while respecting the warp coordinate transform

#### M3: Border Validity Ratio (constraint)

```python
def border_validity_ratio(frame: np.ndarray) -> float:
    """Fraction of pixels with gray > 2."""
```

**Adaptive threshold**: Instead of fixed 0.85, report distribution stats and flag at multiple thresholds:
```json
"border_stats": {
    "min_validity": 0.91,
    "p5_validity": 0.93,
    "mean_validity": 0.97,
    "flagged_at_0.80": 0,
    "flagged_at_0.85": 0,
    "flagged_at_0.90": 1
}
```

#### M4: Flow-Weighted Alignment (diagnostic, optional)

```python
def flow_weighted_alignment(
    ref_gray: np.ndarray,
    cand_gray: np.ndarray,
    flow_magnitude: np.ndarray,
    center_crop_ratio: float = 0.5,
) -> float:
    """MAE weighted by 1/(1 + clip(flow_mag, 0, P95)).
    
    Flow magnitude clipped to P95 to reduce tail instability.
    """
```

### Output Format: `evaluation_report.json`

```json
{
  "method": "raft",
  "metrics_version": "2.0",
  "center_crop_ratios": [0.33, 0.5, 0.66],
  "primary_crop_ratio": 0.5,
  "per_frame": [
    {
      "frame_index": 1,
      "border_validity_ratio": 0.97,
      "is_valid": true,
      "stability_0.50": {
        "raw_intensity_mae": 8.2,
        "stab_intensity_mae": 4.1,
        "raw_gradient_mae": 3.5,
        "stab_gradient_mae": 2.0,
        "raw_combined": 0.023,
        "stab_combined": 0.012,
        "improvement": 0.48
      },
      "stability_0.33": { "...same fields..." },
      "stability_0.66": { "...same fields..." },
      "residual_motion": {
        "raw_magnitude": 4.1,
        "stab_magnitude": 1.8,
        "raw_median_disp": 3.8,
        "stab_median_disp": 1.5,
        "improvement": 0.56,
        "reliable": true,
        "inlier_ratio": 0.82,
        "quadrant_coverage": 4
      }
    }
  ],
  "summary": {
    "num_valid_frames": 19,
    "border_stats": { "min_validity": 0.91, "p5_validity": 0.93, "mean_validity": 0.97,
                       "flagged_at_0.80": 0, "flagged_at_0.85": 0, "flagged_at_0.90": 1 },
    "stability_0.50": {
      "avg_raw_combined": 0.025,
      "avg_stab_combined": 0.014,
      "avg_improvement": 0.44,
      "direction": "lower_is_better"
    },
    "stability_0.33": { "..." },
    "stability_0.66": { "..." },
    "residual_motion": {
      "avg_raw_magnitude": 4.5,
      "avg_stab_magnitude": 2.1,
      "avg_improvement": 0.53,
      "num_reliable": 17,
      "direction": "lower_is_better"
    }
  }
}
```

**Field naming convention**: `_mae` = lower is better. `improvement` = higher is better (positive = stabilization helped). All metric fields include `direction` in summary for unambiguous interpretation.

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Center crop still catches foreground | Inflated stability score | Low-flow proxy mask + report at 3 crop ratios |
| M2 sparse tracking unreliable on stabilized | False signal | Validity checks (inlier ratio, quadrant coverage); `reliable` flag |
| Illumination change between frames | M1 intensity bias | Gradient-domain metric as complement |
| Very small raw motion (already stable) | Division by ~0 in improvement ratio | eps guard; NaN for raw < eps |
| Border validity too strict/loose | Frames wrongly included/excluded | Report at 3 thresholds; let user decide |

---

## Part 3 — Test Plan

### Unit Tests (`tests/ego_video/test_stabilization_metrics.py`)

1. `test_adjacent_stability_identical` — returns 0.0 for both intensity and gradient
2. `test_adjacent_stability_monotonic_with_shift` — larger shift → larger MAE (within tolerance)
3. `test_adjacent_stability_excludes_border` — black regions don't contribute
4. `test_adjacent_stability_pairwise_mask` — uses intersection of both frames' valid regions
5. `test_gradient_domain_illumination_robust` — global brightness shift → gradient_mae ≈ 0
6. `test_residual_motion_zero_static` — identical frames → ~0 magnitude
7. `test_residual_motion_positive_shifted` — shifted frames → positive, monotonic with shift
8. `test_residual_motion_unreliable_flag` — blank frames → reliable=False
9. `test_residual_motion_same_corners` — raw and stab use same initial points
10. `test_border_validity_full` — no borders → 1.0
11. `test_border_validity_partial` — 20% black → ~0.8
12. `test_improvement_eps_guard` — raw ≈ 0 → NaN, not infinity
13. `test_multi_crop_ratios` — all three crop ratios in output
14. `test_flow_weighted_clips_p95` — extreme flow values clipped
15. `test_summary_excludes_nan_frames` — NaN/unreliable frames excluded from summary averages

---

## Part 4 — Rollout & Status

### Implementation Order
1. `src/vibelab/ego_video/motion/stabilization_metrics.py` — metric functions
2. Update `scripts/ego_video/analyze_motion.py` — add `evaluate` subcommand
3. Unit tests
4. Re-evaluate all existing analyses (affine2d + RAFT × 3 fps)

### Status
- [x] Design round 1 reviewed (gpt-5.3-codex)
- [x] Design round 2 reviewed (gpt-5.3-codex)
- [ ] Wayne approval
- [ ] Implementation
- [ ] Re-evaluation with corrected metrics

---

## Review Notes

### Round 1 — Reviewer: gpt-5.3-codex

**Major (3) — all addressed:**
1. ❌→✅ M1 sensitivity to illumination: **Added gradient-domain MAE** as complement to intensity MAE. Combined score uses equal-weight blend of normalized intensity + gradient metrics.
2. ❌→✅ Background mask definition inconsistent: **Defined two explicit tiers** — MVP uses `non_border ∩ center_crop ∩ low_flow_proxy`; V2 uses RANSAC inlier mask. Flow source explicitly specified (raw RAFT flow from estimation).
3. ❌→✅ M2 foreground pollution: **Added validity checks** — inlier ratio minimum, quadrant coverage check, `reliable` flag. Uses median/trimmed-mean displacement (robust stats). Same initial corners for raw and stabilized (eliminates selection bias).

**Minor (4) — all addressed:**
1. Center crop hardcoded: report at 0.33/0.50/0.66, primary=0.50.
2. Border validity threshold: adaptive reporting at 0.80/0.85/0.90 thresholds with distribution stats.
3. M4 flow weight truncation: clip to P95 before weighting.
4. Pairwise valid domain: explicitly documented intersection requirement for all pairwise metrics.

**Nit (3) — all addressed:**
1. eps guard for improvement ratio; NaN when raw < eps.
2. Direction annotation in field names and summary (`direction: "lower_is_better"`).
3. Shift test → monotonicity assertion with tolerance.

### Round 2 — Reviewer: gpt-5.3-codex

**Major (1) — addressed:**
1. ❌→✅ M2 corner coordinate space: **Added explicit coordinate mapping** — detect on raw, map through H_corr to stabilized space, filter by valid region. Both measurements start from geometrically corresponding points.

**Minor (2) — addressed:**
1. `max_grad` undefined: specified as per-frame max of valid gradient values (clamped ≥ 1.0).
2. Schema naming inconsistency: unified `border_stats` field names to `min_validity`/`p5_validity`/`mean_validity` everywhere.

**Nit (2) — addressed:**
1. Pseudocode variable naming: fixed.
2. Test for NaN aggregation: added to test plan.
