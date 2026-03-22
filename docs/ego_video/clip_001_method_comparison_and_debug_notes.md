# clip_001 Method Comparison And Debug Notes

Related follow-up:

- [cross_scene_generalization_notes.md](/home/wei/vibe/vibe_coding/docs/ego_video/cross_scene_generalization_notes.md)
  compares the same three methods on two additional Ego10K scenes to test whether the DPVO gap is specific to `clip_001`

## Scope

This note summarizes a like-for-like comparison on the same extracted frame set:

- frame set: `/data/datasets/ego_video/builddotai/frame_sets/clip_001`
- source window: `factory001_worker001_00000`, start `10.0s`
- rates: `30fps`, `10fps`, `3fps`
- frames per rate: `20`
- evaluation: `stabilization_metrics.py` metrics version `2.0`

Methods compared:

- `affine2d`
- `raft`
- `dpvo`

## High-Level Result

On this exact frame set, the current repo behavior is:

| Method | 30fps Stability | 30fps Motion | 10fps Stability | 10fps Motion | 3fps Stability | 3fps Motion |
|---|---:|---:|---:|---:|---:|---:|
| affine2d | `+9.7%` | `+19.2%` | `-6.0%` | `-73.5%` | `-21.7%` | `+22.2%` |
| raft | `-17.7%` | `-9.6%` | `+16.4%` | `+68.7%` | `+9.4%` | `+79.6%` |
| dpvo | `-11.8%` | `-33.6%` | `-11.2%` | `-32.4%` | `-11.8%` | `N/A` |

Interpretation:

- `affine2d` still helps at `30fps`, then degrades badly as inter-frame motion increases.
- `raft` still helps most clearly at `10fps`, and also helps at `3fps` on the current metrics.
- `dpvo` now runs end to end on the 5090 host, but the current integration degrades the metrics at all three rates on this clip.

## Comparison Against Earlier Conclusions

The same-frame-set reruns broadly support the earlier storyline:

- the classical sparse method is good for small motion and weak for larger motion
- the dense-flow method is the strongest current baseline around `10fps`
- the current DPVO result is not yet competitive

This matters because it suggests the negative DPVO outcome is not caused by a broken frame set or a broken metric.

## DPVO Image-Space Fix Rerun

The first DPVO pass exposed a valid integration bug:

1. raw fisheye frames were undistorted to pinhole space
2. DPVO estimated pose in that undistorted pinhole space
3. `H_corr = K @ R_corr @ K^{-1}` was computed in pinhole space
4. the generic stabilizer then applied that homography directly to raw fisheye frames

That bug has now been fixed in the repo:

- DPVO stabilization first undistorts raw fisheye frames into `analysis/.../_undistorted_for_stab`
- homographies are applied in that undistorted pinhole space
- DPVO evaluation now compares `stabilized/` against the undistorted raw baseline instead of the original fisheye frames

Only `stabilize + evaluate` were rerun. `estimate` outputs were reused.

### Rerun Result After The Image-Space Fix

| Method | 30fps Stability | 30fps Motion | 10fps Stability | 10fps Motion | 3fps Stability | 3fps Motion |
|---|---:|---:|---:|---:|---:|---:|
| dpvo (after fix) | `-11.9%` | `-40.5%` | `-11.4%` | `-19.5%` | `-10.9%` | `-42.3%` |

Interpretation:

- the image-space fix was correct and should stay
- but it did **not** recover DPVO to neutral or positive metrics on this clip
- so the previous `~-11%` pattern was not explained only by the image-space mismatch
- the remaining bottleneck is more likely pose quality / parameter choice than the warp target space alone

## DPVO Parameter Sweep After The Image-Space Fix

A follow-up parameter sweep was run on the same `clip_001` frame set.

Goals:

- test whether higher DPVO input resolution reduces the small-motion quantization seen at `30fps`
- test whether lower `PATCHES_PER_FRAME` can recover higher-resolution runs that otherwise OOM

Additional config used:

- low-memory config: [dpvo_p48.yaml](/home/wei/vibe/vibe_coding/configs/ego_video/dpvo_p48.yaml)
  - `PATCHES_PER_FRAME: 48`

### Sweep Summary

| Run | Stability | Motion | Notes |
|---|---:|---:|---|
| `30fps_dpvo` | `-11.9%` | `-40.5%` | baseline after image-space fix, `scale=0.5` |
| `30fps_dpvo_s075` | `-11.4%` | `-36.9%` | `scale=0.75`, default DPVO config |
| `30fps_dpvo_s100_p48` | `-10.2%` | `-32.5%` | `scale=1.0`, `PATCHES_PER_FRAME=48` |
| `10fps_dpvo` | `-11.4%` | `-19.5%` | baseline after image-space fix, `scale=0.5` |
| `10fps_dpvo_s075` | `-11.3%` | `-23.9%` | `scale=0.75`, default DPVO config |
| `3fps_dpvo` | `-10.9%` | `-42.3%` | baseline after image-space fix, `scale=0.5` |
| `3fps_dpvo_s075_p48` | `-8.2%` | `-10.5%` | `scale=0.75`, `PATCHES_PER_FRAME=48` |

### What changed

- `30fps`
  - higher input scale consistently helped, but only modestly
  - the best run in this sweep was `scale=1.0, p48`, still negative overall
- `10fps`
  - `scale=0.75` barely changed stability and made residual motion somewhat worse
  - this suggests resolution alone is not enough to explain the `10fps` gap
- `3fps`
  - `scale=0.75` needed the lower-memory config to run reliably
  - once it did run, the residual-motion result improved substantially relative to the baseline

### Pose quantization check

The higher-resolution runs clearly reduced the exact-repeat pose issue:

| Run | Exact repeats in `rotvec_raw` | Avg rotation magnitude |
|---|---:|---:|
| `30fps_dpvo` | `13 / 19` | `1.8102 deg` |
| `30fps_dpvo_s075` | `5 / 19` | `1.9914 deg` |
| `30fps_dpvo_s100_p48` | `2 / 19` | `1.8810 deg` |
| `10fps_dpvo` | `4 / 19` | `5.5662 deg` |
| `10fps_dpvo_s075` | `1 / 19` | `5.4771 deg` |
| `3fps_dpvo` | `0 / 19` | `9.0098 deg` |
| `3fps_dpvo_s075_p48` | `0 / 19` | `7.4623 deg` |

Interpretation:

- higher-resolution DPVO input is genuinely improving pose granularity, especially at `30fps`
- that means the earlier repeat-heavy trajectory was not just a logging artifact
- but the metric gains are still smaller than the pose-granularity gains, so there is likely another quality bottleneck after inference
- likely remaining candidates are the smoothing strategy, the rotation-only projection model, or the undistortion crop itself

## Identity Baseline Check

An explicit identity check was run by evaluating `raw` against itself.

For `30fps`, the summary returns:

- `stability_0.33.avg_improvement = 0.0`
- `stability_0.5.avg_improvement = 0.0`
- `stability_0.66.avg_improvement = 0.0`
- `residual_motion.avg_improvement = 0.0`

So the current evaluation stack is not obviously drifting when there is no correction.

## DPVO-Specific Findings

### What was required to make DPVO run

The 5090 host path is documented in:

- [5090_dpvo_setup_guide.md](/home/wei/vibe/vibe_coding/docs/ego_video/5090_dpvo_setup_guide.md)

Key practical points:

- DPVO was installed into the existing `ego` env
- local compatibility fixes were needed for the host's `torch 2.9.1`
- `torch-scatter` had to be built locally
- `--dpvo-scale 0.5` was added to avoid OOM on `1920x1080` image-directory inputs
- DPVO frame-index timestamps had to be normalized to seconds before alignment

### Current DPVO artifact paths

- `30fps`: `/data/datasets/ego_video/builddotai/frame_sets/clip_001/analysis/30fps_dpvo`
- `10fps`: `/data/datasets/ego_video/builddotai/frame_sets/clip_001/analysis/10fps_dpvo`
- `3fps`: `/data/datasets/ego_video/builddotai/frame_sets/clip_001/analysis/3fps_dpvo`

Important files:

- `motion.json`
  What it answers:
  raw DPVO-derived per-frame corrections, raw/smoothed rotvecs, calibration used by inference
- `trajectory.txt`
  What it answers:
  the raw trajectory exported from DPVO in TUM format
- `trajectory.json`
  What it answers:
  the correction actually fed into stabilization
- `transforms.json`
  What it answers:
  the per-frame homographies applied by the current pipeline
- `evaluation_report.json`
  What it answers:
  per-frame and aggregate metric outcomes

### Current DPVO clues worth investigating

#### 1. DPVO pose quality at 30fps was quantized, and higher resolution helps

For `30fps`, many adjacent frames have identical `rotvec_raw` values.

Observed quick summary:

- `30fps`: `raw_exact_repeats = 13 / 19`
- `10fps`: `4 / 19`
- `3fps`: `0 / 19`

This suggests that with current inputs (`undistort + balance=0.0 + scale=0.5`), DPVO may be losing sensitivity to small inter-frame motion.

The parameter sweep strengthened that diagnosis:

- `scale=0.75` reduced `30fps` repeats from `13/19` to `5/19`
- `scale=1.0` with `PATCHES_PER_FRAME=48` reduced them further to `2/19`

So pose quantization is a real issue, and higher-resolution inference is one valid lever for addressing it.

#### 2. The image-space mismatch was real, but not the whole story

The first DPVO pass really did have a space mismatch:

- correction was computed in undistorted pinhole space
- the generic stabilizer applied it to raw fisheye frames

That path has now been fixed, and the rerun still stays negative.

So this item moves from:

- "main hypothesis to test"

to:

- "confirmed bug that was worth fixing, but not sufficient to explain the remaining DPVO gap"

#### 3. K-consistency itself does not look obviously broken

`motion.json` records both:

- `calibration_pinhole`: undistorted full-resolution pinhole intrinsics
- `calibration_dpvo`: those same intrinsics scaled by `0.5` for DPVO input

These are internally consistent with the current inference path.

## Artifact Reading Guide

### If you want to judge pose quality first

Look at:

- `trajectory.txt`
- `motion.json` fields:
  - `rotvec_raw`
  - `rotation_magnitude_deg`
  - `position`
  - `quaternion`

Questions:

- are poses smooth or step-like?
- do adjacent frames repeat too often?
- do magnitudes scale sensibly from `30fps -> 10fps -> 3fps`?

### If you want to judge smoothing/correction behavior

Look at:

- `trajectory.json`

Questions:

- how far is `rotvec_smooth` from `rotvec_raw`?
- are early frames getting non-trivial correction despite zero-ish raw motion?
- is the correction dominated by border-inducing image motion?

### If you want to judge image-space validity

Look at:

- `trajectory.json`
- `motion.json`
- the implementation in:
  - [dpvo_bridge.py](/home/wei/vibe/vibe_coding/src/vibelab/ego_video/motion/dpvo_bridge.py)
  - [raft_flow.py](/home/wei/vibe/vibe_coding/src/vibelab/ego_video/motion/raft_flow.py)

Questions:

- is the correction being applied in the same image space in which it was estimated?
- if DPVO lives in undistorted pinhole space, should stabilization also happen there before any back-projection?

### If you want to judge final usefulness only

Look at:

- `evaluation_report.json`
- `motion_summary.json`
- `report` CLI output

Most comparable fields:

- `summary.stability_0.5.avg_improvement`
- `summary.residual_motion.avg_improvement`
- border validity stats

## Recommended Next Diagnostic Step

The next best diagnostic is now a parameter-quality pass, not another integration pass.

Recommended order:

1. try higher DPVO input resolution by rerunning `estimate` with `--dpvo-scale 0.75` and, if memory allows, `--dpvo-scale 1.0`
2. if memory becomes the blocker, reduce `PATCHES_PER_FRAME` in the DPVO config and retry
3. re-check whether the `30fps` raw pose still shows many exact repeats after the higher-resolution run

The current evidence points more strongly to:

- `balance=0.0` + low DPVO input scale reducing feature quality
- short-sequence smoothing acting on already quantized pose estimates

After the latest sweep, an additional candidate also stands out:

- the stabilization projection itself may still be too lossy even when pose granularity improves
