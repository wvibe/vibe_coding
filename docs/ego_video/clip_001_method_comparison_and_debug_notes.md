# clip_001 Method Comparison And Debug Notes

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

#### 1. DPVO pose quality at 30fps looks quantized

For `30fps`, many adjacent frames have identical `rotvec_raw` values.

Observed quick summary:

- `30fps`: `raw_exact_repeats = 13 / 19`
- `10fps`: `4 / 19`
- `3fps`: `0 / 19`

This suggests that with current inputs (`undistort + balance=0.0 + scale=0.5`), DPVO may be losing sensitivity to small inter-frame motion.

#### 2. There is a strong image-space mismatch hypothesis

Current DPVO logic is:

1. undistort raw fisheye frames to pinhole space
2. optionally downscale those frames for DPVO inference
3. estimate pose in pinhole space
4. compute rotation-only homography `H_corr = K @ R_corr @ K^{-1}` using pinhole intrinsics
5. reuse the generic homography stabilizer

But the generic stabilizer in
[raft_flow.py](/home/wei/vibe/vibe_coding/src/vibelab/ego_video/motion/raft_flow.py)
applies `H_corr` directly to the raw frames from the original `frame_dir`.

That means the current pipeline may be:

- computing correction in undistorted pinhole space
- applying that correction directly in raw fisheye space

If true, this is a larger integration issue than smoothing or metric choice, and it can easily explain why DPVO underperforms despite producing plausible pose magnitudes.

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

The next best diagnostic is not another install pass.

It is to test whether DPVO correction should be applied in undistorted pinhole space instead of directly on raw fisheye frames.

In parallel, it is worth checking whether `balance=0.0` + `dpvo_scale=0.5` is making the pose too quantized at `30fps`.
