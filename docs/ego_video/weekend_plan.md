# Ego Video Weekend Plan

## Goal

Build a minimum viable ego-centric video analysis loop inside this repo and use it to decide which motion-estimation route is worth deeper investment.

This weekend's success criteria:

- Download a very small local sample set from Ego10K.
- Read sample videos, inspect fisheye intrinsics, and render raw vs undistorted key frames.
- Select one representative source video and export one or more shorter clips for fast iteration.
- Run a lightweight motion baseline based on feature tracking and frame-to-frame global motion.
- Export one stabilized video result for qualitative inspection.
- Leave a clear roadmap for the next stage: classical baseline vs VO / SLAM.

## Current Status

### 1. Dataset Peek

Status: mostly done

- Small-sample download works.
- Camera intrinsics and fisheye parameters are visible.
- Browser-friendly preview generation works.
- Raw vs undistorted key-frame comparison works.

### 2. Video IO Baseline

Status: done

- Video metadata inspection is done.
- Frame-level visualization is done.
- Clip extraction is done.

Current clip set:

- `factory001_worker001_00000_s0010_d020`
- `factory001_worker001_00000_s0055_d020`
- `factory001_worker001_00000_s0100_d020`

Verified properties:

- each clip is `20.0` seconds
- each clip is `600` frames at `30 FPS`
- geometry is preserved at `1920x1080`
- browser-friendly previews were generated
- source calibration is preserved in `clip_manifest.json`

Decision:

- We should cut the original long video into shorter clips before serious motion experiments.
- The full 7-minute source video is useful as the master asset.
- The actual motion-estimation notebooks should operate on shorter clips.
- The current working set is three `20` second debug clips.
- If needed later, we can add one longer `30 to 60` second evaluation clip.

Why:

- Iteration is much faster.
- Stabilization and global-motion plots are easier to inspect.
- Later VO / SLAM comparisons become more controlled.

## Scope

In scope:

- Small-sample data download only.
- Local video IO, key-frame peek, clip extraction.
- Camera metadata parsing and fisheye-aware visualization.
- Classical baselines with OpenCV:
  - corner detection
  - sparse optical flow
  - affine or similarity global motion
  - moving-average stabilization
- Planning for later VO / SLAM baselines.

Out of scope for this MVP stage:

- Full dataset download.
- Large learned training runs.
- Deep integration of heavy third-party SLAM repos this weekend.

## Route Map

### Route A: Classical Motion Estimation And Stabilization

This is the first route to implement end-to-end.

Questions it answers:

- Can we estimate useful camera motion from mostly static background?
- Are we dealing with mostly 2D jitter or something that clearly requires 6DoF trajectory modeling?
- How much do moving hands, tools, or machine parts break the estimate?

Methods:

- `cv2.goodFeaturesToTrack`
- `cv2.calcOpticalFlowPyrLK`
- `cv2.estimateAffinePartial2D`
- optional homography comparison
- moving-average camera-path smoothing
- `cv2.warpAffine` stabilization

Expected outputs:

- tracked feature overlay
- per-frame `dx`, `dy`, `dtheta`
- smoothed motion trace
- stabilized clip
- notes on failure cases

### Route B: VO / SLAM Candidates

This is the second route, after the classical baseline is stable enough to serve as a reference.

Priority order:

1. `ORB-SLAM3`
2. `OpenVSLAM`
3. `DROID-SLAM`
4. `DPVO` as a lightweight learned VO comparison

Rationale:

- `ORB-SLAM3` fits monocular fisheye best among the classical SLAM options.
- `OpenVSLAM` is also fisheye-friendly and is a good engineering comparison.
- `DROID-SLAM` is strong but heavier and should come after we understand the data better.
- `DPVO` is interesting as a learned VO baseline, but not the first system to wire in.

Questions it answers:

- Do we need a full trajectory estimator instead of a 2D stabilization baseline?
- How robust are these systems to head motion, motion blur, and dynamic foreground hands?
- Does calibration-aware modeling materially improve the result?

## Execution Order

### 1. Dataset Peek

- Keep only one representative source video for now.
- Record camera intrinsics and distortion parameters.
- Compare raw vs undistorted key frames.

### 2. Clip Preparation

- Completed.
- Source asset: the first `7` minute sample from `factory_001 / worker_001`.
- Exported three `20` second clips for downstream experiments.
- Stored in `/data/datasets/ego_video/builddotai/ego10k_clips/first_sample/`.
- Calibration and source metadata preserved in `clip_manifest.json`.

### 3. Classical Motion Notebook

- Next active step.
- Track sparse features on the raw clip.
- Repeat on undistorted frames if needed.
- Compare robustness under moving foreground objects.
- Save per-frame motion estimates.

### 4. Stabilization Notebook

- Smooth the estimated trajectory.
- Generate a stabilized clip.
- Compare raw and stabilized outputs visually.
- Note where stabilization helps and where it fails.

### 5. VO / SLAM Planning Notebook

- Document what inputs each candidate system needs.
- Record expected setup cost, GPU cost, calibration support, and likely risks.
- Decide which one to integrate first after the classical baseline.

## Notebook Lineup

- `00_dataset_peek.ipynb`
  - sample download, preview, intrinsics, raw vs undistorted key frames
- `01_visualize_clips.ipynb`
  - source-video selection and clip extraction
- `02_motion_baseline.ipynb`
  - sparse tracking, optical flow, global motion estimation
- `03_stabilization_baseline.ipynb`
  - trajectory smoothing and stabilized-video export
- `04_vo_slam_candidates.ipynb`
  - ORB-SLAM3 / OpenVSLAM / DROID-SLAM / DPVO setup notes and comparison plan

## Risks

- Fisheye distortion may destabilize naive affine motion estimates.
- Dynamic foreground hands or tools may dominate the tracked features.
- A stabilized-looking video may still not correspond to a meaningful 3D head-motion estimate.
- VO / SLAM systems may have substantial setup cost before any useful comparison.

## Decision Gates

Before integrating heavy VO / SLAM repos, answer these with the classical baseline:

- Does undistortion materially improve tracking?
- Can the background dominate enough features to estimate stable motion?
- Are the resulting `dx`, `dy`, `dtheta` traces interpretable?
- Is simple stabilization already good enough for the near-term use case?

If the answer is mostly yes, Route A continues.
If the answer is mostly no, escalate to Route B sooner.
