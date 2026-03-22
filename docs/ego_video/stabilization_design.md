# Ego-Video Stabilization Notes

## Goal

This note documents the stabilization experiments in
[`notebooks/ego_video/02_motion_baseline.ipynb`](/home/wei/vibe/vibe_coding/notebooks/ego_video/02_motion_baseline.ipynb)
and the current implementation in
[`src/vibelab/ego_video/motion/stabilize.py`](/home/wei/vibe/vibe_coding/src/vibelab/ego_video/motion/stabilize.py).

The project goal is not to solve full ego-motion estimation in one step. The immediate goal is to build a minimal, inspectable stabilization baseline for egocentric fisheye video and understand where it fails.

## What Was Implemented

### 1. 2D Classical Baseline

The first baseline uses a standard 2D video-stabilization recipe:

1. Detect sparse corners with `cv2.goodFeaturesToTrack`.
2. Track them frame to frame with `cv2.calcOpticalFlowPyrLK`.
3. Fit a global 2D similarity transform with `cv2.estimateAffinePartial2D`.
4. Accumulate the frame-to-frame motion into a camera path.
5. Smooth the path with a moving average.
6. Warp each frame back with the estimated correction.

This is fast and easy to inspect, but it treats the image as a planar signal and ignores camera intrinsics. On fisheye ego video with strong foreground motion, that is often too weak.

### 2. Rotation-Aware 3D Baseline

The newer baseline moves one step closer to camera-extrinsic stabilization:

1. Use the clip calibration from `clip_manifest.json`.
2. Track sparse correspondences between adjacent frames.
3. Undistort the correspondences into normalized camera coordinates.
4. Estimate an essential matrix with `cv2.findEssentialMat`.
5. Recover relative camera rotation with `cv2.recoverPose`.
6. Accumulate relative rotations across the clip.
7. Smooth the accumulated rotation trajectory in Rodrigues vector space.
8. Convert the smoothed-vs-raw difference into a per-frame correction rotation.
9. Apply the correction through intrinsics-aware rectification maps.

This path is implemented in:

- `estimate_camera_rotations(...)`
- `compute_rotation_stabilization_trace(...)`
- `stabilize_frame_3d_rotation(...)`
- `stabilize_video_3d_rotation(...)`

## Important Clarification

Even though we refer to this as a 3D / extrinsic-style baseline, the current implementation is still **rotation-only** in the physically meaningful sense.

Why:

- A monocular video without depth does not provide a globally valid per-pixel 3D reprojection for translation.
- Camera translation induces depth-dependent parallax.
- A single homography-like warp can approximate pure rotation well, but not general translation in scenes with depth variation.

So the current "3D" version should really be read as:

- use camera intrinsics
- use relative camera rotation
- stabilize by compensating the rotational part of the ego motion

It is **not** a full 6DoF scene reprojection system.

## Why The Current Results May Still Look Wrong

Several failure modes are expected at this stage.

### 1. Dynamic Foreground Corrupts Motion Estimation

The worker's hands, manipulated parts, and machine motion can dominate the tracked features. This can bias both:

- the 2D global transform
- the essential-matrix pose estimate

If foreground points are not filtered, the recovered motion may describe hand motion more than camera motion.

### 2. Fisheye Geometry Is Hard

The dataset uses fisheye intrinsics and distortion coefficients. Even after point undistortion, image formation remains challenging near the periphery. A simple sparse-feature pipeline is still brittle under strong distortion and motion blur.

### 3. Rotation Can Improve While Translation Metrics Get Worse

This was already observed in quantitative checks:

- average and tail rotation magnitude can go down
- average translation magnitude can still go up

This suggests the baseline is partially correcting head rotation while introducing image-plane drift elsewhere.

### 4. Visual Judgment Can Be Misleading

Two border strategies were tested:

- `cv2.BORDER_REFLECT`
- `cv2.BORDER_CONSTANT`

Reflect fill can make the video look visually cleaner while hiding the true transformed image footprint. For debugging, `BORDER_CONSTANT` plus a bright pre-warp outline is more honest.

## Current Notebook Diagnostics

The notebook now contains several debugging views:

### Frame-Level Compensation View

This section compares:

- reference frame
- raw frame
- identity rectified frame
- apply `+corr`
- apply `-corr`

For easier inspection:

- the raw/reference frames can be outlined
- corrected frames can warp an outlined source border
- border mode can be forced to constant black

This helps answer:

- is the correction direction flipped?
- is the visible change only coming from fisheye rectification?
- is the transformed source footprint plausible?
- does the corrected frame align better with the chosen reference?

### Latest Reading Of The Diagnostics

By the end of the session, the frame-level diagnostic had become much more trustworthy than the earlier versions.

Important changes:

- the frame-level view now follows the same stabilization model as the preview view
- the debug marker is an inset rectangle with corner dots, not a border drawn on the extreme image edge
- the diagnostic includes an `identity rectified` column

These changes matter because the old edge-border marker could be pushed out of the visible rectified image, which made it look as if no correction had been applied.

Current interpretation:

- the `identity rectified` column explains why columns 3 to 5 all look different from the raw fisheye columns
- yes, those columns already include fisheye-aware rectification, even before any extra stabilization correction is applied
- after making the comparison more apples-to-apples, `+corr` looked more plausible than `-corr` on the final diagnostic grid
- the notebook therefore now defaults preview generation to `STABILIZATION_SIGN=+1.0`, while still leaving the sign configurable for debugging

This suggests that the earlier preference for `-corr` was at least partly caused by misleading visualization, not necessarily by the underlying correction sign.

That does **not** prove that the current `rotation3d` result is good. It only means:

- there is no longer an obvious sign bug in the diagnostic itself
- the current remaining issue is more likely to be method limitation than a trivial wiring mistake

### Why The Effect Still Looks Small

Even when the 3D path is wired correctly, the visual effect can still be subtle.

Reasons:

1. The recovered per-frame rotations can be numerically small for many frames.
2. Rotation-only compensation does not remove translation-like parallax from hands, tools, and nearby machine parts.
3. The stabilized view is being compared against a rectified image, not against the original fisheye image.

So if the current result feels like "mostly rectification plus a modest extra adjustment", that is consistent with the current method.

### Raw-vs-Stabilized Motion Comparison

The same classical motion estimator is run on:

- the raw video
- the stabilized video

for several crops:

- full frame
- center third
- center half

The center crops are more trustworthy because border fill and extreme fisheye periphery can bias full-frame estimates.

## Why The User Was Right To Be Skeptical

There were two real issues during development:

1. The early frame-diagnostic view still used the old 2D compensation path even after the preview path moved to the 3D rotation model.
2. Preview files could be stale because generated browser previews were being reused.
3. The original debug border was drawn on the extreme image boundary, which made the 3D rectification effect hard to interpret.

Those issues have been corrected, but they are a good reminder that stabilization work needs image-level and metric-level validation, not just a single exported preview.

## Recommended Next Steps

### Short-Term

1. Compare 2D vs `rotation3d` on the same frames and the same clip.
2. Evaluate with `BORDER_CONSTANT` and outline enabled.
3. Trust center-crop metrics more than full-frame metrics.
4. Separate rotation-only and translation-like behavior in plots.

### Medium-Term

1. Reject dynamic foreground features more aggressively.
   Ideas:
   - mask central hands by heuristic regions
   - reject large residual tracks
   - keep only RANSAC-consistent background tracks
2. Undistort the full frame before motion estimation and compare.
3. Use a keyframe/reference-frame stabilization strategy instead of only local frame-to-frame accumulation.

### Longer-Term

1. Move to a stronger VO / SLAM baseline:
   - ORB-SLAM3
   - OpenVSLAM
   - DROID-SLAM / DPVO
2. Use those trajectories for comparison against the lightweight baseline.

## Practical Conclusion

The current implementation is useful as a debugging baseline, not yet as a trusted stabilization solution.

The 2D path is fast and interpretable.
The 3D rotation path is more principled for head rotation and fisheye intrinsics.
Neither is yet robust enough to claim success on this ego-video clip.

The most honest current summary is:

- the notebook no longer shows an obvious implementation bug in the visualization path
- the 3D rotation pipeline appears internally consistent enough to keep as a baseline
- but the improvement is modest, and the method is still limited by foreground motion, monocular ambiguity, and rotation-only compensation

That is still a useful result: it narrows the next engineering step from "try random stabilization tweaks" to "improve rotation estimation and background feature selection, or escalate to VO / SLAM."

## Final Notebook Cleanup

The notebook has also been simplified so the verbose frame-level compensation logic now lives in native Python under
[`src/vibelab/ego_video/motion/diagnostics.py`](/home/wei/vibe/vibe_coding/src/vibelab/ego_video/motion/diagnostics.py).

That keeps the notebook focused on:

- choosing the clip
- setting diagnostic parameters
- plotting the returned comparison grid
- exporting preview videos
