# Ego-Video Next Steps

## Today's Goal

Turn the next stage into a decision step, not just another preview export.

The main question for today is:

- is step `3` still problematic because the current method is too weak
- or because the estimator is still being corrupted by foreground motion
- or because there is still an implementation bug in the stabilization path

The current default is to keep the classical route moving for one more focused pass, while making the escalation path to VO / SLAM concrete.

## Why Step 3 Is Still Problematic

Yesterday's work narrowed the problem, but did not fully resolve it.

What now looks mostly settled:

- the earlier visualization and sign-debugging issues were real
- those issues appear mostly corrected
- the `rotation3d` path is now internally consistent enough to keep as a baseline

What still looks weak:

- the visual improvement is modest
- dynamic foreground hands, tools, and machine parts can still dominate the tracked features
- fisheye geometry and motion blur still make sparse tracking brittle
- rotation-only compensation cannot remove translation-like parallax

The honest current reading is that the remaining issue is more likely estimator robustness or method limitation than a trivial wiring bug.

## Notebook 03 Plan

`03_stabilization_baseline.ipynb` should become a validation notebook.

Main tasks:

- compare `affine2d` and `rotation3d` on the same clip and same sampled frames
- compare raw, identity-rectified, and stabilized views side by side
- export one preview per method instead of one mixed result
- plot per-frame correction magnitude, inlier count, and motion traces
- inspect several bad windows from each clip, not only one representative frame

Default evaluation settings:

- use `cv2.BORDER_CONSTANT`
- enable the source outline marker
- trust center-third and center-half metrics more than full-frame metrics

Priority improvements to try before escalating:

1. foreground/background feature filtering
2. inlier and residual diagnostics
3. undistorted-motion comparison
4. keyframe or short-horizon reference comparison

The goal is not to prove the method is good. The goal is to make its failure mode easy to read.

## Notebook 04 Plan

`04_vo_slam_candidates.ipynb` should become a real selection notebook.

For each candidate, record:

- camera model and fisheye support
- monocular support and calibration requirements
- setup and build cost in this repo
- GPU or runtime cost
- expected robustness to dynamic foreground and head motion
- likely export format for trajectories and downstream comparison

Current default ranking:

1. `ORB-SLAM3`
2. `DROID-SLAM`
3. `OpenVSLAM`
4. `DPVO`

Current default first integration candidate:

- `ORB-SLAM3`

This remains the best first heavy baseline because it is the most natural fit for monocular fisheye data and gives a stronger classical trajectory reference than the current 2D or rotation-only pipeline.

## How We Tell Bug vs Weak Method

Treat the following cases differently:

- If inlier counts collapse or spike on hand-heavy frames and the motion traces become erratic, the implementation may be mostly fine but the estimator is being corrupted.
- If the traces look smooth and plausible but the warp still pushes trustworthy frames the wrong way, there is still an implementation problem.
- If the diagnostics stay internally consistent but both `affine2d` and `rotation3d` remain weak, the method is likely the real bottleneck.

Signals to trust more:

- center-crop comparisons
- per-frame inlier counts
- direct raw vs identity vs corrected frame diagnostics

Signals to trust less:

- full-frame metrics near the periphery
- visually clean previews that depend on reflect fill

## Exit Criteria

Stay on the classical route if:

- one method clearly improves center-crop stability on more than one clip
- the failure cases are understandable and limited
- the stabilization traces are interpretable enough to guide further iteration

Escalate to VO / SLAM sooner if:

- both `affine2d` and `rotation3d` remain weak after foreground filtering
- the center-crop results are still problematic
- the diagnostics suggest the method is internally consistent but fundamentally too limited

For now, keep `rotation3d` as the more principled baseline, but treat it as a debugging tool rather than a trusted solution.
