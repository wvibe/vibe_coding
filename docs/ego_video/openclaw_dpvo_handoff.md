# Openclaw / 5090 DPVO Handoff

## What Is Already Settled

1. DPVO host setup works on the 5090.
   The CUDA/installation issues are solved. See:
   - [5090_dpvo_setup_guide.md](/home/wei/vibe/vibe_coding/docs/ego_video/5090_dpvo_setup_guide.md)

2. The image-space bug was real and is already fixed.
   Before the fix, DPVO corrections were computed in undistorted pinhole space but applied to raw fisheye frames.
   That repo-side bug is fixed now:
   - DPVO stabilization happens in undistorted space
   - DPVO evaluation uses the undistorted raw baseline

3. That image-space fix was necessary, but it did not make DPVO competitive.
   So do not treat image-space mismatch as the remaining main issue.

## What Has Already Been Tried

### On `clip_001`

Already tested:

- baseline DPVO run
- image-space fix rerun
- higher-resolution DPVO input
- lower-memory DPVO config via [dpvo_p48.yaml](/home/wei/vibe/vibe_coding/configs/ego_video/dpvo_p48.yaml)

Important outcome:

- higher resolution reduced repeated `rotvec_raw` values, especially at `30fps`
- but the evaluation stayed negative overall

Interpretation:

- input quality / pose granularity is part of the story
- but it is not enough to explain the whole DPVO gap

Reference:

- [clip_001_method_comparison_and_debug_notes.md](/home/wei/vibe/vibe_coding/docs/ego_video/clip_001_method_comparison_and_debug_notes.md)

### On two additional Ego10K scenes

Also tested on:

- `factory085_worker007_00077`
- `factory020_worker002_00157`

Outcome:

- scene choice changes difficulty a lot
- `affine2d` and `raft` can become positive on the friendlier scene
- `dpvo` remained negative on both new scenes

Interpretation:

- the DPVO gap is not just a `clip_001` artifact
- the current issue generalizes across multiple tested scenes

Reference:

- [cross_scene_generalization_notes.md](/home/wei/vibe/vibe_coding/docs/ego_video/cross_scene_generalization_notes.md)

## Current Working Conclusion

The main unknown is no longer:

- "did we apply DPVO in the wrong image space?"
- "did we simply feed DPVO too low a resolution?"

Those have already been tested enough to move on.

The more likely remaining bottlenecks are now:

- DPVO pose quality after fisheye undistortion on this ego-video domain
- smoothing / correction policy after pose estimation
- or the conversion from DPVO pose to the current rotation-only image warp

## What Not To Repeat First

Unless there is a very specific reason, do not spend the next cycle mainly on:

- re-debugging installation
- re-debugging image-space mismatch
- another generic `0.75 -> 1.0 -> lower patches if OOM` sweep

Those paths have already been explored enough to establish that they are not the main blocker.

## Best Next Step

If you continue on DPVO, the highest-value next experiment is likely one of:

1. smoothing-radius / correction-policy sweeps on the better `30fps` cases
2. inspect `trajectory.txt`, `motion.json`, and `trajectory.json` on the friendlier scenes to see whether raw pose is plausible but over-corrected
3. revisit whether the current rotation-only homography projection is too lossy for the DPVO signal

## Short Summary

Current repo evidence says:

- DPVO runs
- DPVO install issues are solved
- the major image-space bug is fixed
- higher DPVO resolution helps pose granularity
- but DPVO is still negative on `clip_001` and on two additional scenes
- `raft` remains the strongest current baseline

So the next DPVO work should focus on correction behavior, not basic setup or another small resolution tweak.
