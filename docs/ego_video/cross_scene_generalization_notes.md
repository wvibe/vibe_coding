# Ego-Video Cross-Scene Generalization Notes

## Scope

This note records a follow-up comparison on two additional Ego10K scenes, using the same extraction protocol as `clip_001`.

Goal:

- test whether the negative DPVO result is mostly a `clip_001` artifact
- compare `affine2d`, `raft`, and `dpvo` on clearly different scenes

Frame sets:

- `/data/datasets/ego_video/builddotai/frame_sets/factory085_worker007_00077`
- `/data/datasets/ego_video/builddotai/frame_sets/factory020_worker002_00157`

Source videos:

- `/data/datasets/ego_video/builddotai/ego10k_samples/scene_scout/factory085_worker007_part07_noxet/videos/factory085_worker007_00077.mp4`
- `/data/datasets/ego_video/builddotai/ego10k_samples/scene_scout/factory020_worker002_part14_noxet/videos/factory020_worker002_00157.mp4`

Preview strips:

- [factory085_worker007_00077_strip.jpg](/data/datasets/ego_video/builddotai/ego10k_samples/scene_scout/previews/factory085_worker007_00077_strip.jpg)
- [factory020_worker002_00157_strip.jpg](/data/datasets/ego_video/builddotai/ego10k_samples/scene_scout/previews/factory020_worker002_00157_strip.jpg)

Extraction protocol:

- start at `10s`
- extract `20` frames at `30fps`, `10fps`, `3fps`
- evaluate with `stabilization_metrics.py` metrics version `2.0`

Methods compared:

- `affine2d`
- `raft`
- `dpvo`

## High-Level Result

Scene choice clearly matters, but it does **not** overturn the main DPVO conclusion.

- `factory020_worker002_00157` is much friendlier to stabilization
- `factory085_worker007_00077` is difficult for all methods
- `affine2d` and `raft` can still become positive on the friendlier scene
- `dpvo` remains negative on both new scenes

That means the current DPVO gap is not just a bad-luck property of `clip_001`.

## Result Tables

### Scene A: `factory085_worker007_00077`

| Method | 30fps Stability | 30fps Motion | 10fps Stability | 10fps Motion | 3fps Stability | 3fps Motion |
|---|---:|---:|---:|---:|---:|---:|
| affine2d | `-0.2%` | `-25.4%` | `-0.3%` | `-23.2%` | `-2.9%` | `N/A` |
| raft | `-0.5%` | `-49.4%` | `+2.7%` | `-77.5%` | `+5.1%` | `N/A` |
| dpvo | `-12.3%` | `-33.0%` | `-11.5%` | `-38.4%` | `-12.9%` | `N/A` |

Quick read:

- this scene is hard overall
- `raft` gets some positive stability at `10fps` and `3fps`, but residual motion still looks poor
- `dpvo` stays consistently negative

### Scene B: `factory020_worker002_00157`

| Method | 30fps Stability | 30fps Motion | 10fps Stability | 10fps Motion | 3fps Stability | 3fps Motion |
|---|---:|---:|---:|---:|---:|---:|
| affine2d | `+2.1%` | `+26.6%` | `-17.4%` | `N/A` | `-2.0%` | `N/A` |
| raft | `+7.0%` | `+28.2%` | `+7.3%` | `N/A` | `+1.4%` | `N/A` |
| dpvo | `-23.8%` | `-43.0%` | `-6.4%` | `-256.1%` | `-9.7%` | `N/A` |

Quick read:

- this scene is meaningfully easier than Scene A
- `affine2d` helps at `30fps`
- `raft` is the strongest baseline here, especially at `30fps` and `10fps`
- `dpvo` still degrades the result across all three rates

## DPVO Runtime Notes

The current repo-side DPVO fixes remained enabled:

- undistorted-space stabilization and evaluation
- timestamp normalization
- host-side low-memory config via [dpvo_p48.yaml](/home/wei/vibe/vibe_coding/configs/ego_video/dpvo_p48.yaml)

Current parameter choices used here:

- Scene A:
  - `30fps`: `scale=1.0`, `PATCHES_PER_FRAME=48`
  - `10fps`, `3fps`: `scale=0.75`, `PATCHES_PER_FRAME=48`
- Scene B:
  - the first `30fps scale=1.0` attempt OOMed because an old DPVO process was still occupying GPU memory
  - after clearing that stale process, Scene B was completed with `scale=0.75`, `PATCHES_PER_FRAME=48` for all three rates

Important note:

- on Scene A `30fps`, `stabilization_report.json` briefly looked mildly positive
- but the canonical `evaluation_report.json` remained negative
- for cross-run comparison, `evaluation_report.json` should be treated as the source of truth

## What This Changes

These new scenes refine the earlier DPVO diagnosis in a useful way:

1. Scene dependence is real.
   The same method can look much better or much worse depending on the video.

2. But DPVO is still not just failing on one unlucky clip.
   It stayed negative on:
   - `clip_001`
   - `factory085_worker007_00077`
   - `factory020_worker002_00157`

3. The stronger baselines still behave sensibly.
   On the friendlier scene, `affine2d` and `raft` both improved the metrics, which argues against the frame sets or evaluation stack being the root cause.

4. OC's parameter-tuning suggestion still has partial value, but it is no longer the whole story.
   We already verified on `clip_001` that:
   - higher resolution reduces repeated DPVO poses
   - image-space mismatch was a real bug and is now fixed
   - those fixes are not enough by themselves to make DPVO competitive

So the remaining bottleneck is more likely:

- DPVO pose quality after undistortion on this ego-video domain
- smoothing / correction policy after inference
- or a mismatch between the VO signal and the current rotation-only image warp model

## Discussion Summary

This note also captures the practical conclusion from the recent back-and-forth on next steps.

### What from OC's feedback still looks useful

- checking for repeated `rotvec_raw` values is still a good quick pose-quality diagnostic
- scene diversity was worth testing, and did reveal meaningful variation in method behavior
- higher DPVO input resolution still has diagnostic value because it reduces pose quantization

### What has now been de-prioritized

- repeating the generic `0.75 -> 1.0 -> lower patches if OOM` sweep as the main next step

Why:

- that sweep was already run on `clip_001`
- it improved DPVO pose granularity but did not flip the method positive
- after that, two additional scenes were tested and DPVO still stayed negative overall

So the current repo evidence does not support the idea that a small additional resolution win is likely to change the qualitative conclusion.

### Current working position

The main repo-side unknown is no longer:

- "can we make DPVO a bit sharper by feeding it better input?"

It is now closer to:

- "assuming DPVO pose is somewhat plausible, are we over-correcting or projecting it into image space in a way that loses the benefit?"

That is why the next higher-value experiments are likely to be:

- smoothing-radius sweeps
- correction-policy changes
- or a closer inspection of how DPVO pose is converted into the current rotation-only homography stabilization path

## Practical Takeaway

The current evidence supports this working summary:

- scene selection changes magnitude
- but not the direction of the DPVO outcome so far
- `raft` remains the strongest baseline on the tested scenes
- `affine2d` still helps on some smaller-motion cases
- `dpvo` is not yet validated as useful under the current pipeline

If more DPVO work is done next, the most useful follow-up is probably not another generic resolution sweep.

It is more likely one of:

- smoothing-radius or correction-policy sweeps on the better `30fps` cases
- inspecting raw `trajectory.txt` / `motion.json` on the friendlier scene to see whether the poses themselves are plausible but being over-corrected
- revisiting whether a rotation-only homography is too lossy for the DPVO signal being estimated
