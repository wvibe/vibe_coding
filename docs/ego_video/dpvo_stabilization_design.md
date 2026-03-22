# DPVO 6DOF Visual Odometry Stabilization — Design Doc

## Part 1 — PRD

### Persona
Wayne (ML engineer) + Collie (AI coding assistant). Development on Mac mini M4, production inference on 5090 Ubuntu. Code synced via Git.

### Problem
Current best stabilization (RAFT + homography) works well at 10-3fps but fails at 30fps due to model overfitting. Fundamental limitation: we're doing 2D image-space compensation for what is actually a 3D camera motion problem. DPVO solves this by directly estimating the 6DOF camera trajectory.

### What is DPVO
**Deep Patch Visual Odometry** (NeurIPS 2023, Princeton Vision Lab):
- Input: image sequence + camera intrinsics
- Output: per-frame 6DOF camera pose (position xyz + orientation quaternion)
- Method: learned patch-based visual odometry with differentiable bundle adjustment
- Speed: real-time on single GPU
- Updated July 2024 with DPV-SLAM backend for loop closure

**Why DPVO over alternatives:**
| | DPVO | DROID-SLAM | ORB-SLAM3 |
|---|---|---|---|
| Speed | Real-time | 3-5× slower | Real-time |
| Accuracy | SOTA VO | SOTA SLAM | Good classical |
| Dependencies | lietorch, CUDA | lietorch, CUDA | C++ build, no Python |
| Maintenance | Active (2024) | Maintenance mode | Mature |
| Output | Poses + points | Poses + dense map | Poses + sparse map |
| Our use case fit | ✅ Perfect | Overkill (full SLAM) | Hard to integrate |

### Development & Deployment Model

**One codebase, two machines, Git sync:**

```
Mac mini M4 (development)                    5090 Ubuntu (production)
┌──────────────────────────┐                ┌──────────────────────────┐
│ Write code + tests       │   git push     │ git pull                 │
│ Run tests (CPU/mock)     │ ──────────────►│ Run full pipeline (CUDA) │
│ Review results           │ ◄──────────────│ git push (results)       │
│ Iterate                  │   git pull     │                          │
└──────────────────────────┘                └──────────────────────────┘
```

- Same CLI, same scripts, same code paths
- `--method dpvo` works identically to `--method raft` and `--method affine2d`
- On Mac: DPVO import guarded, tests use mock/CPU where possible
- On 5090: full CUDA inference with `lietorch` + DPVO

### User Stories

**US-1: Estimate motion with DPVO (run on 5090)**
```
$ python scripts/ego_video/analyze_motion.py estimate \
    --frame-set ./frame_sets/clip_001 \
    --fps 30 \
    --method dpvo \
    --output-dir ./analysis/30fps_dpvo
```
→ Produces same structure as other methods:
```
analysis/30fps_dpvo/
  motion.json           # per-frame 6DOF pose + rotation correction
  transforms.json       # cumulative 3×3 correction homographies
  trajectory.json       # smoothed trajectory + H_corr per frame
  motion_summary.json   # aggregate stats
```

**US-2: Stabilize (same machine or either machine)**
```
$ python scripts/ego_video/analyze_motion.py stabilize \
    --frame-set ./frame_sets/clip_001 \
    --fps 30 \
    --output-dir ./analysis/30fps_dpvo
```

**US-3: Evaluate (same as all methods)**
```
$ python scripts/ego_video/analyze_motion.py evaluate \
    --frame-set ./frame_sets/clip_001 \
    --fps 30 \
    --analysis-dir ./analysis/30fps_dpvo
```

### DPVO Input Requirements

**Images:** Sequential PNG/JPG frames (from our `prepare_frames.py`).

**Calibration:** DPVO requires `fx fy cx cy [k1 k2 k3 k4]` in radtan model.

**⚠️ Distortion model compatibility:**
- Our Ego10K data uses OpenCV fisheye (equidistant) model
- DPVO expects radial-tangential (radtan) model
- These are **incompatible parametrizations**

**Solution: pre-undistort frames** before DPVO inference:
1. Load raw fisheye frame
2. Apply `cv2.fisheye.undistortImage` using manifest calibration
3. Compute new pinhole intrinsics (`cv2.fisheye.estimateNewCameraMatrixForUndistortRectify`)
4. Feed undistorted frames + pinhole-only calibration (fx, fy, cx, cy, no k params) to DPVO

This adds ~2s preprocessing but guarantees correct calibration.

### DPVO Output: TUM Trajectory Format

```
# timestamp tx ty tz qx qy qz qw
0.000000 0.000 0.000 0.000 0.000 0.000 0.000 1.000
0.033333 0.012 -0.003 0.001 0.001 -0.002 0.000 1.000
```

### Trajectory-Frame Alignment Contract

1. `estimate` step generates a `frame_index_map.json` mapping each image filename to `{frame_index, timestamp_sec}`
2. After DPVO inference, trajectory is aligned by timestamp (nearest-neighbor, tolerance = 0.5 / source_fps)
3. Frames without a matching pose → identity correction
4. If < 50% frames matched → error with diagnostic
5. Quaternion sign continuity: enforce `q · q_prev > 0`
6. Stride > 1: interpolate via SLERP (quaternion) + linear (position) for intermediate frames

---

## Part 2 — Technical Design

### New Module: `src/vibelab/ego_video/motion/dpvo_bridge.py`

```python
def estimate_dpvo_motion(
    frame_dir: Path,
    calibration: dict,
    output_dir: Path,
    manifest_fs: dict | None = None,
    dpvo_model_path: str = "dpvo.pth",
    dpvo_config: str = "config/default.yaml",
    stride: int = 1,
    device: str = "cuda",
) -> dict:
    """Run DPVO inference on frame sequence, return motion.json dict.
    
    Steps:
    1. Pre-undistort frames if calibration is fisheye
    2. Write undistorted frames + pinhole calib to temp dir
    3. Run DPVO inference (calls dpvo.dpvo.DPVO directly via Python API)
    4. Parse output poses into per-frame motion data
    5. Convert poses to rotation-only corrections
    6. Return motion.json schema dict
    
    Raises ImportError if lietorch/dpvo not installed (caught by CLI).
    """
```

**DPVO Python API integration** (from demo.py):
```python
from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.stream import image_stream

# Initialize
slam = DPVO(cfg, network_path, ht=H, wd=W, viz=False)

# Feed frames
for t, image, intrinsics in stream:
    image = torch.from_numpy(image).permute(2,0,1).cuda()
    intrinsics = torch.from_numpy(intrinsics).cuda()
    slam(t, image, intrinsics)

# Get poses
poses, tstamps = slam.terminate()
# poses: Nx7 (tx, ty, tz, qx, qy, qz, qw)
```

**Import guard for Mac development:**
```python
try:
    from dpvo.dpvo import DPVO
    from dpvo.config import cfg
    DPVO_AVAILABLE = True
except ImportError:
    DPVO_AVAILABLE = False

def estimate_dpvo_motion(...):
    if not DPVO_AVAILABLE:
        raise ImportError(
            "DPVO not installed. Install on CUDA machine: "
            "git clone https://github.com/princeton-vl/DPVO.git --recursive && "
            "conda env create -f environment.yml && pip install ."
        )
    ...
```

### Rotation-Only Stabilization (default)

**Why rotation-only:**
- Single monocular camera cannot recover absolute translation scale
- Head-mounted ego-motion is dominated by rotation (turning, nodding)
- Background at moderate distance: rotation-induced shift >> translation parallax

**Correction math:**
```python
def poses_to_corrections(
    poses: list[dict],
    calibration_pinhole: dict,
    smoothing_radius: int = 5,
) -> list[dict]:
    """Convert 6DOF poses to per-frame rotation-only stabilization corrections.
    
    1. Extract rotation matrices from quaternions
    2. Compute cumulative rotations (relative to frame 0)
    3. Convert to Rodrigues vectors
    4. Unwrap + smooth in rotvec space (moving average)
    5. Reconstruct smoothed rotations
    6. Correction: R_corr = R_smooth @ R_raw^T
    7. Project to image: H_corr = K @ R_corr @ K^{-1}
       (K = pinhole intrinsic matrix from undistorted calibration)
    
    Returns list of {frame_index, H_corr_3x3, rotvec_raw, rotvec_smooth, ...}
    """
```

**Translation stabilization (experimental, off by default):**
```python
def poses_to_corrections(
    ...,
    use_translation: bool = False,
    translation_attenuation: float = 0.1,
):
    """If use_translation=True:
    - Extract position (tx, ty, tz) from poses
    - Smooth positions
    - Project correction: dt_image ≈ attenuation * K @ (t_smooth - t_raw) / estimated_depth
    - Add as translation component in H_corr
    
    Attenuated by default because monocular scale is unreliable.
    """
```

### Integration with Existing Pipeline

`--method dpvo` follows the exact same flow as `--method raft`:

```
analyze_motion.py estimate --method dpvo
    → dpvo_bridge.estimate_dpvo_motion()
    → motion.json, transforms.json, trajectory.json

analyze_motion.py stabilize
    → reads trajectory.json (method=dpvo → uses H_corr_3x3 with warpPerspective)
    → stabilized/, comparison/

analyze_motion.py evaluate
    → stabilization_metrics.evaluate_stabilization()
    → evaluation_report.json (same M1/M2/M3)
```

The stabilize and evaluate steps don't need DPVO installed — they work from the JSON outputs. So you could even:
1. Run `estimate --method dpvo` on 5090
2. Git push the motion.json + trajectory.json
3. Run `stabilize` + `evaluate` on Mac (no CUDA needed)

### Artifact Contract for Cross-Machine Handoff

**Single source of truth:** `trajectory.json` is the only file needed for stabilize/evaluate.

**Required schema:**
```json
{
  "schema_version": "dpvo_v1",
  "method": "dpvo",
  "smoothing_radius": 5,
  "smoothing_radius_actual": 5,
  "rotation_only": true,
  "calibration_pinhole": {"fx": 500, "fy": 500, "cx": 960, "cy": 540},
  "undistort_params": {"balance": 0.0, "interpolation": "INTER_LINEAR", "border_mode": "BORDER_CONSTANT"},
  "frames": [
    {
      "frame_index": 0,
      "H_corr_3x3": [[1,0,0],[0,1,0],[0,0,1]],
      "rotvec_raw": [0,0,0],
      "rotvec_smooth": [0,0,0],
      "confidence": "normal"
    }
  ]
}
```

**Required keys for stabilize:** `schema_version`, `method`, `frames[].frame_index`, `frames[].H_corr_3x3`.
**Optional/diagnostic keys:** `rotvec_raw`, `rotvec_smooth`, `confidence`, `calibration_pinhole`.
**Fallback:** If `H_corr_3x3` is missing for a frame, use identity (no correction).

**Git commit policy:**
- ✅ Commit: `motion.json`, `trajectory.json`, `transforms.json`, `motion_summary.json`, `evaluation_report.json` (small JSONs)
- ❌ Don't commit: `stabilized/`, `comparison/`, `flow/` (large image dirs — regenerate locally)
- Add to `.gitignore`: `analysis/*/stabilized/`, `analysis/*/comparison/`, `analysis/*/flow/`

### Pre-Undistort Reproducibility

Fixed parameters for `cv2.fisheye.estimateNewCameraMatrixForUndistortRectify`:
```python
balance = 0.0          # no FOV preservation (max undistortion)
new_size = (W, H)      # same size as input
R = np.eye(3)          # no rotation
```
Undistort with:
```python
interpolation = cv2.INTER_LINEAR
border_mode = cv2.BORDER_CONSTANT
border_value = (0, 0, 0)
```
These parameters are stored in `trajectory.json` under `undistort_params` for reproducibility.

### Failure Detection

- **Trajectory too short** (< 50% expected frames): error + suggest lower stride
- **Pose jumps** (rotation > `--max-rotation-deg` (default 30°) or translation > `--max-translation-ratio` (default 10×) median between consecutive): flag as unreliable
- **NaN/inf in poses**: skip, use identity correction
- **> `--max-unreliable-ratio` (default 0.3) unreliable frames**: warn, suggest RAFT fallback

### Known Limitations

- **CUDA required for inference**: lietorch has custom CUDA kernels. No MPS/CPU fallback for DPVO itself.
- **Rolling shutter**: Ego-video cameras often use rolling shutter. DPVO assumes global shutter.
- **Scale ambiguity**: Monocular — translation is up-to-scale. Rotation-only default.

### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| DPVO install fails on 5090 | Can't run | Docker image available; pinned conda env |
| lietorch CUDA mismatch | Build error | Pin pytorch=2.3.1, cuda=12.1 |
| Fisheye → DPVO incompatible | Wrong poses | Pre-undistort; pinhole-only calibration |
| Translation overcorrection | Warp artifacts | Rotation-only default; translation opt-in + attenuated |
| DPVO fails on ego-video | Bad trajectory | Failure detection; RAFT fallback |

---

## Part 3 — Test Plan

### Tests on Mac (no CUDA, no DPVO)

1. `test_calibration_conversion_pinhole` — undistorted intrinsics → pinhole format
2. `test_pre_undistort_frames` — fisheye frames correctly undistorted
3. `test_parse_tum_trajectory` — parses sample TUM file
4. `test_quaternion_to_rotation_matrix` — verified against scipy
5. `test_quaternion_sign_continuity` — q and -q handled correctly
6. `test_trajectory_smoothing_rotvec` — smoothed rotvecs are smoother
7. `test_correction_identity_for_static` — static poses → identity corrections
8. `test_correction_homography_from_rotation` — R_corr → H = K R K^{-1}
9. `test_trajectory_alignment_by_timestamp` — correct frame matching with tolerance
10. `test_trajectory_too_short_error` — < 50% match → error
11. `test_pose_jump_detection` — large rotation flagged
12. `test_dpvo_not_installed_error` — clear error message on Mac
13. `test_stabilize_from_trajectory_json` — stabilize works from saved JSON (no DPVO needed)
14. `test_slerp_interpolation` — stride > 1 intermediate frames correct

### Tests on 5090 (CUDA + DPVO)

15. `test_dpvo_inference_runs` — produces trajectory from sample data
16. `test_end_to_end_ego_video` — full pipeline on Ego10K clip
17. `test_dpvo_with_undistorted_input` — pinhole frames → valid poses

---

## Part 4 — Rollout & Status

### Implementation Order
1. `src/vibelab/ego_video/motion/dpvo_bridge.py` — DPVO integration + pose → correction
2. Update `scripts/ego_video/analyze_motion.py` — add `--method dpvo`
3. Mac-side unit tests (1-14)
4. Wayne installs DPVO on 5090 + runs inference tests (15-17)
5. Evaluate and compare with RAFT + affine2d baselines

### Status
- [x] Design round 1 reviewed
- [x] Design round 2 reviewed
- [x] Design round 3 — architecture simplified (unified pipeline, Git sync)
- [x] Design round 3 reviewed (Go)
- [ ] Wayne approval
- [ ] Implementation
- [ ] Mac tests passing
- [ ] 5090 end-to-end test
- [ ] Comparison with baselines

---

## Review Notes

### Round 1 — Reviewer: gpt-5.3-codex

**Major (5) — all addressed:**
1. 6DOF→2D math incorrect: switched to rotation-only default
2. Monocular translation scale ambiguity: documented, translation off by default
3. Plane homography removed: correction is K @ R_corr @ K^{-1}
4. Fisheye incompatibility: pre-undistort strategy
5. Trajectory-frame alignment: strict contract with timestamp matching

**Minor (4) — all addressed:**
1. Parameterized run script → replaced with unified CLI
2. Rotation-only default
3. Rolling shutter documented
4. Failure detection + RAFT fallback

### Round 2 — Reviewer: gpt-5.3-codex

**Major (1) — addressed:**
1. Stride hardcoded → now parameterized via CLI `--stride`

**Minor (3) — addressed:**
1. Fisheye policy unified: pre-undistort, pinhole-only
2. Calibration branching clarified
3. Manifest includes timestamps

### Round 3 — Architecture Change

**Simplified from split-compute to unified pipeline:**
- Removed: `dpvo-prepare`, `dpvo-stabilize` separate subcommands, `run_dpvo.sh`, rsync/scp workflow
- Added: `--method dpvo` as standard method in existing `estimate` subcommand
- DPVO called via Python API (not shell script)
- Import-guarded: clear error on Mac if DPVO not installed
- Stabilize + evaluate work from saved JSON (no DPVO needed)
- One codebase, Git sync between Mac and 5090

### Round 3 — Reviewer: gpt-5.3-codex

**Major (1) — addressed:**
1. ❌→✅ Artifact contract unspecified for cross-machine handoff: **Added explicit schema** — `trajectory.json` is single source of truth, `schema_version: "dpvo_v1"`, required/optional keys defined. Git commit policy: JSONs only, images in .gitignore.

**Minor (3) — addressed:**
1. Import guard: unified to `ImportError` (not RuntimeError).
2. Git large artifact: explicit commit policy + .gitignore rules.
3. Pre-undistort reproducibility: fixed params (balance=0, INTER_LINEAR, BORDER_CONSTANT) stored in trajectory.json.

**Nit (2) — addressed:**
1. DPVO-specific fields: marked as optional/diagnostic in schema.
2. Failure thresholds: surfaced as CLI tunables (`--max-rotation-deg`, etc.).
