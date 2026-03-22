# 5090 Ubuntu — DPVO Setup & Run Guide

## Prerequisites
- Ubuntu 20/22 with NVIDIA GPU (5090 confirmed)
- Existing `ego` conda env with Python 3.11 + PyTorch 2.9.1 + CUDA 12.9
- `vibe_coding` repo cloned with `vibelab` installed

### Current host status

Already confirmed on the 5090 host:

- `ego` conda env exists
- Python `3.11.15`
- PyTorch `2.9.1`
- CUDA available in torch (`torch.cuda.is_available() == True`)
- `vibelab`, `cv2`, `matplotlib`, `scipy`, `huggingface_hub`, `datasets`, `transformers` already import

Still missing:

- DPVO repo checkout
- `dpvo` Python package
- `lietorch` extension (built as part of DPVO install)
- `torch-scatter`
- DPVO pretrained models / weights

## Step 1: Ensure repo is up to date

```bash
cd ~/vibe_coding
git checkout feat/ego-video-mvp
git pull origin feat/ego-video-mvp

# Reinstall vibelab (picks up new modules)
conda activate ego
pip install -e .
```

## Step 2: Install DPVO into existing ego env

The 5090 `ego` env already has Python 3.11, PyTorch 2.9.1, CUDA 12.9, and all other dependencies. We only need to add DPVO + lietorch + `torch-scatter`.

> **Note:** DPVO officially pins PyTorch 2.3.1 + CUDA 12.1, but lietorch should compile against newer versions. If `pip install .` fails, fall back to Plan B (separate dpvo env).

**Plan A: Install into existing ego env (recommended):**

```bash
# Clone DPVO under ~/vibe, not directly in home
mkdir -p ~/vibe
cd ~/vibe
git clone https://github.com/princeton-vl/DPVO.git --recursive
cd DPVO

conda activate ego

# Install Python deps that DPVO expects but are not in ego by default
pip install numba einops pypose kornia plyfile evo yacs

# torch-scatter: build locally against the existing torch/cuda stack
CPATH=/usr/local/cuda-12.9/include pip install --no-build-isolation torch-scatter

# Install Eigen (required for lietorch CUDA build)
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

# Upstream packaging gap: include loop_closure as a Python package
touch dpvo/loop_closure/__init__.py dpvo/loop_closure/retrieval/__init__.py

# Build and install DPVO (compiles lietorch CUDA kernels)
# Notes:
# - --no-build-isolation is required so setup.py can see torch
# - CPATH exposes CUDA headers so lietorch can find cuda.h
CPATH=/usr/local/cuda-12.9/include pip install --no-build-isolation .

# Download pretrained model weights only
wget https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip
unzip -o models.zip
# Expect: ~/vibe/DPVO/dpvo.pth

# Verify
python -c "from dpvo.dpvo import DPVO; print('DPVO OK')"
python -c "from vibelab.ego_video.motion.dpvo_bridge import DPVO_AVAILABLE; print('DPVO_AVAILABLE:', DPVO_AVAILABLE)"
```

**Important runtime note:** for this repo's image-directory DPVO path, full-resolution `1920x1080` inputs can OOM even on the 5090. The bridge now supports explicit downscaling, and `--dpvo-scale 0.5` is the recommended default on this host.

For host-side sweeps, the repo also includes a lower-memory DPVO config:

- [dpvo_p48.yaml](/home/wei/vibe/vibe_coding/configs/ego_video/dpvo_p48.yaml)
  - `PATCHES_PER_FRAME: 48`

This was enough to make `30fps` full-resolution (`--dpvo-scale 1.0`) run on the 5090 for the `clip_001` experiment.

**Plan B: Separate dpvo env (if Plan A fails):**

```bash
cd ~/vibe/DPVO
conda env create -f environment.yml   # Creates Python 3.10 + PyTorch 2.3.1
conda activate dpvo
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty
touch dpvo/loop_closure/__init__.py dpvo/loop_closure/retrieval/__init__.py
CPATH=/usr/local/cuda-12.9/include pip install --no-build-isolation .
wget https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip
unzip -o models.zip

# Also install vibelab + deps in dpvo env
cd ~/vibe_coding
pip install -e .
pip install matplotlib scipy huggingface_hub datasets transformers numba einops pypose kornia plyfile evo yacs
CPATH=/usr/local/cuda-12.9/include pip install --no-build-isolation torch-scatter

python -c "from vibelab.ego_video.motion.dpvo_bridge import DPVO_AVAILABLE; print(DPVO_AVAILABLE)"
```

### Environment Summary (two machines)

| | Mac mini (ego env) | 5090 Ubuntu (ego env) |
|---|---|---|
| Python | 3.11 | 3.11 |
| PyTorch | 2.10 (MPS) | 2.9.1 (CUDA 12.9) |
| CUDA | N/A | 12.9 ✅ |
| lietorch | N/A | ✅ (via DPVO) |
| vibelab | `pip install -e .` | `pip install -e .` |
| OpenCV | 4.13 | 4.13 (headless) |
| HuggingFace | ✅ | ✅ |
| RAFT | torchvision 0.25 | torchvision 0.25 |
| DPVO | ❌ (import guarded) | ✅ (after Step 2) |

**Key point:** Same Python version, same code. Only difference is CUDA + DPVO.

## Step 4: Prepare Data (video → image sets)

This step downloads a sample ego-video from HuggingFace and extracts multi-rate frame sets (30fps, 10fps, 3fps × 20 frames each). These frame sets are the input to all estimation methods.

```bash
# Set data root (adjust to your machine's storage path)
export DATA_ROOT=~/Data  # or /data, /home/wei/Data, etc.
mkdir -p $DATA_ROOT/datasets/ego_video/builddotai/frame_sets

# HuggingFace login (required — Ego10K is a gated dataset)
# Visit https://huggingface.co/datasets/builddotai/Egocentric-10K to accept terms first
huggingface-cli login

# Option A: Download + extract from HuggingFace (recommended)
cd ~/vibe_coding
python scripts/ego_video/prepare_frames.py download \
    --worker factory_001/worker_001 \
    --max-samples 1 \
    --output-dir $DATA_ROOT/datasets/ego_video/builddotai/ego10k_samples

python scripts/ego_video/prepare_frames.py extract \
    --video $DATA_ROOT/datasets/ego_video/builddotai/ego10k_samples/videos/factory001_worker001_00000.mp4 \
    --start-sec 10 \
    --num-frames 20 \
    --fps 30,10,3 \
    --calibration $DATA_ROOT/datasets/ego_video/builddotai/ego10k_samples/factory_001/workers/worker_001/intrinsics.json \
    --sample-id factory001_worker001_00000 \
    --output-dir $DATA_ROOT/datasets/ego_video/builddotai/frame_sets/clip_001

# Verify extraction
python scripts/ego_video/prepare_frames.py inspect \
    --frame-set $DATA_ROOT/datasets/ego_video/builddotai/frame_sets/clip_001
# Expected: ✅ 30fps: 20/20, ✅ 10fps: 20/20, ✅ 3fps: 20/20

# Option B: Copy from Mac instead (faster, ~60MB)
# scp -r wei@mac:$DATA_ROOT/datasets/ego_video/builddotai/frame_sets/clip_001 $DATA_ROOT/datasets/ego_video/builddotai/frame_sets/
# scp -r wei@mac:$DATA_ROOT/datasets/ego_video/builddotai/ego10k_samples $DATA_ROOT/datasets/ego_video/builddotai/  # need intrinsics for calibration
```

## Step 5: Run DPVO Estimation

```bash
cd ~/vibe_coding
FRAME_SET=$DATA_ROOT/datasets/ego_video/builddotai/frame_sets/clip_001

# Run on all three frame rates
for FPS in 30 10 3; do
  echo "=== DPVO ${FPS}fps ==="
  python scripts/ego_video/analyze_motion.py estimate \
    --frame-set "$FRAME_SET" \
    --fps $FPS \
    --method dpvo \
    --calibration $DATA_ROOT/datasets/ego_video/builddotai/ego10k_samples/factory_001/workers/worker_001/intrinsics.json \
    --output-dir "$FRAME_SET/analysis/${FPS}fps_dpvo" \
    --smoothing-radius 5 \
    --dpvo-model ~/vibe/DPVO/dpvo.pth \
    --dpvo-config ~/vibe/DPVO/config/default.yaml \
    --stride 1 \
    --dpvo-scale 0.5
done
```

### Optional parameter sweep examples

If you want to probe whether DPVO pose quality is being limited by low input resolution, these worked on the 5090 host for `clip_001`:

```bash
# 30fps, higher resolution
python scripts/ego_video/analyze_motion.py estimate \
    --frame-set "$FRAME_SET" \
    --fps 30 \
    --method dpvo \
    --calibration $DATA_ROOT/datasets/ego_video/builddotai/ego10k_samples/factory_001/workers/worker_001/intrinsics.json \
    --output-dir "$FRAME_SET/analysis/30fps_dpvo_s075" \
    --smoothing-radius 5 \
    --dpvo-model ~/vibe/DPVO/dpvo.pth \
    --dpvo-config ~/vibe/DPVO/config/default.yaml \
    --stride 1 \
    --dpvo-scale 0.75

# 30fps, full resolution with lower-memory config
python scripts/ego_video/analyze_motion.py estimate \
    --frame-set "$FRAME_SET" \
    --fps 30 \
    --method dpvo \
    --calibration $DATA_ROOT/datasets/ego_video/builddotai/ego10k_samples/factory_001/workers/worker_001/intrinsics.json \
    --output-dir "$FRAME_SET/analysis/30fps_dpvo_s100_p48" \
    --smoothing-radius 5 \
    --dpvo-model ~/vibe/DPVO/dpvo.pth \
    --dpvo-config configs/ego_video/dpvo_p48.yaml \
    --stride 1 \
    --dpvo-scale 1.0
```

## Step 6: Run Stabilization & Evaluation

```bash
# Stabilize + evaluate (these don't need DPVO, just OpenCV)
for FPS in 30 10 3; do
  echo "=== Stabilize ${FPS}fps ==="
  python scripts/ego_video/analyze_motion.py stabilize \
    --frame-set "$FRAME_SET" \
    --fps $FPS \
    --output-dir "$FRAME_SET/analysis/${FPS}fps_dpvo"

  echo "=== Evaluate ${FPS}fps ==="
  python scripts/ego_video/analyze_motion.py evaluate \
    --frame-set "$FRAME_SET" \
    --fps $FPS \
    --analysis-dir "$FRAME_SET/analysis/${FPS}fps_dpvo"
done
```

### Current repo behavior for DPVO stabilization

The repo now applies DPVO homographies in undistorted pinhole space instead of directly on raw fisheye frames.

Concretely:

- `stabilize` creates `analysis/.../_undistorted_for_stab` when fisheye calibration is available
- the DPVO `H_corr` homographies are applied to those undistorted frames
- `evaluate` compares `stabilized/` against the undistorted raw baseline for DPVO runs

This fixes the earlier image-space mismatch, but on `clip_001` it was not enough by itself to make DPVO positive. The remaining work is now mostly about pose quality and parameter tuning.

## Step 7: Check Results

```bash
# Print reports
for FPS in 30 10 3; do
  echo "=== ${FPS}fps DPVO ==="
  python scripts/ego_video/analyze_motion.py report \
    --analysis-dir "$FRAME_SET/analysis/${FPS}fps_dpvo"
  echo
done
```

## Step 8: Commit & Push Results

```bash
cd ~/vibe_coding

# Only commit JSON results (not large image dirs)
git add -A
git reset -- '*/stabilized/' '*/comparison/' '*/flow/' '*/_undistorted/'
git commit -m "feat(ego-video): DPVO estimation results on 30/10/3fps"
git push origin feat/ego-video-mvp
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: torch_scatter` | `conda activate ego && CPATH=/usr/local/cuda-12.9/include pip install --no-build-isolation torch-scatter` |
| `ImportError: lietorch` | Reinstall in the DPVO checkout: `cd ~/vibe/DPVO && CPATH=/usr/local/cuda-12.9/include pip install --no-build-isolation .` |
| `ModuleNotFoundError: dpvo.loop_closure` | Add package markers, then reinstall: `touch dpvo/loop_closure/__init__.py dpvo/loop_closure/retrieval/__init__.py && CPATH=/usr/local/cuda-12.9/include pip install --no-build-isolation .` |
| `CUDA out of memory` | Reduce frame count or use `--stride 2` |
| `CUDA out of memory` on `1920x1080` image sets | Use `--dpvo-scale 0.5` so the bridge resizes the undistorted DPVO input before inference |
| DPVO still underperforms after the image-space fix | Try `--dpvo-scale 0.75` or `1.0`; if that OOMs, lower `PATCHES_PER_FRAME` in the DPVO config and rerun `estimate` |
| `--dpvo-scale 1.0` OOMs on the 5090 | Retry with `--dpvo-config configs/ego_video/dpvo_p48.yaml` |
| `--dpvo-scale 0.75` runs at `10fps` but OOMs at `3fps` | This can still happen because DPVO memory depends on both resolution and trajectory graph size; retry with `configs/ego_video/dpvo_p48.yaml` and no concurrent GPU jobs |
| `Trajectory alignment failed` with only `1/N` frames matched | The repo now normalizes DPVO's frame-index timestamps to seconds during alignment; make sure you're running the updated `dpvo_bridge.py` |
| `No module named vibelab` | `cd ~/vibe_coding && pip install -e .` |
| `HF gated access denied` | Run `huggingface-cli login` and accept terms at https://huggingface.co/datasets/builddotai/Egocentric-10K |
| DPVO crashes silently | Check `~/vibe/DPVO/dpvo.pth` exists; if missing, download and unzip `models.zip` in `~/vibe/DPVO` |

## Expected Output

After Step 7, you should see results like:
```
=== 30fps DPVO ===
Method: dpvo
Stability (crop 0.5): raw=X.XXXX, stab=X.XXXX, improvement=XX.X%
Residual motion: raw=X.XX, stab=X.XX, improvement=XX.X%
```

Compare with our baselines:
- affine2d@30fps: stability +10.5%, motion +22.8%
- RAFT@10fps: stability +16.8%, motion +58.5%
- RAFT@3fps: stability +7.8%, motion +80.5%
