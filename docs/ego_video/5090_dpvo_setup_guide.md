# 5090 Ubuntu — DPVO Setup & Run Guide

## Prerequisites
- Ubuntu 20/22 with NVIDIA GPU (5090 confirmed)
- Existing `ego` conda env with Python 3.11 + PyTorch 2.9.1 + CUDA 12.9
- `vibe_coding` repo cloned with `vibelab` installed

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

The 5090 `ego` env already has Python 3.11, PyTorch 2.9.1, CUDA 12.9, and all other dependencies. We only need to add DPVO + lietorch.

> **Note:** DPVO officially pins PyTorch 2.3.1 + CUDA 12.1, but lietorch should compile against newer versions. If `pip install .` fails, fall back to Plan B (separate dpvo env).

**Plan A: Install into existing ego env (recommended):**

```bash
# Clone DPVO
cd ~
git clone https://github.com/princeton-vl/DPVO.git --recursive
cd DPVO

conda activate ego

# Install Eigen (required for lietorch CUDA build)
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

# Build and install DPVO (compiles lietorch CUDA kernels)
pip install .

# Download pretrained models (~2GB)
./download_models_and_data.sh

# Verify
python -c "from dpvo.dpvo import DPVO; print('DPVO OK')"
python -c "from vibelab.ego_video.motion.dpvo_bridge import DPVO_AVAILABLE; print('DPVO_AVAILABLE:', DPVO_AVAILABLE)"
```

**Plan B: Separate dpvo env (if Plan A fails):**

```bash
cd ~/DPVO
conda env create -f environment.yml   # Creates Python 3.10 + PyTorch 2.3.1
conda activate dpvo
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty
pip install .
./download_models_and_data.sh

# Also install vibelab + deps in dpvo env
cd ~/vibe_coding
pip install -e .
pip install matplotlib scipy huggingface_hub datasets transformers

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
    --dpvo-model ~/DPVO/dpvo.pth \
    --dpvo-config ~/DPVO/config/default.yaml \
    --stride 1
done
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
| `ImportError: lietorch` | Reinstall in dpvo env: `cd ~/DPVO && pip install .` |
| `CUDA out of memory` | Reduce frame count or use `--stride 2` |
| `No module named vibelab` | `cd ~/vibe_coding && pip install -e .` |
| `HF gated access denied` | Run `huggingface-cli login` and accept terms at https://huggingface.co/datasets/builddotai/Egocentric-10K |
| DPVO crashes silently | Check `~/DPVO/dpvo.pth` exists; run `./download_models_and_data.sh` |

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
