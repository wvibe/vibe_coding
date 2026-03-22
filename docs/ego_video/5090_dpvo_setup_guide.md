# 5090 Ubuntu — DPVO Setup & Run Guide

## Prerequisites
- Ubuntu 20/22
- NVIDIA GPU with CUDA 11+ (5090 with CUDA 12)
- Conda installed
- Git access to `wvibe/vibe_coding`

## Step 1: Clone & Setup Repo

```bash
# Clone vibe_coding repo
git clone git@github.com:wvibe/vibe_coding.git
cd vibe_coding
git checkout feat/ego-video-mvp
git pull origin feat/ego-video-mvp
```

## Step 2: Install DPVO

```bash
# Clone DPVO (outside vibe_coding)
cd ~
git clone https://github.com/princeton-vl/DPVO.git --recursive
cd DPVO

# Create conda env from DPVO's pinned environment
conda env create -f environment.yml
conda activate dpvo

# Install Eigen (required for DPVO build)
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

# Build and install DPVO
pip install .

# Download pretrained models (~2GB)
./download_models_and_data.sh

# Verify installation
python -c "from dpvo.dpvo import DPVO; print('DPVO OK')"
```

## Step 3: Install vibe_coding in DPVO env

**Environment note:** DPVO requires Python 3.10 + PyTorch 2.3.1 + CUDA 12.1 (pinned by its environment.yml). Mac uses Python 3.11 + PyTorch 2.10. These cannot be unified — use separate conda envs on each machine.

```bash
# Back to vibe_coding
cd ~/vibe_coding

# Install vibelab package (editable) — into the dpvo conda env
pip install -e .

# Install additional deps needed by our stabilization/evaluation pipeline
# (these are not in DPVO's environment.yml but our scripts need them)
pip install matplotlib scipy huggingface_hub datasets transformers av ffmpeg-python

# Verify DPVO + vibelab both work
python -c "
from vibelab.ego_video.motion.dpvo_bridge import DPVO_AVAILABLE
from vibelab.ego_video.io.frame_dataset import extract_multi_rate_frames
from vibelab.ego_video.motion.stabilization_metrics import evaluate_stabilization
print('DPVO_AVAILABLE:', DPVO_AVAILABLE)
print('All imports OK')
"
# Expected:
#   DPVO_AVAILABLE: True
#   All imports OK
```

### Environment Summary (two machines, two envs)

| | Mac mini (ego env) | 5090 Ubuntu (dpvo env) |
|---|---|---|
| Python | 3.11 | 3.10 |
| PyTorch | 2.10 | 2.3.1 |
| CUDA | N/A (MPS) | 12.1 |
| lietorch | N/A | ✅ (via DPVO) |
| vibelab | `pip install -e .` | `pip install -e .` |
| OpenCV | 4.13 | (from DPVO env) |
| HuggingFace | ✅ | ✅ (pip install) |
| RAFT | torchvision built-in | torchvision built-in |
| DPVO | ❌ (import guarded) | ✅ |

**Key point:** The Python code is identical on both machines. Only the conda env differs. `--method dpvo` works on 5090; on Mac it raises a clear ImportError.

## Step 4: Prepare Data

```bash
# Create data directory (same structure as Mac)
mkdir -p ~/Data/datasets/ego_video/builddotai/frame_sets

# Option A: Download fresh from HuggingFace
# (need HF login: huggingface-cli login)
cd ~/vibe_coding
python scripts/ego_video/prepare_frames.py download \
    --worker factory_001/worker_001 \
    --max-samples 1 \
    --output-dir ~/Data/datasets/ego_video/builddotai/ego10k_samples

python scripts/ego_video/prepare_frames.py extract \
    --video ~/Data/datasets/ego_video/builddotai/ego10k_samples/videos/factory001_worker001_00000.mp4 \
    --start-sec 10 \
    --num-frames 20 \
    --fps 30,10,3 \
    --calibration ~/Data/datasets/ego_video/builddotai/ego10k_samples/factory_001/workers/worker_001/intrinsics.json \
    --sample-id factory001_worker001_00000 \
    --output-dir ~/Data/datasets/ego_video/builddotai/frame_sets/clip_001

# Option B: Copy frame_sets from Mac via scp/rsync (faster, ~60MB)
# scp -r wei@mac:~/Data/datasets/ego_video/builddotai/frame_sets/clip_001 ~/Data/datasets/ego_video/builddotai/frame_sets/
```

## Step 5: Run DPVO Estimation

```bash
cd ~/vibe_coding
FRAME_SET=~/Data/datasets/ego_video/builddotai/frame_sets/clip_001

# Run on all three frame rates
for FPS in 30 10 3; do
  echo "=== DPVO ${FPS}fps ==="
  python scripts/ego_video/analyze_motion.py estimate \
    --frame-set "$FRAME_SET" \
    --fps $FPS \
    --method dpvo \
    --calibration ~/Data/datasets/ego_video/builddotai/ego10k_samples/factory_001/workers/worker_001/intrinsics.json \
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
