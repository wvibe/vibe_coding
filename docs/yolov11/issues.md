# YOLOv11 Issues

## VOC Segmentation Fine-tuning

*   **Issue:** Ultralytics scans incorrect dataset directory (`/hub/datasets/VOC/`) instead of the one specified in `voc_segment.yaml` (`/hub/datasets/segment_VOC/`) when running `train_segment.py`.
*   **Symptoms:** Training script logs show scanning of `/.../VOC/labels/...` despite correct configuration paths.
*   **Root Cause Hypothesis:** Likely interference from stale Ultralytics cache files (`*.cache`) located in the incorrect `VOC/labels/` directory, or potential override by the `VOC_ROOT` environment variable.
*   **Resolution Attempted (2025-04-05):** Cleared `.cache` files from both `/hub/datasets/VOC/labels/` and `/hub/datasets/segment_VOC/labels/` directories.