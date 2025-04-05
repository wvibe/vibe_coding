# YOLOv11 TODOs

## VOC Segmentation Fine-tuning

*   [ ] **Verify Dataset Path Fix:** Re-run training (`train_segment.py --name ...run4`) after clearing cache files to confirm Ultralytics now scans the correct `/hub/datasets/segment_VOC/` directory.
*   [ ] **Investigate Env Variable:** If the issue persists, temporarily comment out `VOC_ROOT` in `.env` and re-run training to check for environment variable interference.
*   [ ] **Complete Training:** Once the dataset path issue is resolved, run the full fine-tuning process.
*   [ ] **Evaluate Model:** Evaluate the fine-tuned segmentation model on the `test2007` split.