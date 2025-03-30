# YOLOv11 Design Notes

## Core Library

- This module relies heavily on the `ultralytics` Python package.
- We will leverage its `YOLO` class for loading models, running inference (`predict`), and training/finetuning (`train`).

## Initial Scope: Detection

- **Configuration:**
  - Dataset configurations (e.g., `voc_detect.yaml`) follow the Ultralytics format for training/finetuning.
  - Prediction uses separate YAML configs (e.g., `predict_demo.yaml`) specifying model, source, output project dir, Ultralytics prediction args (`conf`, `imgsz`, `save_txt`, `save`, etc.), and custom args (`random_select`).
- **Prediction (New Script):** A script (`predict_detect.py`) provides a command-line interface to run pre-trained YOLOv11 detection models.
  - Takes `--config <yaml_path>` and `--name-prefix <prefix>` as input.
  - Reads parameters from the specified YAML config.
  - Handles single image files or directories as `source`.
  - Supports random selection of N images if `source` is a directory and `random_select: N` is set in config.
  - Saves results to `<config.project>/<name-prefix>_<timestamp>/`.
  - Leverages `ultralytics.YOLO.predict` for core logic and output generation (annotated images if `save: True`, YOLO labels if `save_txt: True`).
- **Finetuning:** A script (`finetune_detect.py`) will allow finetuning pre-trained YOLOv11 detection models on custom datasets defined in the dataset configuration files (starting with VOC).

## Future Scope: Segmentation & Expansion

- **Segmentation:** Similar scripts (`inference_seg.py`, `finetune_seg.py`) and configurations (`*_seg.yaml`) will be added once segmentation datasets are available.
- **Training:** Scripts for training from scratch (`train_*.py`) might be added if needed.
- **Datasets:** Support for COCO will be added via `coco_*.yaml` configurations.
- **Tests:** Unit tests will cover basic script execution and argument parsing.