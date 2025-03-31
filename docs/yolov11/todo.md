# YOLOv11 TODO List

## Phase 1: Detection

- [x] Milestone 1: Environment Confirmation & Project Structure
  - [x] Confirm `bash` shell & activate `conda` environment `vbl`
  - [x] Check/Update `ultralytics` library
  - [x] Create directories (`docs/yolov11`, `src/.../yolov11/configs`, `tests/.../yolov11`)
  - [x] Create initial docs (`README.md`, `design.md`, `todo.md`)
- [x] Milestone 2: VOC Dataset Configuration
  - [x] Confirm VOC dataset root (`$HOME/vibe/hub/datasets/VOC`)
  - [x] Decide on train/val splits (using `voc_combined.yaml` splits)
  - [x] Create `src/models/ext/yolov11/configs/voc_detect.yaml`
- [x] Milestone 3: Basic Prediction Script (Detection)
  - [x] Design config YAML structure (`predict_config.yaml` concept)
  - [x] Create demo config `predict_demo.yaml` (incl. `project` key)
  - [x] Design output folder structure (`<project>/<prefix>_<timestamp>`)
  - [x] Create script `predict_detect.py`
  - [x] Implement argument parsing (`--config`, `--name-prefix`)
  - [x] Implement config loading & output path generation
  - [x] Implement source handling (file/dir/random select)
  - [x] Implement core prediction logic using `model.predict`
  - [x] Add entry point & docstrings
- [ ] Milestone 4: Training/Finetuning Script (Detection on VOC)
  - [x] Create training configs (`voc_finetune.yaml`, `voc_retrain.yaml`)
  - [x] Create script `train_detect.py` (adapting `train_yolov8.py`)
  - [x] Implement config loading, path resolution, resume logic
  - [x] Implement `model.train()` call with args
  - [x] Add CLI args (`--config`, `--name`, `--project`, `--resume`, etc.)

- [ ] Milestone 5: Metrics Implementation (Detection)
  - [x] Step 5.1: Setup directories (`src/utils/metrics`, `tests/utils/metrics`) and base files (`detection.py`, `test_detection.py`, `__init__.py`)
  - [x] Step 5.2: Implement and Test Core Metrics Incrementally:
    - [x] 5.2.1: Implement `calculate_iou` and add unit tests.
    - [x] 5.2.2: Implement `match_predictions` and add unit tests.
    - [x] 5.2.3: Implement `calculate_pr_data` and add unit tests.
    - [x] 5.2.4: Implement AP & mAP calculation and add unit tests.
    - [x] 5.2.5: Implement mAP by Size calculation and add unit tests.
    - [x] 5.2.6: Implement Confusion Matrix generation and add unit tests.
    - [x] 5.2.7: Implement compute utilities (`get_model_params`, `get_peak_gpu_memory_mb`) and add unit tests.
  - [ ] Step 5.3: Develop Evaluation Script (`evaluate_detect.py`):
    - [x] 5.3.1: Setup script structure with single `--config` argument and basic imports.
    - [x] 5.3.2: Create `evaluate_default.yaml` configuration file with comprehensive options.
    - [x] 5.3.3: Implement configuration loading and validation.
    - [x] 5.3.4: Add model loading with parameter counting via `get_model_params`.
    - [x] 5.3.5: Implement inference with warmup and measurement of time/memory.
    - [x] 5.3.6: Implement ground truth loading and format conversion.
    - [x] 5.3.7: Integrate metric calculation (`match_predictions`, `calculate_map`, etc.).
    - [x] 5.3.8: Implement visualization and result saving.
    - [x] 5.3.9: Create evaluation documentation in `docs/yolov11/evaluate.md`.

- [ ] Milestone 6: Benchmarking and Reporting (Detection)
  - [ ] Implement comparison logic: Predictions vs. Ground Truth
  - [ ] Implement comparison logic: Model vs. Model / Run vs. Run
  - [ ] Implement plain text report generation function
  - [ ] Implement HTML report generation function (with plots/visualizations)
  - [ ] Create script/CLI for running benchmarks & generating reports

- [ ] Milestone 7: Documentation Update (Detection)
  - [x] Add usage examples to `README.md` for prediction
  - [x] Add usage examples to `README.md` for training (finetune & retrain)
  - [x] Review/update `design.md` (updated for prediction & training scripts)
  - [x] Add documentation for metrics calculation utilities (Step 5.1, 5.2)
  - [ ] Add documentation for metrics calculation and configuration (Milestone 5)
  - [ ] Add documentation for benchmarking script and reports (Milestone 6)

## Phase 2: Segmentation & Further Expansion

- [x] Milestone 8: Basic Segmentation Support
  - [x] Create segmentation prediction config (`predict_segment.yaml`)
  - [x] Create common utility module for prediction (`predict_utils.py`)
  - [x] Implement segmentation prediction script (`predict_segment.py`)
  - [x] Add proper documentation in README.md
  - [x] Set up proper Python package structure with __init__.py files
- [ ] Milestone 9: Advanced Segmentation Features
  - [ ] Segmentation dataset preparation
  - [ ] Segmentation configurations (`*_seg.yaml`)
  - [ ] Segmentation finetuning script (`train_segment.py`)
  - [ ] Segmentation evaluation script (`evaluate_segment.py`)
  - [ ] Segmentation tests
  - [ ] COCO Dataset Integration (Detection & Segmentation)
  - [ ] Full training scripts (optional)
  - [ ] Final comprehensive documentation update