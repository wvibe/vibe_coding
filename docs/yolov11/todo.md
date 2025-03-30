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
- [ ] Milestone 4: Finetuning Script (Detection on VOC)
  - [ ] Create `src/models/ext/yolov11/finetune_detect.py`
  - [ ] Implement CLI args (`--model`, `--data`, `--epochs`, etc.)
  - [ ] Implement loading YOLOv11 model & training
  - [x] Milestone 4: Training Script (Detection on VOC)
    - [x] Create training configs (`voc_finetune.yaml`, `voc_scratch.yaml`)
    - [x] Create script `train_detect.py` (adapting `train_yolov8.py`)
    - [x] Implement config loading, path resolution, resume logic
    - [x] Implement `model.train()` call with args
    - [x] Add CLI args (`--config`, `--name`, `--project`, `--resume`, etc.)
- [ ] Milestone 5: Unit Tests (Detection)
  - [ ] Create test files in `tests/models/ext/yolov11/`
  - [ ] Add basic tests for `predict_detect.py`
  - [ ] Add basic tests for `train_detect.py`
- [ ] Milestone 6: Documentation Update (Detection)
  - [x] Add usage examples to `README.md` for prediction
  - [x] Review/update `design.md` (updated for prediction script)
  - [ ] Add usage examples to `README.md` for training

## Phase 2: Segmentation & Further Expansion (Deferred)

- [ ] Segmentation dataset preparation
- [ ] Segmentation configurations (`*_seg.yaml`)
- [ ] Segmentation inference script (`inference_seg.py`)
- [ ] Segmentation finetuning script (`finetune_seg.py`)
- [ ] Segmentation tests
- [ ] COCO Dataset Integration (Detection & Segmentation)
- [ ] Full training scripts (optional)
- [ ] Final comprehensive documentation update