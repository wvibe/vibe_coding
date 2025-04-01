# VOC Dataset Conversion TODO

## Overview
This document tracks the tasks needed to convert Pascal VOC dataset to YOLO format for both detection and segmentation tasks.

## Tasks

### 1. Directory Structure and Organization
- [x] Create basic directory structure for YOLO format (initial attempt)
- [x] Update directory structure to match project standard and separate detect/segment labels:
  ```
  /home/wmu/vibe/hub/datasets/VOC/
  ├── images/
  │   ├── train2007/
  │   ├── val2007/
  │   ├── test2007/
  │   ├── train2012/
  │   └── val2012/
  ├── labels_detect/               # Bounding box labels (YOLO format)
  │   ├── train2007/
  │   ├── val2007/
  │   ├── test2007/ # (May be empty)
  │   ├── train2012/
  │   └── val2012/
  └── labels_segment/              # Segmentation polygon labels (YOLO format)
      ├── train2007/
      ├── val2007/
      ├── test2007/ # (May be empty)
      ├── train2012/
      └── val2012/
  ```

### 2. Code Refactoring
- [x] Rename `voc2yolo.py` -> `voc2yolo_segment_labels.py`
- [x] Remove image copying logic
- [x] Add image existence verification (integrated into scripts)
- [x] Implement instance-class matching logic (IoU based)
- [x] Update `segment_voc2yolo.py` output to `labels_segment` directory
- [x] Implement actual mask reading and polygon conversion in `segment_voc2yolo.py`
- [x] Implement actual mask reading and polygon conversion in `voc2yolo_segment_labels.py`
- [x] Identify and move detection script -> `detect_voc2yolo.py`
- [x] Identify and move detection script -> `voc2yolo_detect_labels.py`
- [x] Adapt `detect_voc2yolo.py` for project structure (`labels_detect` output, no image copy)
- [x] Adapt `voc2yolo_detect_labels.py` for project structure (`labels_detect` output, no image copy)
- [x] Refactor common functions (esp. path construction, XML parsing) from `voc2yolo_detect_labels.py`/`voc2yolo_segment_labels.py` scripts into `voc2yolo_utils.py`.
- [x] Ensure scripts load VOC_ROOT from .env (using python-dotenv).

### 3. Testing
- [x] Create unit tests for `segment_voc2yolo.py` core functions
- [x] Create unit tests for `voc2yolo_segment_labels.py` core functions
- [x] Add test data fixtures (XML, Mask)
- [x] Add test cases for edge cases (segmentation)
- [x] Update tests for directory structure & segmentation logic
- [x] Create unit tests for `detect_voc2yolo.py` core functions
- [x] Create unit tests for `voc2yolo_detect_labels.py` core functions
- [x] Add test cases for edge cases (detection)
- [x] Update README.md to include `voc2yolo_images.py` and visualization scripts.

### 4. Documentation
- [x] Rename `explanation.md` to `README.md`
- [x] Update `README.md` with:
  - [x] Finalized directory structure
  - [x] Label formats (detect/segment)
  - [x] Usage examples
  - [x] Script descriptions
  - [x] Testing instructions

### 5. Validation
- [x] Add validation for:
  - [x] Polygon point count (minimum 3 points) in `_mask_to_polygons`
  - [x] Coordinate normalization (0-1 range) in `_parse_xml` and `convert_box`
  - [x] Class ID mapping
  - [x] File existence checks (imagesets, annotations, masks)
- [ ] Investigate memory optimization for reading/processing large masks if needed.

### 6. Performance Optimization (Future)
- [ ] Add option for parallel processing (e.g., `multiprocessing`) to `convert` methods for faster processing of large splits.

### 7. Image Handling and Visualization
- [x] Add `opencv-python` and `python-dotenv` to requirements.txt
- [x] Implement `voc2yolo_images.py` to copy images based on image sets.
- [x] Add visualization drawing utilities (bbox, polygon, mask overlay, colors) to `src/utils/common/image_annotate.py`.
- [x] Implement `vocdev_detect_viz.py` (visualize VOC XML annotations).
- [ ] Implement `vocdev_segment_viz.py` (visualize VOC mask annotations).
- [ ] Implement `yolo_detect_viz.py` (visualize YOLO detection labels).
- [ ] Implement `yolo_segment_viz.py` (visualize YOLO segmentation labels).

### 8. Cleanup
- [x] Create `src/utils/common/` directory.
- [x] Create `src/utils/common/iou.py` and consolidate IoU logic.
- [x] Create `src/utils/common/bbox_format.py` and move format conversion.
- [x] Refactor `metrics/detection.py` to use common IoU.
- [x] Refactor `data_converter/voc2yolo_segment_labels.py` to use common IoU.
- [x] Delete `src/utils/bbox/bbox.py`.
- [x] Add tests for common utilities (`iou.py`, `bbox_format.py`).

## Notes
- Both scripts now primarily read from the `VOCdevkit` structure.
- Ensure `VOCdevkit`