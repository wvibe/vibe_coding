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
- [x] Rename `voc2yolo.py` -> `segment_voc2yolo.py`
- [x] Remove image copying logic
- [x] Add image existence verification (integrated into scripts)
- [x] Implement instance-class matching logic (IoU based)
- [x] Update `segment_voc2yolo.py` output to `labels_segment` directory
- [x] Implement actual mask reading and polygon conversion in `segment_voc2yolo.py`
- [x] Identify and move detection script -> `detect_voc2yolo.py`
- [x] Adapt `detect_voc2yolo.py` for project structure (`labels_detect` output, no image copy)
- [ ] Refactor common functions from `detect_voc2yolo.py` and `segment_voc2yolo.py` into `converter_utils.py`

### 3. Testing
- [x] Create unit tests for `segment_voc2yolo.py` core functions
- [x] Add test data fixtures (XML, Mask)
- [x] Add test cases for edge cases (segmentation)
- [x] Update tests for directory structure & segmentation logic
- [x] Create unit tests for `detect_voc2yolo.py` core functions
- [x] Add test cases for edge cases (detection)

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

### 6. Performance Optimization (Future)
- [ ] Add option for parallel processing (e.g., `multiprocessing`) to `convert` methods for faster processing of large splits.
- [ ] Investigate memory optimization for reading/processing large masks if needed.

## Notes
- Both scripts now primarily read from the `VOCdevkit` structure.
- Ensure `VOCdevkit` is present in the `/home/wmu/vibe/hub/datasets/VOC` directory.
- Images are assumed to exist either in `VOCdevkit/.../JPEGImages` or the top-level `images/` directory.

## Dependencies
- numpy
- opencv-python
- pathlib
- xml.etree.ElementTree
- logging
- tqdm