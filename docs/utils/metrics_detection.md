> **[2024-05 MIGRATION]** Source code for these utilities has moved to `src/vibelab/utils/metrics/detection.py` as part of the vibelab migration. All tests and references have been updated.

# Detection Metrics Utilities Design Notes

This document outlines the design choices and rationale behind the functions in `src/vibelab/utils/metrics/detection.py`.

## `match_predictions(predictions, ground_truths, iou_threshold)`

This function is responsible for matching predicted bounding boxes to ground truth boxes for a single image. Its primary goal is to determine, for a given IoU threshold, whether each prediction is a True Positive (TP) or a False Positive (FP), which are essential inputs for calculating metrics like Precision, Recall, and Average Precision (AP).

### Role of Confidence Scores

- **Input:** The function takes predictions that include a confidence score (`[box, score, class_id]`).
- **No Internal Thresholding:** Crucially, this function does *not* apply a confidence score threshold internally. It processes *all* provided predictions.
- **Purpose:** The scores are retained and associated with the TP/FP status (`(score, is_tp, pred_class_id)`) in the output. This is necessary because downstream AP calculations require sorting all predictions by confidence to generate the Precision-Recall curve across different operating points (confidence thresholds). Applying a threshold here would prematurely discard data needed for the full AP calculation. Filtering by a specific confidence threshold, if desired for other purposes, should be done *after* calling this function.

### Matching Strategy: Class-First, Confidence-Based Greedy

The matching process follows the standard methodology used in benchmarks like COCO mAP:

1.  **Class Constraint:** A prediction can *only* be matched to a ground truth box if they share the **same class label**. High IoU overlap with a ground truth of a *different* class is ignored for matching purposes and results in an FP for the prediction. This emphasizes the importance of correct classification in the detection task.
2.  **Confidence Sorting:** Predictions are processed in descending order of their confidence scores.
3.  **Greedy Matching:** For a given prediction, the algorithm searches for the *best available* (unmatched) ground truth box *of the same class* that has an IoU overlap greater than or equal to the specified `iou_threshold`.
4.  **One Match per GT:** Each ground truth box can be matched at most *once*. The highest-confidence prediction that successfully matches a specific GT "claims" it.
5.  **Outcome:**
    *   If a prediction finds a valid, available, same-class GT match above the IoU threshold, it's marked as a TP (`is_tp = True`). The matched GT is marked as used.
    *   If a prediction cannot find such a match (due to no same-class GT, low IoU, or the best-matching GT already being claimed by a higher-confidence prediction), it's marked as an FP (`is_tp = False`).

- **Rationale:** This class-first, greedy approach is the standard for mAP calculation because it evaluates the combined localization and classification performance, ensuring that only class-correct detections contribute to True Positives. While other strategies like IoU-first matching (e.g., using the Hungarian algorithm) exist, they don't align with the standard mAP definition.

### Handling Mismatched Numbers & Outputs

- **More Predictions than GTs:** Extra predictions that don't find a valid match become FPs (returned as `(score, False, class_id)`).
- **Fewer Predictions than GTs (Missed Detections):** Ground truths that are never matched by a valid prediction represent potential False Negatives (FNs).
- **Outputs:**
    - `match_results`: A list containing `(score, is_tp, pred_class_id)` for *every* input prediction, preserving the original order. This directly identifies TPs and FPs.
    - `num_gt_per_class`: A dictionary mapping `class_id` to the total count of ground truth boxes for that class in the image.
- **Deriving FNs:** False Negatives are not explicitly listed but can be calculated per class after processing all images: `FNs_class_X = total_GTs_class_X - total_TPs_class_X`. The function provides the components needed for this calculation.

## `calculate_pr_data(all_match_results, all_num_gt_per_class)`

This function takes the aggregated results from `match_predictions` across all images in a dataset and calculates the data points needed to construct Precision-Recall (PR) curves for each class.

### Purpose

After matching individual predictions, we need to evaluate performance across the entire dataset. This function aggregates all TP/FP results and computes how precision and recall evolve as we vary the confidence threshold, providing the necessary input for plotting PR curves and calculating Average Precision (AP).

### Logic

1.  **Inputs:** Takes the combined list of `(score, is_tp, pred_class_id)` tuples from all images (`all_match_results`) and the total ground truth counts per class (`all_num_gt_per_class`).
2.  **Sort by Confidence:** The `all_match_results` list is sorted by confidence score in descending order. This simulates iterating through predictions from most to least confident.
3.  **Process per Class:** The function processes each class present in either the predictions or ground truths.
4.  **Accumulate TP/FP:** For each class, it iterates through the sorted predictions belonging to that class. It keeps a running count of accumulated True Positives (`tp_cumulative`) and False Positives (`fp_cumulative`).
5.  **Calculate Precision/Recall Points:** After processing each prediction in the sorted list, it calculates:
    *   `Precision = tp_cumulative / (tp_cumulative + fp_cumulative)`
    *   `Recall = tp_cumulative / total_num_gt_for_class` (Handles `num_gt = 0`)
6.  **Output:** Returns a dictionary where keys are class IDs. Each value is another dictionary containing NumPy arrays for `precision`, `recall`, and `confidence` (the scores at which the P/R points were calculated), along with the total `num_gt` for that class. This structure represents the data points for the PR curve of each class.

### Usage

This function is typically called after iterating through the entire validation dataset and collecting all results from `match_predictions`. Its output is the direct input for subsequent AP calculation functions (like `calculate_ap`) and for plotting PR curves.

## `calculate_ap(precision, recall)`

This function computes the Average Precision (AP) for a single class, given its precision and recall values.

### Purpose

The AP summarizes the shape of the Precision-Recall curve into a single score, representing the weighted average of precisions achieved at each recall threshold.

### Logic

1.  **Inputs:** Takes NumPy arrays `precision` and `recall` obtained from `calculate_pr_data` for one class.
2.  **Handle Empty Input:** Returns 0.0 if either input array is empty.
3.  **Prepend Sentinel Values:** Adds points (recall=0, precision=1) to the beginning of the arrays. This is standard practice to ensure the curve starts correctly.
4.  **Make Precision Monotonically Decreasing:** Iterates backward through the precision array, ensuring that the precision at any point is the maximum precision seen from that point onward (`mpre[i-1] = max(mpre[i-1], mpre[i])`). This smooths out the curve according to standard evaluation methods (like PASCAL VOC / COCO).
5.  **Calculate Area:** Finds the indices where recall values change. It then calculates the AP by summing the areas of the rectangles formed under the monotonically decreasing precision curve: `Area = sum(delta_recall * precision_at_end_of_interval)`. `delta_recall` is the change in recall between points where recall changes, and `precision_at_end_of_interval` is the (monotonically adjusted) precision at the higher recall point.
6.  **Output:** Returns the calculated AP score as a float.

### Example Walkthrough

Assume `calculate_pr_data` produced the following raw Precision (P) and Recall (R) arrays for a class with 3 total Ground Truths:

```
R = [0.33, 0.67, 0.67, 1.0, 1.0]
P = [1.0, 1.0, 0.67, 0.75, 0.6]
```

1.  **Prepend Sentinels:**
    ```
    R' = [0.0, 0.33, 0.67, 0.67, 1.0, 1.0]
    P' = [1.0, 1.0, 1.0, 0.67, 0.75, 0.6]
    ```
2.  **Make Precision Monotonically Decreasing:** Working backwards:
    - `max(P'[5]) = max(0.6) -> P''[5] = 0.6`
    - `max(P'[4], P''[5]) = max(0.75, 0.6) -> P''[4] = 0.75`
    - `max(P'[3], P''[4]) = max(0.67, 0.75) -> P''[3] = 0.75`
    - `max(P'[2], P''[3]) = max(1.0, 0.75) -> P''[2] = 1.0`
    - `max(P'[1], P''[2]) = max(1.0, 1.0) -> P''[1] = 1.0`
    - `max(P'[0], P''[1]) = max(1.0, 1.0) -> P''[0] = 1.0`
    Result:
    ```
    P'' = [1.0, 1.0, 1.0, 0.75, 0.75, 0.6]
    R'  = [0.0, 0.33, 0.67, 0.67, 1.0, 1.0]
    ```
3.  **Calculate Area:** Find where `R'` changes (indices 1, 2, 4) and sum the areas:
    - Interval 1 (Recall 0.0 -> 0.33): `delta_R = R'[1]-R'[0] = 0.33`. Precision at end = `P''[1] = 1.0`. Area = `0.33 * 1.0 = 0.33`.
    - Interval 2 (Recall 0.33 -> 0.67): `delta_R = R'[2]-R'[1] = 0.34`. Precision at end = `P''[2] = 1.0`. Area = `0.34 * 1.0 = 0.34`.
    - Interval 3 (Recall 0.67 -> 1.0): `delta_R = R'[4]-R'[3] = 0.33`. Precision at end = `P''[4] = 0.75`. Area = `0.33 * 0.75 = 0.2475`.
    Total Area (AP) = `0.33 + 0.34 + 0.2475 = 0.9175`.

## `calculate_map(ap_scores_per_class)`

This function computes the mean Average Precision (mAP) across all classes.

### Purpose

The mAP provides a single, overall performance score for the object detector across all object categories.

### Logic

1.  **Input:** Takes a dictionary `ap_scores_per_class` where keys are class IDs and values are the corresponding AP scores calculated by `calculate_ap`.
2.  **Handle Empty Input:** Returns 0.0 if the input dictionary is empty or contains no valid numeric AP scores.
3.  **Calculate Mean:** Extracts the valid AP score values from the dictionary and calculates their mean using `numpy.mean()`.
4.  **Output:** Returns the calculated mAP score as a float.

## `calculate_map_by_size(predictions, ground_truths, gt_areas, size_ranges, iou_threshold)`

Calculates the mean Average Precision (mAP) score separately for different ground truth object size categories (e.g., Small, Medium, Large).

### Purpose

Object detectors can perform differently based on the size of the object. This metric helps diagnose performance issues related to object scale. It follows the standard COCO evaluation methodology for mAP@[Small/Medium/Large].

### Logic

1.  **Inputs:** Takes the list of all `predictions`, the list of all `ground_truths` (`[box, class_id]`), a parallel list/array of `gt_areas`, a `size_ranges` dictionary defining the area boundaries for each category, and the `iou_threshold` for matching.
2.  **Iterate Sizes:** Loops through each size category defined in `size_ranges` (e.g., 'small', 'medium', 'large').
3.  **Filter Ground Truths:** For the current size category, it identifies and selects only the ground truths whose area falls within the specified range (`min_area <= area < max_area`).
4.  **Match & Evaluate Subset:**
    *   It runs the standard `match_predictions` function using *all* predictions against the *filtered subset* of ground truths for the current size.
    *   It then calculates the PR data (`calculate_pr_data`) based on these matches.
    *   It calculates the AP score for each class (`calculate_ap`) using the PR data for this size subset.
    *   Finally, it calculates the mAP (`calculate_map`) across all classes for this specific size category.
5.  **Handle Empty Categories:** If a size category contains no ground truths, its mAP is recorded as 0.0.
6.  **Output:** Returns a dictionary mapping the size category names (e.g., 'small') to their calculated mAP scores.

### Usage

This function is called by the main evaluation script after collecting all predictions and ground truths (including GT areas). It requires the definition of size ranges (typically matching COCO standards) and the desired IoU threshold for AP calculation.

## `generate_confusion_matrix(predictions, ground_truths, iou_threshold, confidence_threshold, target_classes)`

Generates a confusion matrix summarizing the detection performance, focusing on classification accuracy for detected objects and identifying background mistakes.

### Purpose

While mAP provides an overall score, a confusion matrix gives a more detailed view of *which* classes are being confused with each other, and the balance between missing objects (False Negatives) and detecting non-existent objects (False Positives).

### Logic

1.  **Inputs:** Takes lists of all `predictions` and `ground_truths` for the dataset, an `iou_threshold` for matching, a `confidence_threshold` to filter predictions, and a list of `target_classes` to show explicitly.
2.  **Class Mapping:** Creates a mapping from original class IDs to matrix indices:
    *   Target classes get indices `0` to `N-1`.
    *   All other classes are mapped to an `Others` index (`N`).
    *   A `Background` index (`N+1`) is added to represent FPs (predicted object when none exists) and FNs (missed GT objects).
3.  **Filter Predictions:** Removes predictions with confidence below `confidence_threshold`.
4.  **Matching:** Performs a greedy, confidence-based matching similar to the AP calculation:
    *   Predictions are sorted by confidence (descending).
    *   For each prediction, it looks for the best-overlapping *available* ground truth *of the exact same original class* with IoU >= `iou_threshold`.
    *   If a match is found: Increment the matrix cell `[mapped_true_idx, mapped_pred_idx]`. Mark both the prediction and the GT as matched.
5.  **Unmatched GTs (FNs):** Iterates through all ground truths. If a GT was not matched, increment the matrix cell `[mapped_true_idx, background_idx]` (False Negative for that true class).
6.  **Unmatched Predictions (FPs):** Iterates through all *filtered* predictions. If a prediction was not matched, increment the matrix cell `[background_idx, mapped_pred_idx]` (False Positive for that predicted class).
7.  **Output:** Returns the `confusion_matrix` (NumPy array) and the corresponding `class_labels` list.

### Matrix Interpretation

- **Rows:** Represent the True Class (Target Classes..., Others, Background).
- **Columns:** Represent the Predicted Class (Target Classes..., Others, Background).
- `matrix[i, j]` (where `i, j` are indices for target/other classes): Count of objects of true class `i` that were predicted as class `j`.
- `matrix[i, background_idx]`: Count of objects of true class `i` that were missed (False Negatives).
- `matrix[background_idx, j]`: Count of predictions for class `j` that did not correspond to any true object (False Positives).
- `matrix[background_idx, background_idx]`: Should ideally remain 0 (represents true background correctly identified as background, which isn't explicitly counted here).