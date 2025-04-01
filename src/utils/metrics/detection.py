import numpy as np  # Make sure numpy is imported at the top

from src.utils.common.iou import calculate_iou


def match_predictions(predictions, ground_truths, iou_threshold):
    """Matches predictions to ground truth boxes for a single image.

    Uses a greedy approach: predictions are sorted by confidence, and each
    prediction is matched to the highest-IoU available ground truth box of the
    same class, provided the IoU is above the threshold.

    Args:
        predictions (list): A list of predictions, where each prediction is
            represented as [box, score, class_id]. `box` is [xmin, ymin, xmax, ymax].
        ground_truths (list): A list of ground truth annotations, where each
            annotation is represented as [box, class_id]. `box` is [xmin, ymin, xmax, ymax].
        iou_threshold (float): The IoU threshold required for a match.

    Returns:
        tuple: A tuple containing:
            - match_results (list): A list where each element corresponds to an
              original prediction (in the input order), containing
              (score, is_tp, pred_class_id). `is_tp` is True if the prediction
              is a True Positive, False otherwise.
            - num_gt_per_class (dict): A dictionary mapping class_id to the
              total number of ground truth boxes for that class.
    """
    num_preds = len(predictions)
    num_gts = len(ground_truths)

    # Calculate num_gt_per_class
    num_gt_per_class = {}
    gt_boxes_per_class = {}
    for gt_idx, (gt_box, gt_class_id) in enumerate(ground_truths):
        num_gt_per_class[gt_class_id] = num_gt_per_class.get(gt_class_id, 0) + 1
        if gt_class_id not in gt_boxes_per_class:
            gt_boxes_per_class[gt_class_id] = []
        gt_boxes_per_class[gt_class_id].append((gt_idx, gt_box))

    # Sort predictions by confidence score (descending)
    # Keep track of original indices
    pred_indices = sorted(range(num_preds), key=lambda k: predictions[k][1], reverse=True)

    # Keep track of which ground truths have been matched
    gt_matched = [False] * num_gts
    # Store results keyed by original prediction index
    results_dict = {}

    for pred_idx in pred_indices:
        pred_box, pred_score, pred_class_id = predictions[pred_idx]
        best_match_gt_idx = -1
        best_match_iou = -1.0

        # Only consider ground truths of the same class
        if pred_class_id in gt_boxes_per_class:
            for gt_idx, gt_box in gt_boxes_per_class[pred_class_id]:
                # Check if this GT is already matched
                if not gt_matched[gt_idx]:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou >= iou_threshold and iou > best_match_iou:
                        best_match_iou = iou
                        best_match_gt_idx = gt_idx

        is_tp = False
        if best_match_gt_idx != -1:
            gt_matched[best_match_gt_idx] = True
            is_tp = True

        results_dict[pred_idx] = (pred_score, is_tp, pred_class_id)

    # Reorder results to match original prediction order
    match_results = [results_dict[i] for i in range(num_preds)]

    return match_results, num_gt_per_class


def calculate_pr_data(all_match_results, all_num_gt_per_class):
    """Calculates precision-recall data points across an entire dataset.

    Aggregates match results from all images, sorts them by confidence, and computes
    precision and recall values at different confidence levels for each class.

    Args:
        all_match_results (list): A concatenated list of match results from all
            images, where each element is (score, is_tp, pred_class_id).
        all_num_gt_per_class (dict): A dictionary mapping class_id to the total
            number of ground truth boxes for that class across the entire dataset.

    Returns:
        dict: A dictionary where keys are class_ids. Each value is another dictionary
              containing:
                'precision': np.array of precision values.
                'recall': np.array of recall values.
                'confidence': np.array of confidence scores corresponding to P/R points.
                'num_gt': int, the total number of ground truths for this class.
              Returns an empty dictionary if there are no predictions.
              For classes with no ground truths, precision/recall will be empty arrays.
    """
    if not all_match_results:
        return {}

    # Sort all predictions by confidence score (descending)
    all_match_results.sort(key=lambda x: x[0], reverse=True)

    # Separate results by class
    results_by_class = {}
    for score, is_tp, class_id in all_match_results:
        if class_id not in results_by_class:
            results_by_class[class_id] = []
        results_by_class[class_id].append((score, is_tp))

    pr_data = {}
    all_classes = set(all_num_gt_per_class.keys()) | set(results_by_class.keys())

    for class_id in all_classes:
        num_gt = all_num_gt_per_class.get(class_id, 0)
        class_results = results_by_class.get(class_id, [])

        if num_gt == 0 and not class_results:
            # Class exists in neither GT nor predictions
            continue

        pr_data[class_id] = {"precision": [], "recall": [], "confidence": [], "num_gt": num_gt}

        if not class_results:
            # No predictions for this class, but GT exists (all FN)
            pr_data[class_id]["precision"] = np.array([])
            pr_data[class_id]["recall"] = np.array([])
            pr_data[class_id]["confidence"] = np.array([])
            continue

        tp_cumulative = 0
        fp_cumulative = 0
        num_predictions = len(class_results)

        recalls = np.zeros(num_predictions)
        precisions = np.zeros(num_predictions)
        confidences = np.zeros(num_predictions)

        for i, (score, is_tp) in enumerate(class_results):
            if is_tp:
                tp_cumulative += 1
            else:
                fp_cumulative += 1

            # Precision = TP / (TP + FP)
            precisions[i] = tp_cumulative / (tp_cumulative + fp_cumulative)
            # Recall = TP / Total GT
            recalls[i] = tp_cumulative / num_gt if num_gt > 0 else 0.0
            confidences[i] = score

        pr_data[class_id]["precision"] = precisions
        pr_data[class_id]["recall"] = recalls
        pr_data[class_id]["confidence"] = confidences

    return pr_data


def calculate_ap(precision, recall):
    """Calculates Average Precision (AP) from precision and recall arrays.

    Uses the area under the precision-recall curve, following the standard
    VOC/COCO method (makes precision monotonically decreasing).

    Args:
        precision (np.array): Array of precision values.
        recall (np.array): Array of recall values (corresponding to precision points).
                         Assumed to be sorted in ascending order implicitly by the
                         way `calculate_pr_data` generates them based on confidence.

    Returns:
        float: The Average Precision score.
    """
    if precision.size == 0 or recall.size == 0:
        return 0.0

    # Prepend sentinel values (recall=0, precision=1) - standard practice
    # Note: Some implementations append (max_recall, 0) as well, but the monotonically
    # decreasing step effectively handles the end of the curve.
    mrec = np.concatenate(([0.0], recall))
    mpre = np.concatenate(([1.0], precision))  # Use 1.0 based on COCO eval script rationale

    # Make precision monotonically decreasing
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Find indices where recall changes
    indices = np.where(mrec[1:] != mrec[:-1])[0]

    # Compute AP using the area under the curve
    # Sum areas of rectangles: (recall_change) * precision_at_endpoint
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)


def calculate_map(ap_scores_per_class):
    """Calculates mean Average Precision (mAP) from AP scores per class.

    Args:
        ap_scores_per_class (dict): A dictionary mapping class_id to its AP score.

    Returns:
        float: The mean Average Precision (mAP) score, or 0.0 if the input
               dictionary is empty.
    """
    if not ap_scores_per_class:  # Check if dict is empty
        return 0.0

    ap_values = list(ap_scores_per_class.values())
    if not ap_values:  # Check if dict contains values (handles case like {0: None})
        return 0.0

    # Filter out potential None or non-numeric values if necessary, though calculate_ap should return float
    valid_ap_values = [ap for ap in ap_values if isinstance(ap, (int, float))]

    if not valid_ap_values:
        return 0.0

    return float(np.mean(valid_ap_values))


def calculate_map_by_size(predictions, ground_truths, gt_areas, size_ranges, iou_threshold):
    """Calculates mAP for different object size categories (Small, Medium, Large).

    Filters ground truths based on area, then runs the standard matching and AP
    calculation process for each size category using all predictions.

    Args:
        predictions (list): List of all predictions for the dataset,
            each as [box, score, class_id].
        ground_truths (list): List of all ground truths for the dataset,
            each as [box, class_id].
        gt_areas (list or np.array): List/array of areas corresponding to each
            ground truth box in `ground_truths`.
        size_ranges (dict): Dictionary defining size ranges, e.g.,
            {'small': [0, 1024], 'medium': [1024, 9216], 'large': [9216, float('inf')]}
        iou_threshold (float): The IoU threshold to use for matching.

    Returns:
        dict: A dictionary mapping size category names (e.g., 'small') to their
              corresponding mAP scores.
    """
    map_results_by_size = {}

    if len(ground_truths) != len(gt_areas):
        raise ValueError("Length of ground_truths and gt_areas must match.")

    # Convert gt_areas to numpy array for easier filtering
    gt_areas_np = np.array(gt_areas)

    for size_name, (min_area, max_area) in size_ranges.items():
        # Ensure max_area is float for comparison
        max_area = float(max_area) if max_area is not None else float("inf")

        # Find indices of ground truths within the current size range
        gt_indices_in_range = np.where((gt_areas_np >= min_area) & (gt_areas_np < max_area))[0]

        # Create the subset of ground truths for this size
        gt_subset = [ground_truths[i] for i in gt_indices_in_range]

        if not gt_subset:
            # No ground truths in this size category, mAP is undefined or 0
            map_results_by_size[size_name] = 0.0
            continue

        # Run matching using ALL predictions against the filtered GT subset
        # Note: match_predictions internally calculates num_gt_per_class for the subset
        match_results_subset, num_gt_subset = match_predictions(
            predictions, gt_subset, iou_threshold
        )

        # Calculate PR data for the subset
        pr_data_subset = calculate_pr_data(match_results_subset, num_gt_subset)

        # Calculate AP per class for the subset
        ap_scores_subset = {}
        for class_id, pr_data in pr_data_subset.items():
            ap = calculate_ap(pr_data["precision"], pr_data["recall"])
            ap_scores_subset[class_id] = ap

        # Calculate mAP for this size category
        map_size = calculate_map(ap_scores_subset)
        map_results_by_size[size_name] = map_size

    return map_results_by_size


def generate_confusion_matrix(
    predictions, ground_truths, iou_threshold, confidence_threshold, target_classes
):
    """Generates a confusion matrix for object detection results.

    Matches predictions to ground truths based on IoU and confidence thresholds.
    Compares the predicted class to the ground truth class for matches.
    Tracks False Positives (predicting object when none exists or wrong class)
    and False Negatives (failing to detect an existing object).

    Args:
        predictions (list): List of predictions for the entire dataset,
            each as [box, score, class_id].
        ground_truths (list): List of ground truth annotations for the entire
            dataset, each as [box, class_id].
        iou_threshold (float): IoU threshold for matching predictions to GTs.
        confidence_threshold (float): Minimum confidence score for a prediction
            to be considered.
        target_classes (list or set): A list/set of class IDs to show explicitly
            in the matrix. Other classes will be grouped into 'Others'.

    Returns:
        tuple: A tuple containing:
            - confusion_matrix (np.array): The confusion matrix.
              Rows: True Class (Target Classes..., Others, Background/FP)
              Cols: Predicted Class (Target Classes..., Others, Background/FP)
            - class_labels (list): The list of labels corresponding to the matrix
              indices (e.g., ['classA', 'classB', 'Others', 'Background']).
    """
    target_classes = list(target_classes)  # Ensure it's a list for indexing
    num_target_classes = len(target_classes)

    # --- Class Index Mapping ---
    # Index 0 to num_target_classes-1: Target classes
    # Index num_target_classes: 'Others' class
    # Index num_target_classes + 1: 'Background' (for FNs/FPs)
    others_idx = num_target_classes
    background_idx = num_target_classes + 1
    matrix_size = num_target_classes + 2
    confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=np.int64)
    class_labels = target_classes + ["Others", "Background"]

    target_class_to_idx = {cls_id: i for i, cls_id in enumerate(target_classes)}

    def get_mapped_idx(class_id):
        return target_class_to_idx.get(class_id, others_idx)

    # --- Filter Predictions ---
    filtered_preds = [
        (i, p[0], p[1], p[2]) for i, p in enumerate(predictions) if p[1] >= confidence_threshold
    ]
    num_filtered_preds = len(filtered_preds)

    # --- Prepare GTs ---
    num_gts = len(ground_truths)
    gt_matched = [False] * num_gts
    gt_map = [
        (i, gt[0], get_mapped_idx(gt[1])) for i, gt in enumerate(ground_truths)
    ]  # (original_idx, box, mapped_true_idx)

    # --- Prepare Preds ---
    pred_matched = [False] * num_filtered_preds
    # Sort by confidence for greedy matching (descending)
    preds_sorted = sorted(filtered_preds, key=lambda x: x[2], reverse=True)
    # Store original index of filtered_preds within preds_sorted
    original_indices_map = {item[0]: idx for idx, item in enumerate(preds_sorted)}

    # --- Matching Process (similar to mAP but simpler for confusion) ---
    # Find potential matches based on IoU
    matches = []  # Stores (pred_idx_sorted, gt_idx_original, iou)
    for i, (_, pred_box, _, pred_class_id) in enumerate(preds_sorted):
        pred_original_idx = preds_sorted[i][0]
        pred_mapped_idx = get_mapped_idx(pred_class_id)
        for j, gt_box, gt_mapped_idx in gt_map:
            # Only compare if original classes could potentially match
            # (i.e., if GT class is a target, pred must be same target; if GT is 'Others', pred must be 'Others')
            # NOTE: This simplification might slightly differ from pure mAP matching if a high-conf
            # 'target' pred overlaps best with an 'others' GT. Here we focus on CM counts.
            # A simpler approach: match purely on IoU first?
            # Let's stick to class-aware matching for now for consistency.
            if (
                predictions[pred_original_idx][2] == ground_truths[j][1]
            ):  # Compare original class IDs
                iou = calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    matches.append((i, j, iou))

    # Sort matches by IoU (descending) - needed? No, sort by confidence is primary. Let's use confidence order.
    # Sort matches by prediction confidence (already done via preds_sorted iteration)
    # matches.sort(key=lambda x: preds_sorted[x[0]][2], reverse=True)

    # Greedy assignment based on confidence order
    for pred_idx_sorted, gt_idx_original, iou in matches:
        # Check if pred or GT already matched by a higher confidence prediction (implicit due to iteration order)
        if (
            not pred_matched[original_indices_map[preds_sorted[pred_idx_sorted][0]]]
            and not gt_matched[gt_idx_original]
        ):
            pred_original_idx = preds_sorted[pred_idx_sorted][0]
            pred_mapped_idx = get_mapped_idx(predictions[pred_original_idx][2])
            true_mapped_idx = get_mapped_idx(ground_truths[gt_idx_original][1])

            # Check class consistency again (should be guaranteed by loop above)
            if predictions[pred_original_idx][2] == ground_truths[gt_idx_original][1]:
                confusion_matrix[true_mapped_idx, pred_mapped_idx] += 1
                pred_matched[original_indices_map[pred_original_idx]] = True
                gt_matched[gt_idx_original] = True

    # --- Account for Unmatched ---
    # False Negatives: Unmatched GTs
    for i in range(num_gts):
        if not gt_matched[i]:
            true_mapped_idx = get_mapped_idx(ground_truths[i][1])
            confusion_matrix[true_mapped_idx, background_idx] += 1  # FN: Predicted background

    # False Positives: Unmatched Predictions (above confidence threshold)
    for i in range(num_filtered_preds):
        original_pred_index = filtered_preds[i][0]
        if not pred_matched[original_indices_map[original_pred_index]]:
            pred_mapped_idx = get_mapped_idx(filtered_preds[i][3])
            confusion_matrix[background_idx, pred_mapped_idx] += (
                1  # FP: Predicted class, true was background
            )

    return confusion_matrix, class_labels
