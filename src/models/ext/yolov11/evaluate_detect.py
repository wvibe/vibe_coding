"""
Runs YOLOv11 detection evaluation based on a configuration YAML file.

This script:
1. Loads a trained YOLOv11 model
2. Runs inference on a specified dataset
3. Calculates detection metrics (mAP, mAP by size, confusion matrix)
4. Reports computational metrics (parameters, inference time, memory usage)
5. Generates various plots and output files

Implementation plan:
- [x] 5.3.1: Setup script structure with single `--config` argument and basic imports
- [x] 5.3.2: Create `evaluate_default.yaml` configuration file with comprehensive options
- [x] 5.3.3: Implement configuration loading and validation (moved to utils)
- [x] 5.3.4: Add model loading with parameter counting via `get_model_params`
- [x] 5.3.5: Implement inference with warmup and measurement of time/memory
- [x] 5.3.6: Implement ground truth loading and format conversion (moved to utils)
- [x] 5.3.7: Integrate metric calculation (`match_predictions`, `calculate_map`, etc.)
- [ ] 5.3.8: Implement visualization and result saving (moved to utils, placeholder remains)
- [x] 5.3.9: Create evaluation documentation in `docs/yolov11/evaluate.md`
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml  # Keep for saving config copy in main
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Import custom metrics utilities
from vibelab.utils.metrics.compute import get_model_params, get_peak_gpu_memory_mb
from vibelab.utils.metrics.detection import (
    calculate_ap,
    calculate_map,
    calculate_pr_data,
    generate_confusion_matrix,
    match_predictions,
)

# Import from our new utility file
from .evaluate_utils import (
    load_config,
    load_ground_truth,
    save_evaluation_results,
    setup_output_directory,
)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Removed load_config - now in evaluate_utils.py

# Removed setup_output_directory - now in evaluate_utils.py


# load_model remains here as it's core model interaction
def load_model(model_path: str, device: Union[str, int]) -> Tuple[Any, int]:
    """Loads the YOLO model and returns information about it.

    Args:
        model_path: Path to the model file or a model name (e.g., "yolo11n.pt")
        device: Device to load the model on (e.g., 0, "cuda:0", "cpu")

    Returns:
        Tuple of (model, num_parameters)
    """
    logging.info(f"Loading model from {model_path} on device {device}")

    try:
        # Reset GPU memory stats before loading model
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model = YOLO(model_path)

        # Get model parameters
        num_params = get_model_params(model)
        if num_params is not None:
            logging.info(f"Model has {num_params:,} parameters")
        else:
            logging.warning("Could not determine model parameter count")
            num_params = 0  # Assign 0 if None

        # Ensure num_params is always an int
        num_params = num_params if num_params is not None else 0

        return model, num_params

    except Exception as e:
        logging.error(f"Error loading model: {e}")
        # Decide how to handle this - exit or raise?
        # Exiting might be simpler for a script.
        sys.exit(1)


def _find_and_sample_images(
    image_dir: Path, dataset_config: Dict, eval_params: Dict
) -> Optional[List[Path]]:
    """Finds images and applies sampling based on config."""
    logging.info(f"Finding images in: {image_dir}")
    if not image_dir.is_dir():
        logging.error(f"Image directory not found: {image_dir}")
        return None

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    all_image_files = []
    for ext in image_extensions:
        all_image_files.extend(list(image_dir.glob(ext)))

    if not all_image_files:
        logging.warning(f"No image files found in {image_dir}")
        return []
    else:
        logging.info(f"Found {len(all_image_files)} total images in {image_dir}.")

    # Apply Sampling
    num_to_sample = dataset_config.get("sample_num_images")
    if isinstance(num_to_sample, int) and num_to_sample > 0:
        if num_to_sample >= len(all_image_files):
            logging.info(
                f"Requested sample size ({num_to_sample}) >= total images ({len(all_image_files)}). Using all images."
            )
            image_files = all_image_files
        else:
            seed = eval_params.get("random_seed", 42)
            random.seed(seed)
            image_files = random.sample(all_image_files, k=num_to_sample)
            logging.info(
                f"Randomly sampling {len(image_files)} images for evaluation (seed={seed})..."
            )
    else:
        image_files = all_image_files

    if not image_files:
        logging.warning("No images selected for evaluation after sampling attempt.")
        return []

    logging.info(f"Proceeding with {len(image_files)} images for evaluation.")
    return image_files


def _run_warmup(
    model: Any,
    device: Union[str, int],
    warmup_iterations: int,
    image_files: List[Path],
    predict_verbose: bool,
):
    """Performs warmup inference runs."""
    is_gpu = isinstance(device, int) or (isinstance(device, str) and "cuda" in device)
    if is_gpu and warmup_iterations > 0 and image_files:
        logging.info(f"Performing {warmup_iterations} warmup iterations...")
        warmup_imgs = image_files[: min(warmup_iterations, len(image_files))]
        for img_path in warmup_imgs:
            try:
                model.predict(img_path, device=device, verbose=predict_verbose)
            except Exception as e:
                logging.warning(f"Warmup prediction failed for {img_path}: {e}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logging.info("Warmup complete.")


def _save_single_image_results(
    img_path: Path,
    predictions: List[Dict],
    output_dir: Path,
    class_names: List[str],
):
    """Saves annotated image and YOLO format txt for a single image."""
    try:
        # Load image to draw on and get dimensions
        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()  # Basic font
            # font = ImageFont.truetype("arial.ttf", 15) # Example for specific font
        except IOError:
            font = ImageFont.load_default()
            logging.warning("Default font loaded. Install desired font for better labels.")

        txt_lines = []
        color_map = plt.get_cmap("tab20", len(class_names))  # Get distinct colors

        for pred in predictions:
            box = pred["box"]  # [xmin, ymin, xmax, ymax]
            score = pred["score"]
            class_id = pred["class_id"]

            # --- Draw on image ---
            label = class_names[class_id] if 0 <= class_id < len(class_names) else f"ID_{class_id}"
            display_text = f"{label}: {score:.2f}"
            color = tuple(int(c * 255) for c in color_map(class_id)[:3])

            draw.rectangle(box, outline=color, width=2)
            text_bbox = draw.textbbox((box[0], box[1]), display_text, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((box[0], box[1]), display_text, fill=(255, 255, 255), font=font)

            # --- Prepare YOLO format txt line ---
            # Convert absolute [xmin, ymin, xmax, ymax] back to normalized [x_center, y_center, width, height]
            if img_width > 0 and img_height > 0:
                dw = 1.0 / img_width
                dh = 1.0 / img_height
                x_center = ((box[0] + box[2]) / 2.0) * dw
                y_center = ((box[1] + box[3]) / 2.0) * dh
                width = (box[2] - box[0]) * dw
                height = (box[3] - box[1]) * dh
                # Format: class_id x_center y_center width height score
                txt_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}"
                )
            else:
                logging.warning(
                    f"Skipping YOLO txt generation for {img_path.stem} due to zero image dimension."
                )

        # Save annotated image
        output_image_path = output_dir / f"{img_path.stem}_pred.jpg"
        img.save(output_image_path)

        # Save YOLO format text file
        output_txt_path = output_dir / f"{img_path.stem}_pred.txt"
        with open(output_txt_path, "w") as f:
            f.write("\n".join(txt_lines))

    except Exception as e:
        logging.error(f"Failed to save individual results for {img_path.stem}: {e}", exc_info=True)


def _perform_inference_loop(
    model: Any,
    image_files: List[Path],
    device: Union[str, int],
    conf_thres: float,
    iou_thres_nms: float,
    predict_verbose: bool,
    individual_results_dir: Optional[Path],
    class_names: List[str],
) -> Tuple[Dict[str, List[Dict]], float]:
    """Runs the main inference loop over the selected images."""
    all_predictions = {}
    total_inference_time = 0.0
    is_gpu = isinstance(device, int) or (isinstance(device, str) and "cuda" in device)

    if is_gpu and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.synchronize()

    logging.info("Running inference on selected dataset...")
    start_time_all = time.perf_counter()

    for img_path in tqdm(image_files, desc="Inference"):
        try:
            if is_gpu and torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time_img = time.perf_counter()

            results_list: List[Results] = model.predict(
                img_path,
                conf=conf_thres,
                iou=iou_thres_nms,
                device=device,
                verbose=predict_verbose,
            )

            if is_gpu and torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time_img = time.perf_counter()
            total_inference_time += end_time_img - start_time_img

            if results_list:
                res = results_list[0]
                boxes = res.boxes
                image_predictions = []
                if boxes is not None and boxes.xyxy is not None:
                    boxes_xyxy = boxes.xyxy.cpu().tolist()
                    scores = (
                        boxes.conf.cpu().tolist()
                        if boxes.conf is not None
                        else [0.0] * len(boxes_xyxy)
                    )
                    class_ids = (
                        boxes.cls.cpu().tolist() if boxes.cls is not None else [0] * len(boxes_xyxy)
                    )
                    for box, score, cls_id in zip(boxes_xyxy, scores, class_ids):
                        image_predictions.append(
                            {"box": box, "score": score, "class_id": int(cls_id)}
                        )
                all_predictions[img_path.stem] = image_predictions
            else:
                image_predictions = []  # Ensure defined even if no results
                all_predictions[img_path.stem] = []

            # --- Save individual results if enabled ---
            if individual_results_dir is not None and image_predictions:
                _save_single_image_results(
                    img_path, image_predictions, individual_results_dir, class_names
                )

        except Exception as e:
            logging.error(f"Inference failed for image {img_path}: {e}", exc_info=True)
            all_predictions[img_path.stem] = []

    end_time_all = time.perf_counter()
    total_inference_wall_time = end_time_all - start_time_all
    logging.info(f"Inference loop wall time: {total_inference_wall_time:.2f} seconds")

    return all_predictions, total_inference_time


# Removed load_ground_truth - now in evaluate_utils.py


def _aggregate_results(predictions: Dict, ground_truths: Dict) -> Tuple[List, List, List]:
    """Flattens predictions and ground truths for metric calculation."""
    all_flat_preds = []  # List of [box, score, class_id]
    all_flat_gts = []  # List of [box, class_id]
    image_stems = list(predictions.keys())

    for stem in image_stems:
        preds = predictions.get(stem, [])
        gts = ground_truths.get(stem, [])
        for p in preds:
            all_flat_preds.append([p["box"], float(p["score"]), int(p["class_id"])])
        for gt in gts:
            all_flat_gts.append([gt["box"], int(gt["class_id"])])
    return all_flat_preds, all_flat_gts, image_stems


def _calculate_map_at_iou(
    iou_threshold: float,
    predictions: Dict,
    ground_truths: Dict,
    image_stems: List[str],
    class_names: List[str],
) -> Tuple[Dict, Dict, int, Dict]:
    """Calculates PR data, AP per class, and mAP for a single IoU threshold."""
    logging.info(f"Matching predictions to ground truths at IoU={iou_threshold}...")
    aggregated_match_results = []
    aggregated_num_gt_per_class = dict.fromkeys(
        range(len(class_names)), 0
    )  # Initialize for all classes

    for stem in tqdm(image_stems, desc=f"Matching @{iou_threshold}"):
        img_preds = [
            [p["box"], float(p["score"]), int(p["class_id"])] for p in predictions.get(stem, [])
        ]
        img_gts = [[gt["box"], int(gt["class_id"])] for gt in ground_truths.get(stem, [])]
        if not img_preds and not img_gts:
            continue

        match_results, num_gt_per_class = match_predictions(img_preds, img_gts, iou_threshold)
        aggregated_match_results.extend(match_results)
        for class_id, count in num_gt_per_class.items():
            if class_id in aggregated_num_gt_per_class:
                aggregated_num_gt_per_class[class_id] += count
            else:  # Should not happen if initialized, but safe check
                aggregated_num_gt_per_class[class_id] = count

    total_gt_count = sum(aggregated_num_gt_per_class.values())

    logging.info(f"Calculating PR data and AP/mAP at IoU={iou_threshold}...")
    pr_data = calculate_pr_data(aggregated_match_results, aggregated_num_gt_per_class)

    ap_scores = {}  # By class ID
    ap_scores_by_name = {}  # By class name
    for i, name in enumerate(class_names):
        class_id = i
        ap = 0.0
        if (
            class_id in pr_data
            and pr_data[class_id]["recall"] is not None
            and pr_data[class_id]["precision"] is not None
        ):
            precision = pr_data[class_id]["precision"]
            recall = pr_data[class_id]["recall"]
            if isinstance(precision, list):
                precision = np.array(precision)
            if isinstance(recall, list):
                recall = np.array(recall)
            if precision.size > 0 and recall.size > 0:
                ap = calculate_ap(precision, recall)
        ap_scores[class_id] = ap
        ap_scores_by_name[name] = ap

    map_score = calculate_map(ap_scores)  # mAP for this IoU
    return pr_data, ap_scores_by_name, total_gt_count, ap_scores


def _calculate_map_coco(
    predictions: Dict,
    ground_truths: Dict,
    image_stems: List[str],
    class_names: List[str],
    map_iou_range: np.ndarray,
) -> float:
    """Calculates COCO-style mAP@0.5:0.95."""
    logging.info("Calculating mAP @ 0.5:0.95...")
    ap_scores_per_iou = {iou: {} for iou in map_iou_range}
    # Need the total GT count once
    aggregated_num_gt_per_class = dict.fromkeys(range(len(class_names)), 0)
    for stem in image_stems:
        img_gts = ground_truths.get(stem, [])
        for gt in img_gts:
            class_id = gt["class_id"]
            if class_id in aggregated_num_gt_per_class:
                aggregated_num_gt_per_class[class_id] += 1

    for iou in tqdm(map_iou_range, desc="Calculating AP per IoU"):
        aggregated_match_results_iou = []
        for stem in image_stems:
            img_preds = [
                [p["box"], float(p["score"]), int(p["class_id"])] for p in predictions.get(stem, [])
            ]
            img_gts = [[gt["box"], int(gt["class_id"])] for gt in ground_truths.get(stem, [])]
            if not img_preds and not img_gts:
                continue
            match_results, _ = match_predictions(img_preds, img_gts, iou)
            aggregated_match_results_iou.extend(match_results)

        pr_data_iou = calculate_pr_data(aggregated_match_results_iou, aggregated_num_gt_per_class)
        for i, name in enumerate(class_names):
            class_id = i
            ap = 0.0
            if (
                class_id in pr_data_iou
                and pr_data_iou[class_id]["recall"] is not None
                and pr_data_iou[class_id]["precision"] is not None
            ):
                precision = pr_data_iou[class_id]["precision"]
                recall = pr_data_iou[class_id]["recall"]
                if isinstance(precision, list):
                    precision = np.array(precision)
                if isinstance(recall, list):
                    recall = np.array(recall)
                if precision.size > 0 and recall.size > 0:
                    ap = calculate_ap(precision, recall)
            ap_scores_per_iou[iou][class_id] = ap

    mean_ap_per_class = {}
    for i, name in enumerate(class_names):
        class_id = i
        class_aps = [ap_scores_per_iou[iou].get(class_id, 0.0) for iou in map_iou_range]
        mean_ap_per_class[class_id] = np.mean(class_aps)

    map_50_95 = calculate_map(mean_ap_per_class)
    return float(map_50_95)


def _calculate_confusion_matrix(
    all_flat_preds: List,
    all_flat_gts: List,
    class_names: List[str],
    conf_threshold_cm: float,
    iou_threshold_cm: float,
) -> Tuple[Optional[List], Optional[List]]:
    """Generates the confusion matrix data and labels."""
    logging.info(
        f"Generating Confusion Matrix (Conf={conf_threshold_cm}, IoU={iou_threshold_cm})..."
    )
    try:
        cm_predictions = [p for p in all_flat_preds if p[1] >= conf_threshold_cm]
        target_class_ids = list(range(len(class_names)))
        cm_ground_truths = [[gt[0], int(gt[1])] for gt in all_flat_gts]
        confusion_matrix_data, cm_labels = generate_confusion_matrix(
            predictions=cm_predictions,
            ground_truths=cm_ground_truths,
            iou_threshold=iou_threshold_cm,
            confidence_threshold=conf_threshold_cm,
            target_classes=target_class_ids,
        )
        cm_class_labels_named = [
            class_names[int(id)] if isinstance(id, int) and id < len(class_names) else str(id)
            for id in cm_labels[:-1]
        ] + [cm_labels[-1]]
        return confusion_matrix_data.tolist(), cm_class_labels_named
    except Exception as e:
        logging.error(f"Failed to generate confusion matrix: {e}", exc_info=True)
        return None, class_names + ["Background", "Error"]


# Refactored calculate_all_metrics
def calculate_all_metrics(
    predictions: Dict[str, List[Dict]],
    ground_truths: Dict[str, List[Dict]],
    metrics_params: Dict,
    class_names: List[str],
) -> Dict[str, Any]:
    """Calculates all detection metrics based on predictions and GTs."""
    logging.info("--- Calculating Metrics ---")

    map_iou_threshold = metrics_params.get("map_iou_threshold", 0.5)
    map_iou_range = np.round(np.arange(0.5, 1.0, 0.05), 2)
    conf_threshold_cm = metrics_params.get("conf_threshold_cm", 0.45)
    iou_threshold_cm = metrics_params.get("iou_threshold_cm", 0.5)

    all_flat_preds, all_flat_gts, image_stems = _aggregate_results(predictions, ground_truths)

    if not all_flat_preds:
        logging.warning("No predictions found. Skipping metric calculation.")
        # ... (return default empty metrics)
    if not all_flat_gts:
        logging.warning("No ground truths found. mAP will be 0, CM based only on FPs.")

    # --- Calculate mAP @ 0.5 (and related metrics) ---
    pr_data_50, ap_scores_50_by_name, total_gt_count, ap_scores_50 = _calculate_map_at_iou(
        map_iou_threshold, predictions, ground_truths, image_stems, class_names
    )
    map_50 = calculate_map(ap_scores_50)  # Use calculate_map on the dict by ID

    # --- Calculate mAP @ 0.5:0.95 ---
    map_50_95 = _calculate_map_coco(
        predictions, ground_truths, image_stems, class_names, map_iou_range
    )

    # --- Generate Confusion Matrix ---
    cm_data_list, cm_labels_named = _calculate_confusion_matrix(
        all_flat_preds, all_flat_gts, class_names, conf_threshold_cm, iou_threshold_cm
    )

    # --- Assemble Results ---
    detection_metrics = {
        "mAP_50": float(map_50),
        "mAP_50_95": float(map_50_95),
        "ap_per_class_50": {name: float(ap) for name, ap in ap_scores_50_by_name.items()},
        "confusion_matrix": cm_data_list,
        "confusion_matrix_labels": cm_labels_named,
        "pr_data_50": pr_data_50,
        "total_ground_truths": total_gt_count,
        "num_predictions_processed": len(all_flat_preds),
    }
    logging.info("Metric calculation complete.")
    logging.info(f"  mAP@0.50   : {detection_metrics['mAP_50']:.4f}")
    logging.info(f"  mAP@0.5:0.95: {detection_metrics['mAP_50_95']:.4f}")
    return detection_metrics


# Removed save_evaluation_results - now in evaluate_utils.py


def main():
    """Main entry point for the evaluation script orchestrating the steps."""
    # --- Setup ---
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 detection model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the evaluation configuration YAML file"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        output_dir = setup_output_directory(
            config
        )  # setup_output_dir already handles individual dir creation if flag is set
    except Exception as e:
        logging.error(f"Failed during setup: {e}")
        sys.exit(1)

    # Derive configuration sections
    output_config = config.get("output", {})
    eval_params = config.get("evaluation_params", {})
    dataset_config = config.get("dataset", {})
    metrics_params = config.get("metrics", {})
    class_names = dataset_config.get("class_names", [])
    # Read the renamed flag
    save_results_flag = output_config.get("save_results", False)
    # Derive directory path based on the renamed flag
    individual_results_dir = output_dir / "individual_results" if save_results_flag else None

    # Save a copy of the configuration
    config_save_path = output_dir / "config.yaml"
    try:
        with open(config_save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        logging.info(f"Saved configuration to {config_save_path}")
    except Exception as e:
        logging.error(f"Failed to save config copy: {e}")

    # --- Load Model ---
    model_path = config["model"]
    device = eval_params.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    model, num_params = load_model(model_path, device)

    # --- Prepare for Inference ---
    image_dir = Path(dataset_config["image_dir"])
    image_files = _find_and_sample_images(image_dir, dataset_config, eval_params)

    all_predictions = {}
    compute_stats = {}
    detection_metrics = {}
    all_ground_truths = {}

    if image_files is None:  # Error finding directory
        compute_stats = {"error": f"Image directory not found: {image_dir}"}
    elif not image_files:  # No images found/sampled
        compute_stats = {
            "num_model_params": num_params,
            "avg_inference_time_ms": 0,
            "peak_gpu_memory_mb": 0,
            "device": str(device),
            "num_images_processed": 0,
            "warning": "No images found/selected",
        }
    else:
        # --- Run Inference ---
        _run_warmup(
            model,
            device,
            eval_params.get("warmup_iterations", 3),
            image_files,
            eval_params.get("predict_verbose", False),
        )

        all_predictions, total_inference_time = _perform_inference_loop(
            model=model,
            image_files=image_files,
            device=device,
            conf_thres=eval_params.get("conf_thres", 0.25),
            iou_thres_nms=eval_params.get("iou_thres_nms", 0.65),
            predict_verbose=eval_params.get("predict_verbose", False),
            individual_results_dir=individual_results_dir,  # Pass the calculated path
            class_names=class_names,
        )

        # --- Calculate Compute Stats ---
        num_images = len(image_files)
        avg_inference_time_ms = (total_inference_time / num_images * 1000) if num_images > 0 else 0
        peak_gpu_memory_mb = 0
        is_gpu = isinstance(device, int) or (isinstance(device, str) and "cuda" in device)
        if is_gpu and torch.cuda.is_available():
            peak_gpu_memory_mb = get_peak_gpu_memory_mb(device)
        compute_stats = {
            "num_model_params": num_params,
            "avg_inference_time_ms": avg_inference_time_ms,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "device": str(device),
            "num_images_processed": num_images,
        }
        logging.info(f"Compute Stats: {compute_stats}")

        # --- Load Ground Truth ---
        # Only load GT for the images that were actually processed
        image_stems = list(all_predictions.keys())
        if image_stems:  # Only proceed if we actually have prediction results
            label_dir = Path(dataset_config["label_dir"])
            all_ground_truths = load_ground_truth(
                label_dir=label_dir,
                image_dir=image_dir,
                image_stems=image_stems,
                class_names=class_names,
            )

            # --- Calculate Metrics ---
            detection_metrics = calculate_all_metrics(
                predictions=all_predictions,
                ground_truths=all_ground_truths,
                metrics_params=metrics_params,
                class_names=class_names,
            )
        else:
            logging.warning("No predictions generated, skipping GT loading and metric calculation.")
            # Leave all_ground_truths and detection_metrics as empty dicts

    # --- Handle potential compute errors / no images processed ---
    if compute_stats.get("error") or compute_stats.get("num_images_processed", 0) == 0:
        log_message = compute_stats.get("error") or compute_stats.get("warning", "Unknown issue")
        logging.error(f"Evaluation incomplete: {log_message}")
        # Save partial results if possible (config, compute_stats with error/warning)
        save_evaluation_results(output_dir, config, compute_stats, detection_metrics, None, None)
        sys.exit(1)

    # --- Save Final Results ---
    save_evaluation_results(
        output_dir=output_dir,
        config=config,
        compute_stats=compute_stats,
        detection_metrics=detection_metrics,
        # Optionally pass predictions/GTs if saving them is desired later
        # predictions=all_predictions,
        # ground_truths=all_ground_truths
    )

    logging.info(f"Evaluation script finished. Results are in {output_dir}")


if __name__ == "__main__":
    main()
