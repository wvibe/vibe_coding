"""
Utility functions for the YOLOv11 detection evaluation script.
"""

import json
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from PIL import Image
from tqdm import tqdm


def load_config(config_path: str) -> Dict:
    """Loads and performs basic validation on the YAML config file.

    Args:
        config_path: Path to the configuration YAML file

    Returns:
        Dict containing the loaded configuration

    Raises:
        SystemExit: If the config file is not found, invalid, or missing required keys
    """
    config_path = Path(config_path)
    logging.info(f"Loading configuration from: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError("Config file is empty or invalid.")

        # Validate required top-level sections
        required_sections = ["model", "dataset", "evaluation_params", "metrics"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config.")

        # Validate required dataset fields
        required_dataset_fields = ["image_dir", "label_dir", "class_names"]
        for field in required_dataset_fields:
            if field not in config["dataset"]:
                raise ValueError(f"Missing required field '{field}' in dataset section.")

        # Ensure class_names is a list
        if (
            not isinstance(config["dataset"]["class_names"], list)
            or not config["dataset"]["class_names"]
        ):
            raise ValueError("class_names must be a non-empty list in dataset section.")

        return config

    except FileNotFoundError:
        logging.error(f"Error: Configuration file not found at {config_path}")
        # Re-raise or sys.exit? Re-raising might be better for callers to handle.
        raise  # Or sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise  # Or sys.exit(1)
    except ValueError as e:
        logging.error(f"Error in config file: {e}")
        raise  # Or sys.exit(1)


def setup_output_directory(config: Dict) -> Path:
    """Creates the output directory and potentially subdirs based on the configuration.

    Args:
        config: The loaded configuration dictionary

    Returns:
        Path object for the main output directory.
        # Returns tuple: (main_output_dir, individual_results_dir or None)
    """
    output_config = config.get("output", {})
    project_dir = Path(output_config.get("project", "runs/evaluate/detect"))

    name = output_config.get("name")
    if not name:
        model_name = Path(config["model"]).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{model_name}_{timestamp}"

    output_dir = project_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")

    individual_results_dir = None
    if output_config.get("save_results", False):
        individual_results_dir = output_dir / "individual_results"
        individual_results_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created individual results directory: {individual_results_dir}")

    # Return only the main dir for now, individual dir path can be derived
    # or passed explicitly if needed elsewhere.
    return output_dir  # Keep return type simple


def _get_image_dimensions(image_path: Path) -> Optional[Tuple[int, int]]:
    """Reads the dimensions of a single image file."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as img_e:
        logging.error(f"Error reading image dimensions for {image_path}: {img_e}", exc_info=True)
        return None


def _parse_label_file(label_path: Path, img_width: int, img_height: int) -> Tuple[List[Dict], int]:
    """Parses a single YOLO format label file and converts coordinates."""
    img_gts = []
    parse_errors = 0
    try:
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                logging.warning(
                    f"Skipping invalid line ({line_num + 1}) in {label_path}: expected 5 values, got {len(parts)}"
                )
                parse_errors += 1
                continue

            try:
                class_id = int(parts[0])
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])

                if not (
                    0 <= x_center_norm <= 1
                    and 0 <= y_center_norm <= 1
                    and 0 <= width_norm <= 1
                    and 0 <= height_norm <= 1
                ):
                    logging.warning(
                        f"Skipping invalid normalized value in line {line_num + 1} of {label_path}: {parts[1:]}"
                    )
                    parse_errors += 1
                    continue

                x_center_abs = x_center_norm * img_width
                y_center_abs = y_center_norm * img_height
                width_abs = width_norm * img_width
                height_abs = height_norm * img_height

                xmin = x_center_abs - width_abs / 2.0
                ymin = y_center_abs - height_abs / 2.0
                xmax = x_center_abs + width_abs / 2.0
                ymax = y_center_abs + height_abs / 2.0

                xmin = max(0.0, xmin)
                ymin = max(0.0, ymin)
                xmax = min(img_width, xmax)
                ymax = min(img_height, ymax)

                if xmax > xmin and ymax > ymin:
                    img_gts.append({"box": [xmin, ymin, xmax, ymax], "class_id": class_id})
                else:
                    logging.warning(
                        f"Skipping zero-area box after conversion in line {line_num + 1} of {label_path}"
                    )

            except ValueError as ve:
                logging.warning(
                    f"Skipping invalid numeric value in line {line_num + 1} of {label_path}: {ve}"
                )
                parse_errors += 1
                continue

    except Exception as e:
        logging.error(f"Error processing label file {label_path}: {e}", exc_info=True)
        parse_errors += 1  # Count file processing error
        return [], parse_errors  # Return empty list if file fails

    return img_gts, parse_errors


def load_ground_truth(
    label_dir: Path,
    image_dir: Path,
    image_stems: List[str],
    class_names: List[str],
) -> Dict[str, List[Dict]]:
    """Loads ground truth annotations from YOLO format files, determining image dimensions."""
    logging.info(f"--- Loading Ground Truth from: {label_dir} (using images from {image_dir}) ---")
    all_ground_truths = {stem: [] for stem in image_stems}
    total_missing_labels = 0
    total_parse_errors = 0
    total_image_read_errors = 0

    for stem in tqdm(image_stems, desc="Loading GT Labels & Images"):
        label_path = label_dir / f"{stem}.txt"
        image_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            potential_path = image_dir / f"{stem}{ext}"
            if potential_path.is_file():
                image_path = potential_path
                break

        has_label = label_path.is_file()
        has_image = image_path is not None

        if not has_label and not has_image:
            logging.warning(f"Neither label nor image file found for stem: {stem}. Skipping.")
            continue
        if not has_label:
            total_missing_labels += 1
            continue  # Valid case: image with no GT objects
        if not has_image:
            logging.warning(f"Label file found but no image file for stem: {stem}. Cannot load GT.")
            total_image_read_errors += 1
            continue

        # Get Image Dimensions
        dimensions = _get_image_dimensions(image_path)
        if dimensions is None:
            total_image_read_errors += 1
            continue
        img_width, img_height = dimensions

        # Process Label File
        img_gts, parse_errors = _parse_label_file(label_path, img_width, img_height)
        total_parse_errors += parse_errors
        all_ground_truths[stem] = img_gts

    if total_missing_labels > 0:
        logging.info(
            f"Note: {total_missing_labels} label files were not found (expected for images with no objects)."
        )
    if total_parse_errors > 0:
        logging.warning(f"Encountered {total_parse_errors} errors while parsing label files.")
    if total_image_read_errors > 0:
        logging.error(f"Failed to read dimensions/find image for {total_image_read_errors} stems.")

    valid_gts_loaded = sum(
        1
        for stem in image_stems
        if stem in all_ground_truths and all_ground_truths[stem] is not None
    )  # Better count needed
    processed_stems = len(image_stems) - (
        len(image_stems) - len(all_ground_truths.keys())
    )  # Approximation
    logging.info(
        f"Ground truth loading complete. Processed {len(all_ground_truths)} stems."
        # f"Ground truth loading complete. Loaded annotations for {valid_gts_loaded} images."
    )
    return all_ground_truths


def _plot_pr_curve(pr_data: Dict, ap_per_class: Dict, class_names: List[str], output_path: Path):
    """Generates and saves the Precision-Recall curve plot."""
    logging.info("Generating Precision-Recall curve plot...")
    try:
        plt.figure(figsize=(10, 8))
        # Assuming class IDs map to indices of class_names
        class_id_to_name = {i: name for i, name in enumerate(class_names)}

        for class_id, data in pr_data.items():
            class_name = class_id_to_name.get(class_id, f"Class_{class_id}")
            ap = ap_per_class.get(class_name, 0.0)  # Get AP using class name key
            recall = data.get("recall")
            precision = data.get("precision")
            # Ensure recall and precision are numpy arrays for concatenation
            recall = np.array(recall) if recall is not None else np.array([])
            precision = np.array(precision) if precision is not None else np.array([])

            if recall.size > 0 and precision.size > 0:
                # Append the (0,1) point and ensure monotonically decreasing precision
                mrec = np.concatenate(([0.0], recall))
                mpre = np.concatenate(([1.0], precision))
                for i in range(mpre.size - 1, 0, -1):
                    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
                plt.plot(mrec, mpre, label=f"{class_name} (AP={ap:.3f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (IoU@0.5)")  # Assuming PR data is for 0.5
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Precision-Recall curve saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to generate PR curve plot: {e}", exc_info=True)


def _plot_confusion_matrix(
    cm_data: Optional[List[List[int]]],  # Made Optional
    labels: Optional[List[str]],  # Made Optional
    output_path: Path,
):
    """Generates and saves the Confusion Matrix heatmap."""
    logging.info("Generating Confusion Matrix plot...")
    if cm_data is None or not labels:
        logging.warning("Confusion matrix data or labels missing, skipping plot.")
        return
    try:
        cm_np = np.array(cm_data)
        plt.figure(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.5)))
        sns.heatmap(
            cm_np,
            annot=True,
            fmt="d",  # Integer format
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.title("Confusion Matrix")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Confusion Matrix plot saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to generate Confusion Matrix plot: {e}", exc_info=True)


def _format_config_summary(config: Dict) -> List[str]:
    """Formats the configuration part of the text summary."""
    lines = []
    lines.append("[Configuration]")
    lines.append(f"Model: {config.get('model', 'N/A')}")
    lines.append(f"Image Dir: {config.get('dataset', {}).get('image_dir', 'N/A')}")
    lines.append(f"Label Dir: {config.get('dataset', {}).get('label_dir', 'N/A')}")
    eval_p = config.get("evaluation_params", {})
    lines.append(f"Device: {eval_p.get('device', 'N/A')}")
    lines.append(f"Conf Thres (Pred): {eval_p.get('conf_thres', 'N/A')}")
    # Use the correct key for NMS IoU threshold
    lines.append(f"IoU Thres (NMS): {eval_p.get('iou_thres_nms', 'N/A')}")
    metric_p = config.get("metrics", {})
    lines.append(f"IoU Thres (mAP): {metric_p.get('map_iou_threshold', 0.5)}")
    lines.append(f"Conf Thres (CM): {metric_p.get('conf_threshold_cm', 0.45)}")
    lines.append(f"IoU Thres (CM): {metric_p.get('iou_threshold_cm', 0.5)}")
    return lines


def _format_compute_summary(compute_stats: Dict) -> List[str]:
    """Formats the compute stats part of the text summary."""
    lines = []
    lines.append("[Compute Stats]")
    if compute_stats:
        num_params_str = (
            f"{compute_stats.get('num_model_params', 'N/A'):,}"
            if compute_stats.get("num_model_params") is not None
            else "N/A"
        )
        avg_time_str = (
            f"{compute_stats.get('avg_inference_time_ms', 'N/A'):.2f}"
            if compute_stats.get("avg_inference_time_ms") is not None
            else "N/A"
        )
        peak_mem_str = (
            f"{compute_stats.get('peak_gpu_memory_mb', 'N/A'):.2f}"
            if compute_stats.get("peak_gpu_memory_mb") is not None
            else "N/A"
        )
        lines.append(f"Model Parameters: {num_params_str}")
        lines.append(f"Images Processed: {compute_stats.get('num_images_processed', 'N/A')}")
        lines.append(f"Avg. Inference Time (ms/img): {avg_time_str}")
        lines.append(f"Peak GPU Memory (MB): {peak_mem_str}")
    else:
        lines.append("N/A")
    return lines


def _format_metrics_summary(detection_metrics: Dict) -> List[str]:
    """Formats the detection metrics part of the text summary."""
    lines = []
    lines.append("[Detection Metrics]")
    if detection_metrics:
        map50_str = (
            f"{detection_metrics.get('mAP_50', 'N/A'):.4f}"
            if detection_metrics.get("mAP_50") is not None
            else "N/A"
        )
        map5095_str = (
            f"{detection_metrics.get('mAP_50_95', 'N/A'):.4f}"
            if detection_metrics.get("mAP_50_95") is not None
            else "N/A"
        )
        lines.append(f"Total Ground Truths: {detection_metrics.get('total_ground_truths', 'N/A')}")
        lines.append(f"mAP@0.50        : {map50_str}")
        lines.append(f"mAP@0.50:0.95   : {map5095_str}")
        lines.append("\nAP@0.50 per Class:")
        ap_50 = detection_metrics.get("ap_per_class_50", {})
        if ap_50:
            # Filter out potential None values before calculating max_len
            valid_keys = [name for name in ap_50.keys() if name is not None]
            max_len = max(len(name) for name in valid_keys) if valid_keys else 0
            for name, ap in ap_50.items():
                ap_str = f"{ap:.4f}" if ap is not None else "N/A"
                lines.append(f"  - {str(name):<{max_len}} : {ap_str}")  # Ensure name is string
        else:
            lines.append("  N/A")
    else:
        lines.append("N/A (Calculation might have failed or been skipped)")
    return lines


def _generate_text_summary(config: Dict, compute_stats: Dict, detection_metrics: Dict) -> str:
    """Generates a formatted text summary of the evaluation results."""
    summary = []
    summary.append("=" * 60)
    summary.append(" YOLOv11 Evaluation Summary")
    summary.append("=" * 60)
    summary.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("-" * 60)

    summary.extend(_format_config_summary(config))
    summary.append("-" * 60)

    summary.extend(_format_compute_summary(compute_stats))
    summary.append("-" * 60)

    summary.extend(_format_metrics_summary(detection_metrics))
    summary.append("-" * 60)

    summary.append(f"Results saved in: {str(Path('.'))}")  # Placeholder, will be replaced by caller
    summary.append("=" * 60)

    return "\n".join(summary)


def _save_results_json(
    output_dir: Path, config: Dict, compute_stats: Dict, detection_metrics: Dict
):
    """Saves the aggregated results to a JSON file."""
    json_path = output_dir / "evaluation_results.json"
    logging.info(f"Saving aggregated results to {json_path}...")
    try:
        # Create a deep copy to avoid modifying the original metrics dict
        # Use a robust default function for json.dumps
        def json_default(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(
                obj,
                (
                    np.int_,
                    np.intc,
                    np.intp,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                ),
            ):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
                return {"real": obj.real, "imag": obj.imag}
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, (np.void)):
                return None  # Or other representation?
            # Add handling for other non-serializable types if needed
            return str(obj)  # Fallback to string representation

        all_results_serializable = {
            "config_used": config,
            "compute_stats": compute_stats,
            "detection_metrics": json.loads(json.dumps(detection_metrics, default=json_default)),
        }

        with open(json_path, "w") as f:
            json.dump(all_results_serializable, f, indent=4)
        logging.info("Aggregated evaluation results saved successfully.")

    except Exception as e:
        logging.error(f"Failed to save evaluation results to JSON: {e}", exc_info=True)


def save_evaluation_results(
    output_dir: Path,
    config: Dict,
    compute_stats: Dict,
    detection_metrics: Dict,
    predictions: Optional[Dict] = None,
    ground_truths: Optional[Dict] = None,
):
    """Saves all evaluation results, metrics, plots, and summary."""
    logging.info(f"--- Saving Results to: {output_dir} ---")

    # 1. Save aggregated results to JSON
    _save_results_json(output_dir, config, compute_stats, detection_metrics)

    # 2. Generate and save PR curve plot
    pr_data = detection_metrics.get("pr_data_50") if detection_metrics else None
    ap_data = detection_metrics.get("ap_per_class_50") if detection_metrics else None
    if pr_data and ap_data:
        _plot_pr_curve(
            pr_data=pr_data,
            ap_per_class=ap_data,
            class_names=config.get("dataset", {}).get("class_names", []),
            output_path=output_dir / "pr_curve.png",
        )
    else:
        logging.warning("PR data or AP per class missing, skipping PR curve plot.")

    # 3. Generate and save confusion matrix plot
    cm_plot_data = detection_metrics.get("confusion_matrix") if detection_metrics else None
    cm_plot_labels = detection_metrics.get("confusion_matrix_labels") if detection_metrics else None
    if cm_plot_data is not None and cm_plot_labels is not None:
        _plot_confusion_matrix(
            cm_data=cm_plot_data,
            labels=cm_plot_labels,
            output_path=output_dir / "confusion_matrix.png",
        )
    else:
        logging.warning("Confusion matrix data or labels missing, skipping CM plot.")

    # 4 & 5. Generate, save, and print text summary
    summary_text = _generate_text_summary(config, compute_stats, detection_metrics)
    summary_text = summary_text.replace(
        f"Results saved in: {str(Path('.'))}", f"Results saved in: {str(output_dir.resolve())}"
    )
    summary_path = output_dir / "summary.txt"
    try:
        with open(summary_path, "w") as f:
            f.write(summary_text)
        logging.info(f"Text summary saved to {summary_path}")
    except Exception as e:
        logging.error(f"Failed to save text summary: {e}")

    logging.info("\n" + textwrap.indent(summary_text, "  "))

    # 6. Optionally save raw predictions/GTs (Placeholder)
    # ...
