"""Main script to run the object detection benchmark."""

import argparse
import logging
import sys
import time  # For timestamp in output dir
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch  # <-- ADDED Import
import yaml
from pydantic import ValidationError

# Ultralytics
from ultralytics import YOLO
from ultralytics.engine.results import Results  # For type hinting

# Assuming the script is run from the project root (vibe_coding)
try:
    from src.models.ext.yolov8.benchmark.config import BenchmarkConfig
    from src.models.ext.yolov8.benchmark.metrics import (
        GroundTruthBox,
        calculate_detection_metrics,
        parse_yolo_labels,
    )
    from src.models.ext.yolov8.benchmark.reporting import generate_html_report
    from src.models.ext.yolov8.benchmark.utils import find_dataset_files, select_subset
except ImportError:
    # Allow running script directly for testing, adjust path if needed
    from config import BenchmarkConfig
    from metrics import GroundTruthBox, calculate_detection_metrics, parse_yolo_labels
    from reporting import generate_html_report

    from utils import find_dataset_files, select_subset


# Basic logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def benchmark_single_model(
    model: YOLO,
    model_name: str,
    benchmark_files: List[Tuple[Path, Optional[Path]]],
    config: BenchmarkConfig,
    output_dir: Path,
    num_classes: int,
) -> Dict[str, Any]:
    """
    Runs the benchmark for a single loaded model.
    Calculates inference time statistics and calls metric calculation.
    """
    logging.info(f"--- Benchmarking Model: {model_name} ---")
    start_time_total = time.time()

    inference_times_ms = []
    all_predictions: List[Results] = []  # Store Ultralytics Results objects
    all_ground_truths: List[List[GroundTruthBox]] = []  # Store parsed GT boxes
    num_images = len(benchmark_files)

    # Determine device
    device = config.compute.device
    if device == "auto":
        effective_device = model.device  # Use device determined by YOLO
        logging.info(f"Using auto-detected device: {effective_device}")
    else:
        effective_device = device
        logging.info(f"Using specified device: {effective_device}")

    # Check if CUDA is available and the effective device is a CUDA device
    is_cuda = torch.cuda.is_available() and str(effective_device).startswith("cuda")
    if is_cuda:
        logging.info(f"Resetting peak memory stats for device: {effective_device}")
        try:
            torch.cuda.reset_peak_memory_stats(effective_device)
        except Exception as e:
            logging.warning(f"Could not reset CUDA memory stats for {effective_device}: {e}")
            is_cuda = False  # Proceed without memory measurement if reset fails

    peak_mem_mb = -1.0  # Default value if not CUDA or error

    logging.info(f"Processing {num_images} images...")

    # --- Inference Loop ---
    for i, (img_path, label_path) in enumerate(benchmark_files):
        if i % 50 == 0 and i > 0:  # Log progress every 50 images
            logging.info(f"Processing image {i}/{num_images}...")
        try:
            # Run inference and time it
            inf_start_time = time.perf_counter()
            results = model.predict(source=str(img_path), device=effective_device, verbose=False)
            inf_end_time = time.perf_counter()
            inference_times_ms.append((inf_end_time - inf_start_time) * 1000)

            # Store predictions (typically results is a list of one Results object)
            if results:
                all_predictions.append(results[0])
            else:
                # Handle cases where predict might return empty list (shouldn't happen often)
                all_predictions.append(None)  # Or handle appropriately

            # Parse and store ground truth
            if config.dataset.annotation_format == "yolo_txt":
                gt_boxes = parse_yolo_labels(label_path)
                all_ground_truths.append(gt_boxes)
            # TODO: Add elif for voc_xml parsing
            else:
                # If format is not supported for parsing here, add empty list
                all_ground_truths.append([])
                if config.dataset.annotation_format != "yolo_txt":  # Avoid repeating warning
                    logging.warning(
                        f"Ground truth parsing not implemented for format '{config.dataset.annotation_format}'. Metrics requiring GT will be 0.",
                        once=True,
                    )

        except Exception as e:
            logging.error(f"Error processing image {img_path.name}: {e}", exc_info=True)
            # Ensure lists stay aligned if an error occurs
            all_predictions.append(None)
            all_ground_truths.append(None)  # Use None to indicate error for this image

    # Filter out errored entries before metric calculation
    valid_indices = [
        i
        for i, pred in enumerate(all_predictions)
        if pred is not None and all_ground_truths[i] is not None
    ]
    if len(valid_indices) != num_images:
        logging.warning(
            f"Excluded {num_images - len(valid_indices)} images due to errors during processing."
        )

    valid_predictions = [all_predictions[i] for i in valid_indices]
    valid_ground_truths = [all_ground_truths[i] for i in valid_indices]
    num_valid_images = len(valid_predictions)

    # --- Calculate Inference Time Stats ---
    if not inference_times_ms:
        logging.warning(
            f"No successful inferences recorded for {model_name}. Cannot calculate times."
        )
        mean_time_ms, p75_time_ms, p90_time_ms, p95_time_ms = -1.0, -1.0, -1.0, -1.0
    else:
        mean_time_ms = pd.Series(inference_times_ms).mean()
        p75_time_ms, p90_time_ms, p95_time_ms = pd.Series(inference_times_ms).quantile(
            [0.75, 0.90, 0.95]
        )
        logging.info(f"Inference Time (ms) - Mean: {mean_time_ms:.2f}, 95th: {p95_time_ms:.2f}")

    # --- Calculate Detection Metrics ---
    if num_valid_images > 0:
        detection_metrics = calculate_detection_metrics(
            predictions=valid_predictions,
            ground_truths=valid_ground_truths,
            num_classes=num_classes,
            config=config,
        )
    else:
        logging.warning(
            f"No valid predictions/ground truths to calculate detection metrics for {model_name}."
        )
        # Return default zero values for metrics
        detection_metrics = {
            "mAP_50": 0.0,
            "mAP_50_95": 0.0,
            "mAP_small": 0.0,
            "mAP_medium": 0.0,
            "mAP_large": 0.0,
        }

    # --- Calculate Peak GPU Memory ---
    if is_cuda:
        try:
            peak_mem_bytes = torch.cuda.max_memory_allocated(effective_device)
            peak_mem_mb = round(peak_mem_bytes / (1024 * 1024), 2)  # Convert bytes to MB
            logging.info(f"Peak GPU Memory Allocated: {peak_mem_mb:.2f} MB")
        except Exception as e:
            logging.warning(f"Could not get max CUDA memory allocated for {effective_device}: {e}")

    end_time_total = time.time()
    total_time_sec = end_time_total - start_time_total
    logging.info(f"--- Finished benchmarking {model_name} in {total_time_sec:.2f} seconds ---")

    # --- Return Combined Metrics ---
    metrics = {
        "model_name": model_name,
        "num_images_processed": num_valid_images,
        "num_images_requested": num_images,
        "total_time_sec": round(total_time_sec, 2),
        "inference_time_ms_mean": round(mean_time_ms, 2),
        "inference_time_ms_p75": round(p75_time_ms, 2),
        "inference_time_ms_p90": round(p90_time_ms, 2),
        "inference_time_ms_p95": round(p95_time_ms, 2),
        "peak_gpu_memory_mb": peak_mem_mb,  # Update placeholder
    }
    metrics.update(detection_metrics)  # Add mAP values etc.

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run object detection benchmark.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the benchmark configuration YAML file."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        logging.error(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    logging.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Validate configuration using Pydantic
        config = BenchmarkConfig(**config_data)
        logging.info("Configuration loaded and validated successfully.")
        # logging.debug(config.model_dump_json(indent=2)) # Optional: Print validated config

        # --- Setup Output Directory ---
        output_dir_template = config.output.output_dir
        final_output_dir_str = output_dir_template  # Default to template string

        placeholder_pattern = "{timestamp:"
        if placeholder_pattern in output_dir_template:
            try:
                # Extract format like '%Y%m%d_%H%M%S'
                strftime_format = output_dir_template.split(placeholder_pattern)[1].split("}")[0]
                # Rebuild the exact placeholder string like '{timestamp:%Y%m%d_%H%M%S}'
                placeholder_to_replace = f"{{timestamp:{strftime_format}}}"
                # Generate the actual timestamp string like '20250330_043256'
                timestamp_str = time.strftime(strftime_format)
                # Replace the placeholder in the original template
                final_output_dir_str = output_dir_template.replace(
                    placeholder_to_replace, timestamp_str
                )
            except IndexError:
                # Handle cases like "{timestamp:}" or missing '}'
                logging.warning(
                    f"Malformed timestamp placeholder in output_dir: '{output_dir_template}'. Using directory name as is."
                )
                # final_output_dir_str remains output_dir_template
            except Exception as e:
                # Handle other potential errors during strftime or split
                logging.warning(
                    f"Error processing timestamp placeholder in output_dir: '{output_dir_template}'. Error: {e}. Using directory name as is."
                )
                # final_output_dir_str remains output_dir_template

        # If the pattern was never found, final_output_dir_str remains output_dir_template
        final_output_dir = Path(final_output_dir_str)

        logging.info(f"Creating output directory: {final_output_dir}")
        final_output_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Setup file logging handler to log to a file in final_output_dir

        # --- Load Dataset ---
        logging.info("Loading dataset index...")
        all_files = find_dataset_files(config.dataset)
        if not all_files:
            logging.error("No dataset files found. Exiting.")
            sys.exit(1)

        benchmark_files = select_subset(all_files, config.dataset)
        if not benchmark_files:
            logging.error("No files selected for benchmark after subset selection. Exiting.")
            sys.exit(1)

        logging.info(f"Selected {len(benchmark_files)} files for benchmarking.")

        # --- Run Benchmarks for each model ---
        all_results = []
        num_classes = config.dataset.num_classes

        for model_name_or_path in config.models_to_test:
            try:
                logging.info(f"Loading model: {model_name_or_path}...")
                # YOLO class handles both names (e.g., 'yolov8n.pt') and paths
                model = YOLO(model_name_or_path)
                logging.info("Model loaded successfully.")

                # Run benchmark for this specific model
                model_results = benchmark_single_model(
                    model=model,
                    model_name=model_name_or_path,
                    benchmark_files=benchmark_files,
                    config=config,
                    output_dir=final_output_dir,  # Pass output dir for model-specific files
                    num_classes=num_classes,
                )
                all_results.append(model_results)

            except Exception as e:
                logging.error(f"Failed to benchmark model {model_name_or_path}: {e}", exc_info=True)
                # Optionally add a placeholder result indicating failure
                all_results.append(
                    {
                        "model_name": model_name_or_path,
                        "error": str(e),
                        "num_images_processed": 0,
                        "num_images_requested": len(benchmark_files),
                        "total_time_sec": 0.0,
                        "inference_time_ms_mean": -1.0,
                        "inference_time_ms_p75": -1.0,
                        "inference_time_ms_p90": -1.0,
                        "inference_time_ms_p95": -1.0,
                        "peak_gpu_memory_mb": -1.0,
                        "mAP_50": -1.0,
                        "mAP_50_95": -1.0,
                        "mAP_small": -1.0,
                        "mAP_medium": -1.0,
                        "mAP_large": -1.0,
                    }
                )

        # --- Aggregate and Save Results ---
        if not all_results:
            logging.warning("No models were successfully benchmarked.")
        else:
            logging.info("\n--- Aggregated Results ---")
            results_df = pd.DataFrame(all_results)
            print(results_df.to_string())  # Print to console for now

            # Save to CSV
            csv_path = final_output_dir / config.output.results_csv
            try:
                results_df.to_csv(csv_path, index=False)
                logging.info(f"Aggregated results saved to: {csv_path}")
            except Exception as e:
                logging.error(f"Failed to save results CSV to {csv_path}: {e}")

            # Generate HTML report
            try:
                generate_html_report(
                    results_df=results_df,
                    config_data=config_data,  # Pass the original loaded config dict
                    output_dir=final_output_dir,
                    report_filename=config.output.results_html,
                )
            except Exception as e:
                logging.error(f"Failed to generate HTML report: {e}", exc_info=True)

    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        sys.exit(1)
    except ValidationError as e:
        logging.error(f"Error validating configuration:\n{e}")
        sys.exit(1)
    except Exception as e:
        logging.exception(
            f"An unexpected error occurred: {e}"
        )  # Use logging.exception to include traceback
        sys.exit(1)


if __name__ == "__main__":
    main()
