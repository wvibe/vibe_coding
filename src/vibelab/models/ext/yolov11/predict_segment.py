"""
Runs YOLOv11 segmentation prediction based on a configuration YAML file
and command-line arguments specifying the dataset and run name.

Configuration primarily defines model and inference parameters.
Command-line arguments specify the dataset, split/tag, output name,
and limited runtime overrides (device, save, show).
Calculates and reports prediction time statistics.
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

# Assuming utils are in the same directory or accessible via PYTHONPATH
from .predict_utils import (
    _load_expand_validate_config,
    _merge_config_and_args,
    _prepare_output_directory,
    _run_yolo_prediction,
    _validate_final_config,
)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Argument Parsing --- #


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLOv11 segmentation prediction with dataset/tag selection."
    )

    # Required arguments (Config, Tag, Name remain required)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Path to the prediction configuration YAML file (defines model, project, "
            "inference params)."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="voc",  # Default dataset set to 'voc'
        help=(
            "Dataset identifier (e.g., 'voc'). Used to find base path via env vars. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Dataset tag/split (e.g., 'val2007', 'test'). Appended to dataset path.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for the specific experiment run (output directory name).",
    )

    # Optional limited overrides - Set default to None to distinguish from user input
    parser.add_argument(
        "--device",
        type=str,
        default=None,  # Changed from 'cpu'
        help=(
            "Override compute device (e.g., 'cpu', '0'). Defaults to config file or "
            "Ultralytics default if not specified here."
        ),
    )
    parser.add_argument(
        "--save",
        type=lambda x: str(x).lower() == "true",
        default=None,  # Changed from True
        help=(
            "Override save results (True/False). Defaults to config "
            "file setting if not specified here."
        ),
    )
    parser.add_argument(
        "--show",
        type=lambda x: str(x).lower() == "true",
        default=None,  # Changed from False
        help=(
            "Override display results in window (True/False). Defaults to config "
            "file setting if not specified here."
        ),
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=None,
        help=("Randomly select N images if source is a directory. Processes all if None or 0."),
    )

    return parser.parse_args()


# --- Source Path Construction & Processing --- #


def construct_source_path(dataset_id: str, tag: str) -> Path:
    """Constructs the image source directory path based on dataset id and tag."""
    env_var_map = {
        "voc": "VOC_SEGMENT",
        # Add other datasets here, e.g., "coco": "COCO_SEGMENT"
    }
    dataset_env_var = env_var_map.get(dataset_id.lower())

    if not dataset_env_var:
        logger.error(f"Error: Unknown dataset identifier '{dataset_id}'. Add mapping to env var.")
        sys.exit(1)

    base_path_str = os.getenv(dataset_env_var)
    if not base_path_str:
        logger.error(
            f"Error: Environment variable '{dataset_env_var}' for dataset "
            f"'{dataset_id}' is not set."
        )
        sys.exit(1)

    source_dir = Path(base_path_str) / "images" / tag
    logger.info(f"Constructed source path: {source_dir}")

    if not source_dir.is_dir():
        logger.error(f"Error: Constructed source path is not a valid directory: {source_dir}")
        sys.exit(1)

    # Check if directory is empty
    if not any(source_dir.iterdir()):
        logger.warning(f"Warning: Source directory is empty: {source_dir}")
        # Allow proceeding, YOLO might handle it or error later

    return source_dir


def process_source(source_dir: Path, sample_count: int | None) -> str | list[str]:
    """Processes the source directory, applying random sampling if requested."""
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
    try:
        all_images = [p for p in source_dir.glob("*") if p.suffix.lower() in image_extensions]
    except Exception as e:
        logger.error(f"Error listing images in {source_dir}: {e}")
        sys.exit(1)

    num_found = len(all_images)
    if num_found == 0:
        logger.warning(f"No image files found in source directory: {source_dir}")
        # Return the dir path for now, YOLO predict handles it.
        return str(source_dir)

    if sample_count and sample_count > 0:
        num_to_select = min(sample_count, num_found)
        if num_found < sample_count:
            logger.warning(
                f"Requested {sample_count} images, but only found {num_found}. "
                f"Using all {num_found} found images."
            )
        selected_images = random.sample(all_images, num_to_select)
        processed_source = [str(p) for p in selected_images]
        logger.info(f"Randomly selected {num_to_select} images for processing.")
        return processed_source

    # If sample_count was None or 0, process all
    logger.info(f"Processing all {num_found} found images in directory: {source_dir}")
    return str(source_dir)


# --- Statistics Calculation --- #


def _extract_and_average_times(results: list) -> tuple[dict[str, float] | None, int]:
    """
    Extracts and averages processing times (preprocess, inference, postprocess)
    from the 'speed' dictionary in YOLO results.

    Returns:
        A tuple containing:
        - A dictionary with average times {'preprocess', 'inference', 'postprocess', 'total'}
          if valid times are found, otherwise None.
        - An integer count of results with valid speed information.
    """
    total_preprocess = 0.0
    total_inference = 0.0
    total_postprocess = 0.0
    valid_speeds_count = 0

    for res in results:
        if hasattr(res, "speed") and isinstance(res.speed, dict) and res.speed:
            # Use .get with default 0.0, handles None automatically
            pre = res.speed.get("preprocess", 0.0)
            inf = res.speed.get("inference", 0.0)
            post = res.speed.get("postprocess", 0.0)

            # Consider a result valid only if inference time is reported (>0)
            if inf > 0:
                total_preprocess += pre
                total_inference += inf
                total_postprocess += post
                valid_speeds_count += 1
        # else: pass # Skip results without valid speed info

    if valid_speeds_count == 0:
        logger.warning("Could not extract any valid speed information from results.")
        return None, 0

    avg_times = {
        "preprocess": total_preprocess / valid_speeds_count,
        "inference": total_inference / valid_speeds_count,
        "postprocess": total_postprocess / valid_speeds_count,
        "total": (total_preprocess + total_inference + total_postprocess) / valid_speeds_count,
    }

    if valid_speeds_count < len(results):
        logger.warning(
            f"Timing info (preprocess/inference/postprocess) was missing or invalid "
            f"for {len(results) - valid_speeds_count} out of {len(results)} images. "
            f"Averages are based on {valid_speeds_count} images."
        )

    return avg_times, valid_speeds_count


def _calculate_and_log_stats(
    results: list | None, predict_duration: float, cli_args: argparse.Namespace
):
    """Calculates and logs prediction statistics including average component times."""
    logger.info("--- Prediction Statistics ---")
    if results is None or not isinstance(results, list):
        logger.error("Prediction failed or returned unexpected result type.")
        return

    num_images_processed = len(results)
    logger.info(f"Successfully processed {num_images_processed} images.")

    if num_images_processed == 0:
        logger.warning("Prediction ran but returned zero results.")
        return

    # --- Wall Clock Stats ---
    avg_time_per_image_wall = (predict_duration / num_images_processed) * 1000  # ms
    logger.info(f"Total YOLO Prediction Wall Time: {predict_duration:.3f} s")
    logger.info(f"Average Time per Image (Wall Clock): {avg_time_per_image_wall:.2f} ms")
    logger.info(f"Overall FPS (Wall Clock): {num_images_processed / predict_duration:.2f}")

    # --- Ultralytics Speed Stats (Averages) ---
    avg_times, valid_speeds = _extract_and_average_times(results)

    if avg_times and valid_speeds > 0:
        logger.info(f"--- Avg Times from Ultralytics 'speed' (over {valid_speeds} images) ---")
        logger.info(f"  Avg Preprocess : {avg_times['preprocess']:.2f} ms")
        logger.info(f"  Avg Inference  : {avg_times['inference']:.2f} ms")
        logger.info(f"  Avg Postprocess: {avg_times['postprocess']:.2f} ms")
        logger.info(f"  Avg Total      : {avg_times['total']:.2f} ms")
    else:
        logger.warning("Could not calculate average times from Ultralytics 'speed' results.")

    logger.info("--- End of Statistics ---")  # Added separator


# --- Main Pipeline --- #


def predict_pipeline(cli_args: argparse.Namespace):
    """Orchestrates the segmentation prediction pipeline."""
    start_time_total = time.time()

    # 1. Load base config, expand env vars (utils handles .env loading)
    config_path = Path(cli_args.config)
    base_config = _load_expand_validate_config(config_path)

    # 2. Validate base config (model, project)
    _validate_final_config(base_config, config_path)  # Utils func validates model/project

    # 3. Construct source path from args
    source_path_dir = construct_source_path(cli_args.dataset, cli_args.tag)

    # 4. Process source (apply sampling if needed)
    processed_source = process_source(source_path_dir, cli_args.sample_count)

    # Estimate number of images being processed
    num_images_expected = "unknown"
    if isinstance(processed_source, list):
        num_images_expected = len(processed_source)
    elif isinstance(processed_source, str):
        source_path_obj = Path(processed_source)
        if source_path_obj.is_dir():
            # Re-count images in the directory if processing all
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
            try:
                num_images_expected = len(
                    [p for p in source_path_obj.glob("*") if p.suffix.lower() in image_extensions]
                )
            except Exception:
                pass  # Keep num_images_expected as "unknown"
        elif source_path_obj.is_file():  # Handle case where source somehow became a file
            num_images_expected = 1
    logger.info(f"Expecting to process {num_images_expected} image(s).")

    # 5. Merge limited CLI args into config
    final_config = _merge_config_and_args(base_config, cli_args)

    # 6. Prepare output directory info using name from args
    project_dir, exp_name_ts = _prepare_output_directory(final_config["project"], cli_args.name)

    # 7. Run YOLO segmentation prediction
    start_time_predict = time.time()
    results = _run_yolo_prediction(
        config=final_config,
        source=processed_source,
        project_dir_str=project_dir,
        exp_name=exp_name_ts,
        task_type="segment",
    )
    end_time_predict = time.time()
    predict_duration = end_time_predict - start_time_predict

    # 8. Calculate and Report Statistics using the refactored function
    _calculate_and_log_stats(results, predict_duration, cli_args)

    end_time_total = time.time()
    total_script_duration = end_time_total - start_time_total
    logger.info(f"Total Script Execution Time: {total_script_duration:.3f} s")


def main():
    args = parse_args()
    predict_pipeline(args)


if __name__ == "__main__":
    main()
