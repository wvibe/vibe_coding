# src/models/ext/yolov8/benchmark/reporting.py

"""Functions for generating benchmark reports (HTML, plots)."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jinja2 import Template
import numpy as np

# Basic logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Plotting Functions --- #


def save_comparison_barplot(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    filename: Path,
    ylabel: str | None = None,
) -> None:
    """Generates and saves a bar plot comparing a metric across models."""
    if df.empty or metric_col not in df.columns or pd.isna(df[metric_col]).all():
        logging.warning(
            f"Skipping barplot for {metric_col}: DataFrame empty, column missing, or all values NaN."
        )
        return

    plt.figure(figsize=(10, 6))
    try:
        # Use model_name on x-axis
        sns.barplot(data=df, x="model_name", y=metric_col)
        plt.title(title)
        plt.xlabel("Model")
        plt.ylabel(ylabel if ylabel else metric_col)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        logging.info(f"Saved bar plot: {filename}")
    except Exception as e:
        logging.error(f"Failed to generate bar plot {filename}: {e}", exc_info=True)
        plt.close()


def save_inference_boxplot(
    # Placeholder - Needs modification to work with per-image times if collected
    # df_aggregated: pd.DataFrame,
    filename: Path,
) -> None:
    """Generates and saves a box plot of inference time distributions."""
    # This requires per-image inference times, which are not currently
    # stored in the aggregated DataFrame. This needs more complex data handling.
    logging.warning(
        "Inference time boxplot generation requires per-image timings and is not yet implemented."
    )
    # Placeholder logic:
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(data=df_individual_times, x="model_name", y="inference_time_ms")
    # plt.title("Inference Time Distribution")
    # plt.xlabel("Model")
    # plt.ylabel("Inference Time (ms)")
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.savefig(filename)
    # plt.close()
    pass  # Skip for now


def save_map_iou_plot(
    results_df: pd.DataFrame,
    iou_thresholds: Optional[np.ndarray],
    mean_ap_per_iou_dict: Dict[str, Optional[np.ndarray]],
    output_dir: Path,
) -> Optional[str]:
    """Generates and saves a line plot comparing mAP vs IoU threshold across models."""
    if iou_thresholds is None or not mean_ap_per_iou_dict:
        logging.warning("Skipping mAP vs IoU plot: Missing IoU thresholds or AP data.")
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    filename = plots_dir / "map_vs_iou.png"

    plt.figure(figsize=(12, 7))
    try:
        all_aps_valid = False
        for model_name in results_df["model_name"]:
            aps = mean_ap_per_iou_dict.get(model_name)
            if aps is not None and aps.shape == iou_thresholds.shape:
                plt.plot(iou_thresholds, aps, marker='o', linestyle='-', label=model_name)
                all_aps_valid = True # Mark if at least one model has valid data
            else:
                logging.warning(f"Skipping model {model_name} in mAP vs IoU plot: Missing or mismatched AP data.")

        if not all_aps_valid:
            logging.warning("Skipping mAP vs IoU plot: No valid AP data found for any model.")
            plt.close()
            return None

        plt.title("Mean Average Precision (mAP) vs. IoU Threshold")
        plt.xlabel("IoU Threshold")
        plt.ylabel("mAP")
        plt.grid(True)
        plt.legend(loc="best")
        plt.ylim(0, 1) # mAP is between 0 and 1
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        logging.info(f"Saved mAP vs IoU plot: {filename}")
        # Return relative path for HTML report
        return filename.relative_to(output_dir).as_posix()

    except Exception as e:
        logging.error(f"Failed to generate mAP vs IoU plot {filename}: {e}", exc_info=True)
        plt.close()
        return None


def save_confusion_matrix_plot(
    model_name: str,
    cm_data: Optional[np.ndarray],
    class_names: List[str],
    output_dir: Path,
) -> Optional[str]:
    """Generates and saves a confusion matrix heatmap for a specific model."""
    if cm_data is None or not class_names:
        logging.warning(f"Skipping confusion matrix for {model_name}: Missing matrix data or class names.")
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    filename = plots_dir / f"confusion_matrix_{model_name}.png"

    # Ensure class names match confusion matrix dimensions
    if cm_data.shape[0] != len(class_names) or cm_data.shape[1] != len(class_names):
        logging.warning(
            f"Skipping confusion matrix for {model_name}: Matrix dimensions ({cm_data.shape}) "
            f"do not match number of class names ({len(class_names)})."
        )
        return None

    plt.figure(figsize=(max(8, len(class_names) * 0.5), max(6, len(class_names) * 0.4)))
    try:
        sns.heatmap(
            cm_data,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout() # Adjust layout
        plt.savefig(filename)
        plt.close()
        logging.info(f"Saved confusion matrix: {filename}")
        # Return relative path for HTML report
        return filename.relative_to(output_dir).as_posix()
    except Exception as e:
        logging.error(f"Failed to generate confusion matrix plot {filename}: {e}", exc_info=True)
        plt.close()
        return None


def save_qualitative_results(
    # Placeholder - Needs significant implementation
    # image_paths: List[Path],
    # predictions: List,
    # ground_truths: List,
    # output_dir: Path,
    # num_images: int,
) -> List[str]:
    """Draws bounding boxes and saves qualitative result images."""
    logging.warning("Qualitative result image generation is not yet implemented.")
    return []  # Return empty list of image paths for now


# --- HTML Report Template --- #

# More robust to load from file, but embedding for simplicity initially
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Report</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .plot-container { margin-bottom: 30px; text-align: center; }
        img { max-width: 80%; height: auto; border: 1px solid #ccc; margin-top: 10px; }
        pre { background-color: #f5f5f5; padding: 10px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h1>Object Detection Benchmark Report</h1>
    <p>Generated on: {{ timestamp }}</p>

    <h2>Configuration Summary</h2>
    <pre><code>{{ config_yaml }}</code></pre>

    <h2>Aggregated Results</h2>
    {{ results_table_html | safe }}

    <h2>Visualizations</h2>

    {% if plot_paths.get('map50_comparison') %}
    <div class="plot-container">
        <h2>mAP@0.5 Comparison</h2>
        <img src="{{ plot_paths.map50_comparison }}" alt="mAP@0.5 Comparison Plot">
    </div>
    {% endif %}

    {% if plot_paths.get('map5095_comparison') %}
    <div class="plot-container">
        <h2>mAP@0.5:0.95 Comparison</h2>
        <img src="{{ plot_paths.map5095_comparison }}" alt="mAP@0.5:0.95 Comparison Plot">
    </div>
    {% endif %}

    {% if plot_paths.get('time_comparison') %}
    <div class="plot-container">
        <h2>Mean Inference Time Comparison</h2>
        <img src="{{ plot_paths.time_comparison }}" alt="Mean Inference Time Comparison Plot">
    </div>
    {% endif %}

    {% if plot_paths.get('gpu_comparison') %}
    <div class="plot-container">
        <h2>Peak GPU Memory Comparison</h2>
        <img src="{{ plot_paths.gpu_comparison }}" alt="Peak GPU Memory Comparison Plot">
    </div>
    {% endif %}

    {% if plot_paths.get('map_iou_plot') %}
    <div class="plot-container">
        <h2>mAP vs. IoU Threshold</h2>
        <img src="{{ plot_paths.map_iou_plot }}" alt="mAP vs. IoU Threshold Plot">
    </div>
    {% endif %}

    {% if plot_paths.get('time_distribution') %}
    {# Boxplot implementation is still pending #}
    {# <div class="plot-container">
        <h2>Inference Time Distribution</h2>
        <img src="{{ plot_paths.time_distribution }}" alt="Inference Time Distribution Plot">
        <p><i>Note: Boxplot generation pending implementation.</i></p>
    </div> #}
    {% endif %}

    {# Add placeholders for confusion matrix plots if implemented #}
    {% if plot_paths.get('confusion_matrices') %}
    <h2>Confusion Matrices</h2>
    {% for model_name, cm_path in plot_paths.get('confusion_matrices', {}).items() %}
        <div class="plot-container">
            <h3>{{ model_name }}</h3>
            <img src="{{ cm_path }}" alt="Confusion Matrix for {{ model_name }}">
        </div>
    {% endfor %}
    {% endif %}

    {% if qualitative_image_paths %}
    <h2>Qualitative Results (Sample)</h2>
    <p><i>Note: Qualitative result generation pending implementation.</i></p>
    {% for img_path in qualitative_image_paths %}
    <div class="plot-container">
        <img src="{{ img_path }}" alt="Qualitative Result">
    </div>
    {% endfor %}
    {% endif %}

</body>
</html>
"""

# --- Main Reporting Function --- #


def generate_html_report(
    results_df: pd.DataFrame,
    config_data: Dict,  # Original config dict for display
    # New detailed metrics data
    iou_thresholds: Optional[np.ndarray],
    mean_ap_per_iou_dict: Dict[str, Optional[np.ndarray]],
    confusion_matrices_dict: Dict[str, Optional[np.ndarray]],
    class_names: List[str], # Needed for CM plot labels
    output_dir: Path,
    report_filename: str = "report.html",
) -> None:
    """Generates the main HTML report with tables and plots."""
    logging.info("Generating HTML report...")

    # Ensure plots subdirectory exists
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Ensure qualitative subdirectory exists (even if unused for now)
    qualitative_dir = output_dir / "qualitative"
    qualitative_dir.mkdir(exist_ok=True)

    # --- Generate and Save Plots --- #
    plot_paths = {}

    # Simple comparison plots (relative paths for HTML)
    map50_plot_path = plots_dir / "map50_comparison.png"
    save_comparison_barplot(results_df, "mAP_50", "mAP@0.5 Comparison", map50_plot_path)
    if map50_plot_path.exists():
        plot_paths["map50_comparison"] = map50_plot_path.relative_to(output_dir).as_posix()

    map5095_plot_path = plots_dir / "map5095_comparison.png"
    save_comparison_barplot(results_df, "mAP_50_95", "mAP@0.5:0.95 Comparison", map5095_plot_path)
    if map5095_plot_path.exists():
        plot_paths["map5095_comparison"] = map5095_plot_path.relative_to(output_dir).as_posix()

    time_plot_path = plots_dir / "time_comparison.png"
    save_comparison_barplot(
        results_df,
        "inference_time_ms_mean",
        "Mean Inference Time Comparison",
        time_plot_path,
        ylabel="Time (ms)",
    )
    if time_plot_path.exists():
        plot_paths["time_comparison"] = time_plot_path.relative_to(output_dir).as_posix()

    gpu_plot_path = plots_dir / "gpu_comparison.png"
    save_comparison_barplot(
        results_df,
        "peak_gpu_memory_mb",
        "Peak GPU Memory Comparison",
        gpu_plot_path,
        ylabel="Memory (MB)",
    )
    if gpu_plot_path.exists():
        plot_paths["gpu_comparison"] = gpu_plot_path.relative_to(output_dir).as_posix()

    # NEW: mAP vs IoU Plot
    map_iou_rel_path = save_map_iou_plot(
        results_df,
        iou_thresholds,
        mean_ap_per_iou_dict,
        output_dir,
    )
    if map_iou_rel_path:
        plot_paths["map_iou_plot"] = map_iou_rel_path

    # Placeholders for more complex plots
    time_dist_path = plots_dir / "time_distribution.png"
    save_inference_boxplot(time_dist_path)
    if time_dist_path.exists():  # If the function is ever implemented and saves a file
        plot_paths["time_distribution"] = time_dist_path.relative_to(output_dir).as_posix()

    # Confusion Matrices (Per Model)
    cm_plot_paths = {}
    for model_name in results_df["model_name"]:
        cm_data = confusion_matrices_dict.get(model_name)
        cm_rel_path = save_confusion_matrix_plot(
            model_name=model_name,
            cm_data=cm_data,
            class_names=class_names,
            output_dir=output_dir,
        )
        if cm_rel_path:
            cm_plot_paths[model_name] = cm_rel_path
    if cm_plot_paths: # Only add if we generated at least one CM plot
        plot_paths["confusion_matrices"] = cm_plot_paths

    # Qualitative Results (Still Pending)
    qualitative_image_paths_relative = []
    # qualitative_image_files = save_qualitative_results(..., qualitative_dir, ...)
    # for img_file in qualitative_image_files:
    #    qualitative_image_paths_relative.append(Path(img_file).relative_to(output_dir).as_posix())

    # --- Prepare Data for Template --- #
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    results_table_html = results_df.to_html(
        index=False, float_format="%.3f", na_rep="N/A", justify="left"
    )

    # Convert config dict back to YAML string for display
    try:
        import yaml

        config_yaml = yaml.dump(config_data, sort_keys=False, default_flow_style=False)
    except ImportError:
        config_yaml = str(config_data)  # Fallback
    except Exception:
        config_yaml = "Error dumping config to YAML"

    # --- Render HTML --- #
    try:
        # Using embedded template string
        template = Template(HTML_TEMPLATE)
        html_content = template.render(
            timestamp=timestamp,
            config_yaml=config_yaml,
            results_table_html=results_table_html,
            plot_paths=plot_paths,
            qualitative_image_paths=qualitative_image_paths_relative,
        )

        report_path = output_dir / report_filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info(f"HTML report saved to: {report_path}")

    except Exception as e:
        logging.error(f"Failed to generate HTML report: {e}", exc_info=True)
