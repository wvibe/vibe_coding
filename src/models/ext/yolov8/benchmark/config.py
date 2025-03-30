# src/models/ext/yolov8/benchmark/config.py

"""Pydantic models for validating benchmark configuration."""

from typing import List, Optional, Union

from pydantic import BaseModel, DirectoryPath, FilePath, validator

# Use basic types for now to avoid potential edit issues.
# We can add stricter validation (Literals, Field constraints, validators) later.


class DatasetConfig(BaseModel):
    test_images_dir: DirectoryPath
    annotations_dir: DirectoryPath
    annotation_format: str  # e.g., "voc_xml", "yolo_txt"
    num_classes: int  # Number of classes in the dataset
    subset_method: str  # e.g., "random", "first_n", "all"
    subset_size: int
    image_list_file: Optional[FilePath] = None


class ObjectSizeDefinition(BaseModel):
    # Using List temporarily as Tuple caused issues before
    small: List[Union[int, float]]  # e.g., [0, 1024]
    medium: List[Union[int, float]]  # e.g., [1024, 9216]
    large: List[Union[int, float]]  # e.g., [9216, math.inf] or [9216, ".inf"]


class MetricsConfig(BaseModel):
    iou_threshold_map: float
    # Using List temporarily as Tuple caused issues before
    iou_range_coco: List[float]  # e.g., [0.5, 0.95, 0.05]
    object_size_definitions: ObjectSizeDefinition
    confusion_matrix_classes: Optional[List[str]] = None


class ComputeConfig(BaseModel):
    device: str  # e.g., "cpu", "cuda:0", "auto"
    batch_size: int


class OutputConfig(BaseModel):
    output_dir: str
    results_csv: str
    results_html: str
    save_plots: bool
    save_qualitative_results: bool
    num_qualitative_images: int


class BenchmarkConfig(BaseModel):
    """Root model for the benchmark configuration."""

    models_to_test: List[str]
    dataset: DatasetConfig
    metrics: MetricsConfig
    compute: ComputeConfig
    output: OutputConfig

    # Pydantic doesn't automatically handle float('inf') from YAML's '.inf'
    # Adding a simple validator to handle this for the size definitions
    @validator("metrics")
    def convert_inf_in_sizes(cls, metrics_config: MetricsConfig) -> MetricsConfig:
        for size_key in ["small", "medium", "large"]:
            size_range = getattr(metrics_config.object_size_definitions, size_key)
            if (
                isinstance(size_range, list)
                and len(size_range) == 2
                and isinstance(size_range[1], str)
                and size_range[1].lower() == ".inf"
            ):
                size_range[1] = float("inf")
                # Update the attribute back
                setattr(metrics_config.object_size_definitions, size_key, size_range)
        return metrics_config

    # Re-add basic cross-field validation after individual fields are parsed
    @validator("dataset", check_fields=False)  # check_fields=False for root validators
    def check_dataset_logic(cls, dataset_config: DatasetConfig) -> DatasetConfig:
        if dataset_config.subset_method != "all" and dataset_config.subset_size <= 0:
            raise ValueError("subset_size must be positive unless subset_method is 'all'")
        # Relaxing this constraint for now, maybe user wants to select randomly *from* a list?
        # if dataset_config.image_list_file and dataset_config.subset_method != 'all':
        #     raise ValueError("image_list_file is currently only supported when subset_method is 'all'")
        return dataset_config

    @validator("output", check_fields=False)
    def check_output_logic(cls, output_config: OutputConfig) -> OutputConfig:
        if not output_config.save_qualitative_results and output_config.num_qualitative_images > 0:
            # Optionally warn or reset num_qualitative_images if save_qualitative_results is false
            # For now, just allow it.
            pass
        if output_config.num_qualitative_images < 0:
            raise ValueError("num_qualitative_images cannot be negative")
        return output_config
