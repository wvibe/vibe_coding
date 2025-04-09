# 🎮 Vibe Coding Playground

Welcome to the Vibe Coding playground! This repository serves as an experimental space for testing ideas, learning new concepts, and exploring various coding projects.

## 🎯 Purpose

This is a sandbox environment where we:
- Test new coding concepts
- Experiment with different technologies
- Practice coding techniques
- Store code snippets and examples
- Try out new ideas

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/wvibe/vibe_coding.git
cd vibe_coding

# Install as a development package
pip install -e .
```

### Project Structure

The project uses a modern src-layout and is organized as follows:

```
vibe_coding/                # Git Repository Root ($HOME//vibe/vibe_coding)
├── configs/                # Configuration files (datasets, training, models)
├── docs/                   # Project documentation (guides, designs)
├── ext/                    # External git libraries/submodules (READ-ONLY)
├── notebooks/              # Jupyter notebooks for experiments and analysis
├── ref/                    # External reference code snippets/examples (READ-ONLY)
├── scripts/                # Standalone utility scripts (not part of the src package)
├── src/                    # Source code package (installable via setup.py)
│   ├── models/             # Core model implementations & integrated external models (e.g., YOLO)
│   └── utils/              # Shared utility functions (e.g., data converters)
├── tests/                  # Unit and integration tests (mirroring src structure)
│   ├── models/             # Tests for src/models
│   └── utils/              # Tests for src/utils
├── README.md               # Project overview, setup, and usage guide
├── requirements.txt        # Python dependencies (managed by pip)
└── setup.py                # Enables `pip install -e .` for the src package
```

### Environment Setup

Create a `.env` file in the project root with necessary paths:

```
# Required paths
VIBE_ROOT=/path/to/vibe
VHUB_ROOT=${VIBE_ROOT}/vhub
DATA_ROOT=${VHUB_ROOT}/data
```

## 🖥️ Cursor IDE Setup

For an enhanced development experience using the Cursor IDE, please follow the setup guide located here:

➡️ **[Cursor Setup Guide](./docs/cursor/setup-guide.md)**

This guide covers:
- Configuring global and project-specific rules.
- Setting up recommended auto-run behavior.
- Using Model Configuration Profiles (MCPs).
- Effective development workflows with the AI assistant.

### Helpful Videos

*   **Intro to Cursor Features:** [https://www.youtube.com/watch?v=TQsP_PlCY1I](https://www.youtube.com/watch?v=TQsP_PlCY1I&list=PLhBPcrfhLEXuzJRfriP_C6KDf3M0oXmRg&index=4)
*   **Cursor AI Pair Programming:** [https://www.youtube.com/watch?v=v7UcVPO4y3c](https://www.youtube.com/watch?v=v7UcVPO4y3c&list=PLhBPcrfhLEXuzJRfriP_C6KDf3M0oXmRg&index=8)

## 📝 Note

This is a playground repository - perfect for experimentation and learning. Feel free to break things, fix them, and learn in the process!

## 👥 Contributing

When contributing to this playground:
- Create a new branch for your experiments
- Document your findings and approaches
- Feel free to add new directories for distinct project ideas
- Share insights in the documentation

## YOLOv8 Training

A script is provided to train or fine-tune YOLOv8 models on configured datasets.

### Configuration

1.  **Dataset Definition (e.g., `src/models/ext/yolov8/configs/voc_combined.yaml`):** Defines the paths to the training, validation, and test image directories and the class names for the dataset.
2.  **Training Parameters (e.g., `voc_finetune_config.yaml`, `voc_scratch_config.yaml`):** Specifies the base YOLOv8 model/architecture (`.pt` for fine-tuning, `.yaml` for scratch), the dataset YAML to use, and all training hyperparameters (epochs, batch size, image size, device, optimizer, learning rate, output directories, etc.). Ensure `pretrained` is `True` for fine-tuning and `False` for scratch training.

### Running Training

1.  **Activate Environment:** Ensure the `vbl` conda environment is active:
    ```bash
    conda activate vbl
    ```
2.  **Login to Wandb (Optional but Recommended):** If you want to log runs to Weights & Biases, log in first:
    ```bash
    wandb login
    ```
3.  **Run the Script:** Execute the training script from the project root directory (`vibe_coding/`), providing the path to the desired training configuration file and a unique run name:

    *   **Fine-tuning Example:**
        ```bash
        python src/models/ext/yolov8/train_yolov8.py \
            --config src/models/ext/yolov8/configs/voc_finetune_config.yaml \
            --name yolov8l_voc_finetune_run1
        ```
    *   **Training from Scratch Example:**
        ```bash
        python src/models/ext/yolov8/train_yolov8.py \
            --config src/models/ext/yolov8/configs/voc_scratch_config.yaml \
            --name yolov8l_voc_scratch_run1
        ```
    *   **Resuming a Run Example (using WandB ID `abc123xyz`):**
        ```bash
        # Ensure the --name matches the original run you want to resume
        python src/models/ext/yolov8/train_yolov8.py \
            --config src/models/ext/yolov8/configs/voc_finetune_config.yaml \
            --resume \
            --wandb-id abc123xyz \
            --name yolov8l_voc_finetune_run1
        ```
    *   **Overriding Project Directory Example:**
        ```bash
        python src/models/ext/yolov8/train_yolov8.py \
            --config src/models/ext/yolov8/configs/voc_scratch_config.yaml \
            --project runs/scratch_experiments \
            --name yolov8l_voc_scratch_run2
        ```

4.  **Output:** Training progress will be displayed in the terminal. Results, including trained model weights (`best.pt`, `last.pt`) and logs, will be saved to the directory specified by the `--project` and `--name` parameters. If wandb is enabled and configured, metrics will also be logged there.

## YOLOv11 Detection Training

A similar script is provided for training YOLOv11 detection models.

### Configuration

1.  **Dataset Definition:** Similar to YOLOv8, define dataset paths and classes in a dedicated YAML file (e.g., `configs/yolov11/voc_detect.yaml`). Ensure paths are correct relative to the project root or absolute.
2.  **Training Parameters:** Create a main training YAML (e.g., `configs/yolov11/voc11_finetune_config.yaml`). This file specifies:
    *   `model`: Path to the base model (`.pt`) for fine-tuning or architecture (`.yaml`) for scratch.
    *   `data`: Path (relative to project root) to the dataset definition YAML.
    *   Training hyperparameters (epochs, batch, imgsz, optimizer, lr0, device, etc.).
    *   `project`: (Optional) Base directory for saving runs (defaults to `runs/train/detect`).

### Running Training

1.  **Activate Environment:** Ensure the `vbl` conda environment is active:
    ```bash
    conda activate vbl
    ```
2.  **Login to Wandb (Optional but Recommended):** If using Weights & Biases:
    ```bash
    wandb login
    ```
3.  **Run the Script:** Execute from the project root (`vibe_coding/`).

    *   **Fine-tuning Example:**
        ```bash
        python src/models/ext/yolov11/train_detect.py \
            --config configs/yolov11/voc11_finetune_config.yaml \
            --name voc11_finetune_run1
        ```
    *   **Training from Scratch Example (assuming a `voc11_scratch_config.yaml` exists):**
        ```bash
        python src/models/ext/yolov11/train_detect.py \
            --config configs/yolov11/voc11_scratch_config.yaml \
            --name voc11_scratch_run1
        ```
    *   **Resuming a Run Example:**
        Provide the path to the specific run directory you want to resume.
        ```bash
        # Example assuming the run to resume is runs/train/detect/voc11_finetune_run1_20240801_100000
        python src/models/ext/yolov11/train_detect.py \
            --config configs/yolov11/voc11_finetune_config.yaml \
            --resume_with runs/train/detect/voc11_finetune_run1_20240801_100000 \
            --name voc11_finetune_run1 # Base name is still required but overridden by resume_with path
        ```
        **Note on Resuming WandB:** The script will automatically attempt to find the corresponding WandB run ID within the directory specified by `--wandb-dir` (defaulting to `./wandb`) based on the run name in the Ultralytics directory (`voc11_finetune_run1_20240801_100000` in this example). If found, it resumes the WandB log; otherwise, it starts a new WandB run.

    *   **Overriding Project Directory Example:**
        ```bash
        python src/models/ext/yolov11/train_detect.py \
            --config configs/yolov11/voc11_scratch_config.yaml \
            --project runs/yolov11_experiments \
            --name voc11_scratch_run2
        ```

4.  **Output:** Results are saved similarly to YOLOv8, within the directory defined by `--project` and the specific run name (e.g., `voc11_finetune_run1_YYYYMMDD_HHMMSS`).

## Data Conversion (VOC to YOLO)

Scripts are provided to convert Pascal VOC datasets (detection and segmentation annotations) into the YOLO format required by many training libraries.

### Detection Labels

To convert VOC XML annotations (bounding boxes) for a specific year and ImageSet tag (e.g., `trainval` from `ImageSets/Main/`) into YOLO detection format (`.txt` files containing `class_id x_center y_center width height`), use the following script:

```bash
python src/utils/data_converter/voc2yolo_detect_labels.py \
    --devkit-path /path/to/your/VOCdevkit \
    --year 2012 \
    --tag trainval \
    --output-dir /path/to/output/labels_detect
```

- `--devkit-path`: Path to the root `VOCdevkit` folder.
- `--year`: Dataset year (e.g., `2007`, `2012`).
- `--tag`: The ImageSet tag to process (e.g., `train`, `val`, `test`, `trainval`). The script looks for `<tag>.txt` inside `VOCdevkit/VOC<year>/ImageSets/Main/`.
- `--output-dir`: The directory where the resulting YOLO label files (`<image_id>.txt`) will be saved directly (flat structure).

### Segmentation Labels

To convert VOC segmentation annotations (instance masks from `.png` files in `SegmentationObject/`, matched with XML object info) for a specific year and ImageSet tag (e.g., `trainval` from `ImageSets/Segmentation/`) into YOLO segmentation format (`.txt` files containing `class_id norm_x1 norm_y1 norm_x2 norm_y2 ...`), use the following script:

```bash
python src/utils/data_converter/voc2yolo_segment_labels.py \
    --devkit-path /path/to/your/VOCdevkit \
    --year 2012 \
    --tag trainval \
    --output-dir /path/to/output/labels_segment \
    --iou-threshold 0.5
```

- `--devkit-path`: Path to the root `VOCdevkit` folder.
- `--year`: Dataset year (e.g., `2007`, `2012`).
- `--tag`: The ImageSet tag to process (e.g., `train`, `val`, `trainval`). The script looks for `<tag>.txt` inside `VOCdevkit/VOC<year>/ImageSets/Segmentation/`.
- `--output-dir`: The directory where the resulting YOLO label files (`<image_id>.txt`) will be saved directly (flat structure).
- `--iou-threshold` (Optional): The IoU threshold used to match mask instances to bounding boxes in the XML file (default: 0.5).

**Note on Segmentation Data:** Ensure that the `SegmentationObject` directory exists and contains the necessary `.png` masks for the specified year and tag. Not all images listed in `ImageSets/Segmentation/` might have corresponding masks, and the script will log skips for missing masks.

**Note on Instance vs. Class IDs:** The pixel values in the `SegmentationObject` PNG masks represent *instance IDs* (unique for each object instance in an image, regardless of class), not *class IDs*. The conversion script uses Intersection over Union (IoU) between the mask's bounding box and the bounding boxes defined in the corresponding XML file to determine the correct *class ID* for each instance mask/polygon.

---
Happy Coding! ✨