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

The project uses a modern src-layout for better packaging:

```
vibe_coding/
├── src/                # Source code with proper package structure
│   ├── data_loaders/   # Dataset utilities
│   ├── models/         # Model implementations
│   │   ├── hf/         # Huggingface-based models
│   │   └── py/         # Pure PyTorch models
│   └── utils/          # Shared utilities
├── tests/              # Test files mirroring src structure
├── docs/               # Documentation
├── setup.py            # Package installation
└── requirements.txt    # Dependencies
```

### Environment Setup

Create a `.env` file in the project root with necessary paths:

```
# Required paths
VIBE_ROOT=/path/to/vibe
VHUB_ROOT=${VIBE_ROOT}/vhub
DATA_ROOT=${VHUB_ROOT}/data
```

## 📝 Note

This is a playground repository - perfect for experimentation and learning. Feel free to break things, fix them, and learn in the process!

## 👥 Contributing

When contributing to this playground:
- Create a new branch for your experiments
- Document your findings and approaches
- Feel free to add new directories for distinct project ideas
- Share insights in the documentation

## YOLOv8 Fine-tuning

A script is provided to fine-tune YOLOv8 models on configured datasets.

### Configuration

1.  **Dataset Definition (`src/models/ext/yolov8/configs/voc_combined.yaml`):** Defines the paths to the training, validation, and test image directories and the class names for the combined PASCAL VOC 2007+2012 dataset.
2.  **Training Parameters (`src/models/ext/yolov8/configs/voc_finetune_config.yaml`):** Specifies the base YOLOv8 model (e.g., `yolov8l.pt`), the dataset YAML to use, and all training hyperparameters (epochs, batch size, image size, device, optimizer, learning rate, output directories, etc.).

### Running Training

1.  **Activate Environment:** Ensure the `vbl` conda environment is active:
    ```bash
    conda activate vbl
    ```
2.  **Login to Wandb (Optional but Recommended):** If you want to log runs to Weights & Biases, log in first:
    ```bash
    wandb login
    ```
3.  **Run the Script:** Execute the training script from the project root directory (`vibe_coding/`), providing the path to the training configuration file:
    ```bash
    python src/models/ext/yolov8/finetune_yolov8.py --config src/models/ext/yolov8/configs/voc_finetune_config.yaml
    ```
4.  **Output:** Training progress will be displayed in the terminal. Results, including trained model weights (`best.pt`, `last.pt`) and logs, will be saved to the directory specified by the `project` and `name` parameters in the training config (default: `runs/detect/yolov8l_voc_combined_finetune/`). If wandb is enabled and configured, metrics will also be logged there.

---
Happy Coding! ✨