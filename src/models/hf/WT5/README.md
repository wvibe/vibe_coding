# WT5 for Sentiment Analysis

This repository contains an implementation of WT5, a customizable T5 model variant, applied to sentiment analysis on the IMDB movie review dataset.

## Overview

WT5 (Weighted T5) is a customizable implementation of the T5 model architecture that works with the HuggingFace ecosystem. This implementation allows for experimenting with different model configurations and training approaches for the IMDB sentiment analysis task.

The model converts the sentiment classification task into a text-to-text format:
- Input: "Review: [movie review text]"
- Output: "positive" or "negative"

## Requirements

- Python 3.12+
- PyTorch 2.0+
- Transformers 4.18+
- Datasets
- scikit-learn
- tqdm

You can install the required packages with:

```bash
# Clone the repository and install
git clone https://github.com/wvibe/vibe_coding.git
cd vibe_coding
pip install -e .
```

## Directory Structure

```
src/models/hf/WT5/
├── configuration.py    # WT5 configuration class
├── modeling.py         # WT5 model implementation
├── trainer.py          # Training implementation
├── run_experiment.py   # Script to run experiments
└── README.md           # Documentation
```

## Usage

### Running an Experiment

The simplest way to run an experiment is to use the `run_experiment.py` script:

```bash
python -m src.models.hf.WT5.run_experiment --config small --batch_size 4 --epochs 3
```

This will train a small WT5 model on the IMDB dataset for 3 epochs.

### Available Configurations

The script includes three predefined configurations:

1. `tiny`: A very small model for debugging and testing (2 layers, 64 hidden dim)
2. `small`: A smaller model suitable for MacBook training (4 layers, 256 hidden dim)
3. `base`: A larger model similar to T5-base (12 layers, 768 hidden dim)

### Command Line Arguments

- `--config`: Model size to use (tiny, small, or base)
- `--output_dir`: Directory to save models and results
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--max_length`: Maximum sequence length
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients

## Advanced Usage

### Custom Training

You can also use the trainer directly in your own code:

```python
from src.models.hf.WT5.configuration import WT5Config
from src.models.hf.WT5.trainer import WT5Trainer

# Create custom configuration
config = WT5Config(
    vocab_size=32128,
    d_model=256,
    d_kv=32,
    d_ff=1024,
    num_layers=4,
    num_decoder_layers=4,
    num_heads=4,
)

# Create trainer
trainer = WT5Trainer(
    model_config=config,
    output_dir="./my_experiment"
)

# Train model
results = trainer.train(
    batch_size=4,
    max_length=512,
    epochs=3,
    learning_rate=3e-5,
    gradient_accumulation_steps=2,
)
```

### Loading a Trained Model

You can load a trained model using the `from_pretrained` method:

```python
from src.models.hf.WT5.trainer import WT5Trainer

# Load model from saved checkpoint
trainer = WT5Trainer.from_pretrained("./my_experiment/best_model")

# Evaluate on test data
test_loader = trainer._prepare_data(batch_size=8, max_length=512)[1]
metrics = trainer.evaluate(test_loader)
print(f"Test metrics: {metrics}")
```

## Performance Notes

- The smaller models can be trained effectively on a MacBook, but for better performance, consider using a machine with GPU support.
- Using gradient accumulation (e.g., `--gradient_accumulation_steps 2`) can help train with larger effective batch sizes when memory is limited.
- The default sequence length of 512 works well for most IMDB reviews, but you can adjust it based on your memory constraints.

## Extending

This implementation can be extended for other text classification tasks by modifying the dataset formatting in `IMDBDataset` class to fit the specific requirements of your task.