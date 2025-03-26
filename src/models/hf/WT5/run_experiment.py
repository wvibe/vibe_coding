#!/usr/bin/env python3
"""
Run an experiment with the WT5 model on the IMDB sentiment analysis dataset.
This script provides a convenient way to train and evaluate the model with different configurations.
"""

import argparse
import os
from datetime import datetime

from src.models.hf.WT5.configuration import WT5Config
from src.models.hf.WT5.trainer import WT5Trainer


def create_smaller_config():
    """Create a smaller WT5 model configuration suitable for MacBook training."""
    return WT5Config(
        vocab_size=32128,
        d_model=256,
        d_kv=32,
        d_ff=1024,
        num_layers=4,
        num_decoder_layers=4,
        num_heads=4,
    )


def create_tiny_config():
    """Create a tiny WT5 model configuration for debugging and quick experiments."""
    return WT5Config(
        vocab_size=32128,
        d_model=64,
        d_kv=16,
        d_ff=256,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=2,
    )


def create_base_config():
    """Create a base WT5 model configuration (similar to T5-base)."""
    return WT5Config(
        vocab_size=32128,
        d_model=768,
        d_kv=64,
        d_ff=3072,
        num_layers=12,
        num_decoder_layers=12,
        num_heads=12,
    )


def run_experiment(
    config_name,
    output_dir,
    batch_size,
    epochs,
    learning_rate,
    max_length,
    gradient_accumulation_steps,
    use_wandb=False,
    wandb_project="wt5-sentiment",
    eval_steps=500,
    logging_steps=50,
    save_steps=1000,
):
    """
    Run a training experiment with the specified configuration.

    Args:
        config_name: Name of the configuration to use (tiny, small, or base)
        output_dir: Directory to save the model and results
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        max_length: Maximum sequence length
        gradient_accumulation_steps: Number of steps to accumulate gradients
        use_wandb: Whether to use Weights & Biases for logging
        wandb_project: Name of the wandb project
        eval_steps: Number of steps between evaluations
        logging_steps: Number of steps between logging
        save_steps: Number of steps between saving checkpoints
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Select the appropriate configuration
    if config_name == "tiny":
        config = create_tiny_config()
    elif config_name == "small":
        config = create_smaller_config()
    elif config_name == "base":
        config = create_base_config()
    else:
        raise ValueError(f"Unknown configuration: {config_name}")

    # Create the trainer
    trainer = WT5Trainer(
        model_config=config,
        output_dir=output_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
    )

    # Train the model
    results = trainer.train(
        batch_size=batch_size,
        max_length=max_length,
        epochs=epochs,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
    )

    # Print the results
    print(f"Experiment completed. Best F1 score: {results['best_f1']}")

    return results


def main():
    """Main function to parse arguments and run the experiment."""
    parser = argparse.ArgumentParser(description="Run a WT5 experiment for IMDB sentiment analysis")

    # Experiment configuration
    parser.add_argument(
        "--config",
        type=str,
        default="small",
        choices=["tiny", "small", "base"],
        help="Model configuration to use (tiny, small, or base)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_outputs",
        help="Directory to save model and results",
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of steps to accumulate gradients",
    )

    # Logging and evaluation parameters
    parser.add_argument("--eval_steps", type=int, default=500, help="Steps between evaluations")
    parser.add_argument("--logging_steps", type=int, default=50, help="Steps between logging")
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Steps between saving checkpoints"
    )

    # Wandb parameters
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument(
        "--wandb_project", type=str, default="wt5-sentiment", help="Wandb project name"
    )

    args = parser.parse_args()

    # Create a timestamped run name for the output directory
    if args.output_dir == "./model_outputs":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./models/hf/WT5/model_outputs/{args.config}_run_{timestamp}"

    # Run the experiment
    run_experiment(
        config_name=args.config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )


if __name__ == "__main__":
    main()
