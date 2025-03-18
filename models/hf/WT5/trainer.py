import argparse
import os
import time

import torch
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, T5Tokenizer, get_linear_schedule_with_warmup

from models.hf.WT5.configuration import WT5Config
from models.hf.WT5.modeling import WT5ForConditionalGeneration

# Load environment variables from .env file
load_dotenv()


class IMDBDataset(Dataset):
    """
    Dataset for IMDB sentiment analysis using WT5.
    This formats the sentiment classification task as a text-to-text problem.
    """

    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = f"Review: {item['text']}"

        # Convert sentiment label (0 or 1) to text
        label = "negative" if item["label"] == 0 else "positive"

        # Tokenize input and output
        input_encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            label,
            max_length=10,  # Short output
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = input_encoding["input_ids"].squeeze()
        attention_mask = input_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()
        # Replace padding token id with -100 so it's ignored in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class WT5Trainer:
    """
    Trainer for WT5 model for sentiment analysis.
    """

    def __init__(
        self,
        model_config,
        tokenizer_name="t5-small",
        output_dir="./output",
        device=None,
        use_wandb=True,
        wandb_project="wt5-sentiment",
    ):
        self.model_config = model_config
        # Add cache directory and handle possible network errors
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name, local_files_only=False)
        except OSError:
            print(f"Unable to load tokenizer '{tokenizer_name}' online, trying 't5-small'...")
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small", local_files_only=False)
        self.output_dir = output_dir
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Using device: {self.device}")
        self.model = self._create_model()

    def _create_model(self):
        """Create and initialize the WT5 model."""
        model = WT5ForConditionalGeneration(self.model_config)
        model.to(self.device)
        return model

    def _prepare_data(self, batch_size, max_length):
        """Load and prepare IMDB dataset."""
        # Load IMDB dataset
        dataset = load_dataset("stanfordnlp/imdb")

        # Create train and validation datasets
        train_dataset = IMDBDataset(dataset["train"], self.tokenizer, max_length=max_length)
        val_dataset = IMDBDataset(
            dataset["test"], self.tokenizer, max_length=max_length  # Using test split as validation
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def _calculate_perplexity(self, loss):
        """Calculate perplexity from loss."""
        return torch.exp(loss).item()

    def _log_gpu_memory_usage(self):
        """Log GPU memory usage if CUDA is available."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
            return {"gpu_allocated_memory_mb": allocated, "gpu_reserved_memory_mb": reserved}
        return {}

    def train(
        self,
        batch_size=8,
        max_length=512,
        learning_rate=5e-5,
        weight_decay=0.01,
        epochs=3,
        warmup_steps=0,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
    ):
        """Train the model."""
        # Initialize wandb if enabled
        if self.use_wandb:
            # Use API key from environment if available
            api_key = os.getenv("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key)

            wandb.init(
                project=self.wandb_project,
                config={
                    "architecture": "WT5",
                    "dataset": "IMDB",
                    "batch_size": batch_size,
                    "max_length": max_length,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "warmup_steps": warmup_steps,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "device": self.device,
                    "model_config": self.model_config.to_dict(),
                },
            )

        train_loader, val_loader = self._prepare_data(batch_size, max_length)

        # Prepare optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Training loop
        global_step = 0
        best_f1 = 0.0
        epoch_start_time = time.time()

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_step_time = []
            step_start_time = time.time()

            # Progress bar for training
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss

                # Normalize loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()

                epoch_loss += loss.item()

                # Calculate step time for steps/second metric
                if step > 0:
                    step_time = time.time() - step_start_time
                    epoch_step_time.append(step_time)
                    steps_per_second = 1.0 / step_time if step_time > 0 else 0
                step_start_time = time.time()

                # Log training progress
                if (step + 1) % logging_steps == 0:
                    # Calculate current learning rate
                    current_lr = scheduler.get_last_lr()[0]

                    # Calculate perplexity
                    perplexity = self._calculate_perplexity(loss * gradient_accumulation_steps)

                    # Get gradient norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    ).item()

                    # Log to wandb
                    if self.use_wandb:
                        wandb_log = {
                            "train/loss": epoch_loss / (step + 1),
                            "train/perplexity": perplexity,
                            "train/learning_rate": current_lr,
                            "train/gradient_norm": grad_norm,
                            "train/steps_per_second": steps_per_second if step > 0 else 0,
                            "global_step": global_step,
                        }

                        # Add GPU memory metrics if available
                        wandb_log.update(self._log_gpu_memory_usage())

                        wandb.log(wandb_log)

                    progress_bar.set_postfix(
                        {
                            "loss": epoch_loss / (step + 1),
                            "lr": current_lr,
                            "perplexity": perplexity,
                        }
                    )

                # Update weights
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    # Update weights
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1

                    # Evaluate and save model
                    if global_step % eval_steps == 0:
                        metrics = self.evaluate(val_loader)
                        print(f"Eval metrics at step {global_step}: {metrics}")

                        # Log evaluation metrics to wandb
                        if self.use_wandb:
                            wandb.log(
                                {
                                    "eval/accuracy": metrics["accuracy"],
                                    "eval/f1": metrics["f1"],
                                    "eval/precision": metrics["precision"],
                                    "eval/recall": metrics["recall"],
                                    "global_step": global_step,
                                    **self._log_gpu_memory_usage(),
                                }
                            )

                            # Log confusion matrix as a visualization
                            if "confusion_matrix" in metrics:
                                wandb.log(
                                    {
                                        "eval/confusion_matrix": wandb.plot.confusion_matrix(
                                            probs=None,
                                            y_true=metrics["true_labels"],
                                            preds=metrics["pred_labels"],
                                            class_names=["negative", "positive"],
                                        ),
                                        "global_step": global_step,
                                    }
                                )

                        # Save best model
                        if metrics["f1"] > best_f1:
                            best_f1 = metrics["f1"]
                            self.save_model(os.path.join(self.output_dir, "best_model"))

                    if global_step % save_steps == 0:
                        self.save_model(os.path.join(self.output_dir, f"checkpoint-{global_step}"))

            # Epoch completed metrics
            epoch_time = time.time() - epoch_start_time
            avg_steps_per_second = len(train_loader) / epoch_time if epoch_time > 0 else 0

            # Evaluate at the end of each epoch
            metrics = self.evaluate(val_loader)
            print(f"Epoch {epoch+1} metrics: {metrics}")

            # Log epoch summary to wandb
            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "epoch/training_time_seconds": epoch_time,
                        "epoch/avg_steps_per_second": avg_steps_per_second,
                        "epoch/loss": epoch_loss / len(train_loader),
                        "epoch/accuracy": metrics["accuracy"],
                        "epoch/f1": metrics["f1"],
                        "epoch/precision": metrics["precision"],
                        "epoch/recall": metrics["recall"],
                        **self._log_gpu_memory_usage(),
                    }
                )

                # Log confusion matrix as a visualization
                if "confusion_matrix" in metrics:
                    wandb.log(
                        {
                            "epoch/confusion_matrix": wandb.plot.confusion_matrix(
                                probs=None,
                                y_true=metrics["true_labels"],
                                preds=metrics["pred_labels"],
                                class_names=["negative", "positive"],
                            ),
                            "epoch": epoch + 1,
                        }
                    )

            # Save model at the end of each epoch
            self.save_model(os.path.join(self.output_dir, f"epoch-{epoch+1}"))

            # Reset epoch timer for next epoch
            epoch_start_time = time.time()

        # Save final model
        self.save_model(os.path.join(self.output_dir, "final_model"))

        # Close wandb run
        if self.use_wandb:
            wandb.finish()

        return {"best_f1": best_f1}

    def evaluate(self, data_loader):
        """Evaluate the model."""
        self.model.eval()
        all_preds = []
        all_labels = []
        eval_loss = 0.0
        eval_steps = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Get loss
                outputs = self.model(**batch)
                eval_loss += outputs.loss.item()
                eval_steps += 1

                # Get model predictions
                gen_outputs = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=10,
                )

                # Decode predictions
                preds = self.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)

                # Convert text predictions to labels (0 or 1)
                pred_labels = [1 if p.strip().lower() == "positive" else 0 for p in preds]

                # Decode true labels
                true_labels = []
                for label_ids in batch["labels"]:
                    # Remove -100 tokens
                    label_ids = label_ids.masked_fill(
                        label_ids == -100, self.tokenizer.pad_token_id
                    )
                    label_text = self.tokenizer.decode(label_ids, skip_special_tokens=True)
                    true_labels.append(1 if label_text.strip().lower() == "positive" else 0)

                all_preds.extend(pred_labels)
                all_labels.extend(true_labels)

        # Calculate average loss and perplexity
        avg_loss = eval_loss / eval_steps if eval_steps > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds).tolist()

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "loss": avg_loss,
            "perplexity": perplexity,
            "confusion_matrix": cm,
            "true_labels": all_labels,
            "pred_labels": all_preds,
        }

    def save_model(self, path):
        """Save model and configuration."""
        if not os.path.exists(path):
            os.makedirs(path)

        # Save model weights with safe_serialization=False to handle shared tensors
        self.model.save_pretrained(path, safe_serialization=False)

        # Save tokenizer for convenience
        self.tokenizer.save_pretrained(path)

        print(f"Model saved to {path}")

    @classmethod
    def from_pretrained(cls, model_path, device=None):
        """Load pretrained model."""
        # Load configuration
        config = WT5Config.from_pretrained(model_path)

        # Create trainer instance
        trainer = cls(model_config=config, output_dir=os.path.dirname(model_path), device=device)

        # Load model weights
        trainer.model = WT5ForConditionalGeneration.from_pretrained(model_path)
        trainer.model.to(trainer.device)

        return trainer


def main():
    """Main function to run training from command line."""
    parser = argparse.ArgumentParser(description="Train WT5 model for sentiment analysis")

    # Model configuration
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--d_kv", type=int, default=64, help="Key/Value dimension")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of encoder layers")
    parser.add_argument(
        "--num_decoder_layers", type=int, default=8, help="Number of decoder layers"
    )
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--vocab_size", type=int, default=32128, help="Vocabulary size")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")

    # Output and checkpointing
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Steps between saving checkpoints"
    )
    parser.add_argument("--eval_steps", type=int, default=1000, help="Steps between evaluations")
    parser.add_argument("--logging_steps", type=int, default=100, help="Steps between logging")

    # Wandb configuration
    parser.add_argument(
        "--use_wandb", action="store_true", help="Whether to use Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="wt5-sentiment", help="W&B project name"
    )

    args = parser.parse_args()

    # Create model configuration
    config = WT5Config(
        d_model=args.d_model,
        d_kv=args.d_kv,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
    )

    # Create trainer
    trainer = WT5Trainer(
        model_config=config,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )

    # Train model
    trainer.train(
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
    )


if __name__ == "__main__":
    main()
