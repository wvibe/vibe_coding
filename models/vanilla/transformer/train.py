import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vanilla.transformer.transformer import Transformer


class TransformerTrainer:
    def __init__(
        self,
        model: Transformer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.01
        )
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model.to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.logger = logging.getLogger(__name__)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        labels = batch["labels"].to(self.device)

        outputs, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        loss = self.criterion(outputs.view(-1, self.model.config.vocab_size), labels.view(-1))

        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item()

    def validate(self) -> float:
        if self.val_dataloader is None:
            return 0.0

        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_dataloader)

        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                labels = batch["labels"].to(self.device)

                outputs, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

                loss = self.criterion(
                    outputs.view(-1, self.model.config.vocab_size), labels.view(-1)
                )
                total_loss += loss.item()

        return total_loss / num_batches

    def train(
        self, num_epochs: int, save_path: Optional[str] = None, save_steps: int = 1000
    ) -> Dict[str, Any]:
        best_val_loss = float("inf")
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            epoch_losses = []
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for step, batch in enumerate(progress_bar):
                loss = self.train_step(batch)
                epoch_losses.append(loss)

                if (step + 1) % save_steps == 0 and save_path:
                    self.save_checkpoint(save_path, f"checkpoint_epoch{epoch + 1}_step{step + 1}")

                progress_bar.set_postfix({"loss": sum(epoch_losses) / len(epoch_losses)})

            train_loss = sum(epoch_losses) / len(epoch_losses)
            train_losses.append(train_loss)

            if self.val_dataloader is not None:
                val_loss = self.validate()
                val_losses.append(val_loss)

                if val_loss < best_val_loss and save_path:
                    best_val_loss = val_loss
                    self.save_checkpoint(save_path, "best_model")

                self.logger.info(
                    f"Epoch {epoch + 1}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}"
                )
            else:
                self.logger.info(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
        }

    def save_checkpoint(self, save_path: str, name: str) -> None:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config,
        }
        if self.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, f"{save_path}/{name}.pt")
