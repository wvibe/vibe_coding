import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.vanilla.transformer.transformer import Transformer


class TransformerPredictor:
    def __init__(
        self,
        model: Transformer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.logger = logging.getLogger(__name__)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[List[float]]]:
        batch_size = input_ids.shape[0]

        # Move input to device
        input_ids = input_ids.to(self.device)
        attention_scores_history = []

        # Initialize sequence scores for beam search
        sequence_scores = torch.zeros((batch_size, num_return_sequences), device=self.device)

        for _ in range(max_length):
            outputs, attention_scores = self.model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = (
                    next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=num_return_sequences)

            # Update sequence scores
            next_scores = torch.gather(probs, -1, next_tokens)
            sequence_scores += torch.log(next_scores)

            # Append new tokens to input_ids
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            # Store attention scores
            attention_scores_history.append(
                [score.cpu().numpy().tolist() for score in attention_scores]
            )

            # Check if EOS token is generated
            if eos_token_id is not None and (next_tokens == eos_token_id).any():
                break

        return input_ids, attention_scores_history

    @torch.no_grad()
    def get_attention_maps(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        _, attention_scores = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return [score.cpu().numpy() for score in attention_scores]

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
