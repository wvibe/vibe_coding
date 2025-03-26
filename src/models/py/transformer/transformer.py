from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.py.transformer.config import TransformerConfig
from src.models.py.transformer.layers import TransformerLayer


class TransformerEmbeddings(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Create position IDs matrix
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = word_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embeddings = TransformerEmbeddings(config)
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def get_attention_mask(
        self,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = self.get_attention_mask(attention_mask)
        hidden_states = self.embeddings(input_ids, position_ids)

        all_attention_probs = []
        for layer in self.layers:
            hidden_states, attention_probs = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
            )
            all_attention_probs.append(attention_probs)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states, torch.stack(all_attention_probs)
