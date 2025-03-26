from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int = 30000
    max_position_embeddings: int = 512
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: int = 0.1
    attention_probs_dropout_prob: int = 0.1
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads