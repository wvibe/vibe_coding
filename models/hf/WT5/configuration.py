from typing import Optional

from transformers import T5Config


class WT5Config(T5Config):
    """
    Configuration class for WT5, a customizable implementation of T5 model.
    This class extends HuggingFace's T5Config to allow for easier experimentation with
    different model configurations while maintaining compatibility with the HF ecosystem.
    """

    model_type = "wt5"

    def __init__(
        self,
        vocab_size: int = 32128,
        d_model: int = 512,  # Reduced from standard T5-base (768)
        d_kv: int = 64,
        d_ff: int = 1024,  # Reduced from standard T5-base (2048)
        num_layers: int = 6,  # Reduced from standard T5-base (12)
        num_decoder_layers: Optional[int] = None,
        num_heads: int = 8,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        initializer_factor: float = 1.0,
        feed_forward_proj: str = "relu",
        is_encoder_decoder: bool = True,
        use_cache: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 0,
        **kwargs,
    ):
        """
        Initialize a WT5Config instance.

        Args:
            vocab_size: Vocabulary size
            d_model: Size of the hidden states
            d_kv: Size of the key, query, value projections per attention head
            d_ff: Size of the intermediate feed forward layer
            num_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers (if different from encoder)
            num_heads: Number of attention heads
            relative_attention_num_buckets: Number of buckets for relative attention
            relative_attention_max_distance: Maximum distance for relative attention
            dropout_rate: Dropout rate
            layer_norm_epsilon: Epsilon value for layer normalization
            initializer_factor: Factor for initializing weights
            feed_forward_proj: Non-linear activation for feed forward layer
            is_encoder_decoder: Whether the model is an encoder-decoder model
            use_cache: Whether to use cache for decoding
            pad_token_id: ID of padding token
            eos_token_id: ID of EOS token
            bos_token_id: ID of BOS token
        """
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            d_kv=d_kv,
            d_ff=d_ff,
            num_layers=num_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            dropout_rate=dropout_rate,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_factor=initializer_factor,
            feed_forward_proj=feed_forward_proj,
            is_encoder_decoder=is_encoder_decoder,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.bos_token_id = bos_token_id