import copy

from transformers.configuration_utils import PretrainedConfig


class WT5Config(PretrainedConfig):
    """
    Configuration class for WT5 model.
    This class is a subclass of PretrainedConfig and contains all the parameters required to
    build a WT5 model.
    """

    model_type = "wt5"

    def __init__(
        self,
        vocab_size=32128,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=8,
        num_decoder_layers=None,
        num_heads=8,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        is_gated_act=False,
        dense_act_fn="relu",
        decoder_start_token_id=0,
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
            num_decoder_layers: Number of decoder layers (if different from num_layers)
            num_heads: Number of attention heads
            relative_attention_num_buckets: Number of buckets for relative attention
            relative_attention_max_distance: Maximum distance for relative attention
            dropout_rate: Dropout probability
            layer_norm_epsilon: Epsilon for layer normalization
            initializer_factor: Factor for initializer scaling
            feed_forward_proj: Activation function for feed forward layer
            is_encoder_decoder: Whether this is an encoder-decoder model
            use_cache: Whether to use cache for decoding
            pad_token_id: ID of the padding token
            eos_token_id: ID of the end-of-sequence token
            is_gated_act: Whether to use gated activation function
            dense_act_fn: Activation function for dense layers
            decoder_start_token_id: ID of the decoder start token (usually set to
                pad_token_id in T5)
            **kwargs: Additional arguments
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers or num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
        self.is_gated_act = is_gated_act
        self.dense_act_fn = dense_act_fn
        self.decoder_start_token_id = decoder_start_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )

    def copy(self):
        """
        Create a copy of the config.

        Returns:
            WT5Config: A copy of the current configuration.
        """
        return copy.deepcopy(self)
