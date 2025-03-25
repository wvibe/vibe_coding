import pytest
import torch

from models.vanilla.transformer.config import TransformerConfig
from models.vanilla.transformer.layers import (
    FeedForward,
    MultiHeadAttention,
    TransformerLayer,
)
from models.vanilla.transformer.transformer import Transformer, TransformerEmbeddings


@pytest.fixture
def transformer_config():
    return TransformerConfig(
        vocab_size=1000,
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
    )


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_length():
    return 10


def test_embeddings(transformer_config, batch_size, seq_length):
    embeddings = TransformerEmbeddings(transformer_config)
    input_ids = torch.randint(
        0, transformer_config.vocab_size, (batch_size, seq_length)
    )

    output = embeddings(input_ids)

    assert output.shape == (batch_size, seq_length, transformer_config.hidden_size)


def test_attention(transformer_config, batch_size, seq_length):
    attention = MultiHeadAttention(transformer_config)
    hidden_states = torch.randn(batch_size, seq_length, transformer_config.hidden_size)

    output, attention_probs = attention(hidden_states)

    assert output.shape == (batch_size, seq_length, transformer_config.hidden_size)
    assert attention_probs.shape == (
        batch_size,
        transformer_config.num_attention_heads,
        seq_length,
        seq_length,
    )


def test_feed_forward(transformer_config, batch_size, seq_length):
    feed_forward = FeedForward(transformer_config)
    hidden_states = torch.randn(batch_size, seq_length, transformer_config.hidden_size)

    output = feed_forward(hidden_states)

    assert output.shape == (batch_size, seq_length, transformer_config.hidden_size)


def test_transformer_layer(transformer_config, batch_size, seq_length):
    layer = TransformerLayer(transformer_config)
    hidden_states = torch.randn(batch_size, seq_length, transformer_config.hidden_size)

    output, attention_probs = layer(hidden_states)

    assert output.shape == (batch_size, seq_length, transformer_config.hidden_size)
    assert attention_probs.shape == (
        batch_size,
        transformer_config.num_attention_heads,
        seq_length,
        seq_length,
    )


@pytest.fixture
def transformer_model(transformer_config):
    return Transformer(transformer_config)


def test_forward(transformer_model, transformer_config, batch_size, seq_length):
    input_ids = torch.randint(
        0, transformer_config.vocab_size, (batch_size, seq_length)
    )
    attention_mask = torch.ones(batch_size, seq_length)

    outputs, attention_probs = transformer_model(
        input_ids, attention_mask=attention_mask
    )

    assert outputs.shape == (batch_size, seq_length, transformer_config.hidden_size)
    assert len(attention_probs) == transformer_config.num_hidden_layers
    assert attention_probs[0].shape == (
        batch_size,
        transformer_config.num_attention_heads,
        seq_length,
        seq_length,
    )


def test_attention_mask(transformer_model, transformer_config, batch_size, seq_length):
    input_ids = torch.randint(
        0, transformer_config.vocab_size, (batch_size, seq_length)
    )
    attention_mask = torch.zeros(batch_size, seq_length)
    attention_mask[:, :5] = 1  # Only attend to first 5 tokens

    outputs, attention_probs = transformer_model(
        input_ids, attention_mask=attention_mask
    )

    # Check that attention probabilities are zero for masked positions
    for layer_attention in attention_probs:
        masked_attention = layer_attention[:, :, :, 5:]
        assert torch.allclose(
            masked_attention, torch.zeros_like(masked_attention), atol=1e-4
        )
