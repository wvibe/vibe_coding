#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple inference script to test a WT5 model with pre-trained T5 weights.
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer

from models.hf.WT5.configuration import WT5Config
from models.hf.WT5.modeling import WT5ForConditionalGeneration


def load_pretrained_model(model_name="t5-small"):
    """
    Create a WT5 model and initialize it with T5 pre-trained weights.

    Args:
        model_name (str): Name of the model to load from Hugging Face.

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading pre-trained weights from {model_name}")

    # First load the original T5 model to get its weights
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Create our WT5 model with matching configuration
    config = WT5Config(
        vocab_size=t5_model.config.vocab_size,
        d_model=t5_model.config.d_model,
        d_kv=t5_model.config.d_kv,
        d_ff=t5_model.config.d_ff,
        num_layers=t5_model.config.num_layers,
        num_decoder_layers=t5_model.config.num_decoder_layers,
        num_heads=t5_model.config.num_heads,
        relative_attention_num_buckets=t5_model.config.relative_attention_num_buckets,
        dropout_rate=t5_model.config.dropout_rate,
        layer_norm_epsilon=t5_model.config.layer_norm_epsilon,
        initializer_factor=t5_model.config.initializer_factor,
        feed_forward_proj=t5_model.config.feed_forward_proj,
    )

    # Create WT5 model with this config
    wt5_model = WT5ForConditionalGeneration(config)

    # Load the state dict from T5 model
    wt5_model.load_state_dict(t5_model.state_dict(), strict=False)

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return wt5_model, tokenizer


def generate_text(model, tokenizer, input_text, max_length=50):
    """
    Generate text using the given model and tokenizer.

    Args:
        model: The WT5 model
        tokenizer: The WT5 tokenizer
        input_text (str): Input text to transform
        max_length (int): Maximum output length

    Returns:
        str: Generated text
    """
    # T5 models require input text in a specific format: "translate English to German: {text}"
    # For summarization: "summarize: {text}", etc.

    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate text
    output_ids = model.generate(
        input_ids, max_length=max_length, num_beams=4, no_repeat_ngram_size=2, early_stopping=True
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text


def main():
    """Main function to test model inference."""
    # Load model and tokenizer
    model, tokenizer = load_pretrained_model()

    # Example input for different T5 tasks
    long_text = (
        "A very long article about the history of artificial intelligence "
        "that needs to be summarized. " * 5
    )

    examples = [
        "translate English to German: Hello, how are you?",
        f"summarize: {long_text}",
        "answer: What is the capital of France? context: France is in Europe. "
        "Paris is the capital of France.",
    ]

    print("\n" + "=" * 50)
    print("T5 Inference Examples")
    print("=" * 50)

    # Process each example
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Input: {example[:50]}...")

        # Generate output
        output = generate_text(model, tokenizer, example)
        print(f"Output: {output}")

    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
