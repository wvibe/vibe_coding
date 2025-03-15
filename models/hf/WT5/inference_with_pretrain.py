#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple inference script to load and test a T5-small model from Hugging Face.
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer


def load_pretrained_model(model_name="t5-small"):
    """
    Load a pre-trained T5 model and tokenizer from Hugging Face.

    Args:
        model_name (str): Name of the model to load from Hugging Face.

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate_text(model, tokenizer, input_text, max_length=50):
    """
    Generate text using the given model and tokenizer.

    Args:
        model: The T5 model
        tokenizer: The T5 tokenizer
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