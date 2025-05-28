"""
Utility functions for Qwen2.5-VL models.
"""

import logging
import os
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import requests
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import AutoModelForVision2Seq

# Default model ID
DEFAULT_QWEN_VL_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a Vision Language Model specialized in interpreting visual data
from chart images. Your task is to analyze the provided chart image and respond to queries with
concise answers, usually a single word, number, or short phrase. The charts include a variety of
types (e.g., line charts, bar charts) and contain colors, labels, and text. Focus on delivering
accurate, succinct answers based on the visual information. Avoid additional explanation unless
absolutely necessary."""

logger = logging.getLogger(__name__)


def load_qwen_vl_model_and_processor(
    model_id: str = DEFAULT_QWEN_VL_MODEL_ID,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    model_kwargs: Optional[Dict] = None,
    processor_kwargs: Optional[Dict] = None,
) -> Tuple[AutoModelForVision2Seq, AutoProcessor]:
    """
    Load a Qwen2.5-VL model and its processor from Hugging Face.

    Args:
        model_id: The Hugging Face model identifier.
        device_map: Device map for model loading.
        torch_dtype: torch.dtype for model loading.
        trust_remote_code: Whether to trust remote code from the model hub.
        model_kwargs: Additional keyword arguments for model loading.
        processor_kwargs: Additional keyword arguments for processor loading.

    Returns:
        A tuple containing the loaded model and processor.
    """
    if model_kwargs is None:
        model_kwargs = {}
    if processor_kwargs is None:
        processor_kwargs = {}

    logger.info(f"Loading Qwen2.5-VL model: {model_id}")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=trust_remote_code, **processor_kwargs
    )

    # Load model
    model_load_params = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        **model_kwargs,
    }

    model = AutoModelForVision2Seq.from_pretrained(model_id, **model_load_params)

    model.eval()  # Set to evaluation mode
    logger.info("Model and processor loaded successfully")
    return model, processor


def load_image_pil(image_input: Union[str, Image.Image]) -> Image.Image:
    """
    Load an image from a path, URL, or directly use a PIL image.
    Ensures the image is in RGB format.

    Args:
        image_input: A file path (str), a URL (str), or a PIL.Image.Image object.

    Returns:
        A PIL.Image.Image object in RGB format.
    """
    if isinstance(image_input, Image.Image):
        image = image_input
    elif isinstance(image_input, str):
        if image_input.startswith("http://") or image_input.startswith("https://"):
            try:
                response = requests.get(image_input, stream=True)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                logger.debug(f"Downloaded image from URL: {image_input}")
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Could not download image from {image_input}: {e}")
        elif os.path.exists(image_input):
            image = Image.open(image_input)
            logger.debug(f"Loaded image from path: {image_input}")
        else:
            raise FileNotFoundError(f"Image file not found at {image_input}")
    else:
        raise TypeError(
            f"Unsupported image_input type: {type(image_input)}. Expected str or PIL.Image.Image."
        )

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def format_qwen_vl_messages(
    question: str,
    image_input: Optional[Union[str, Image.Image]] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, any]]:
    """
    Format messages for Qwen2.5-VL chat template.

    Args:
        question: The user's question.
        image_input: Optional image (path, URL, or PIL Image).
        system_prompt: System prompt.

    Returns:
        List of message dictionaries for the chat template.
    """
    messages = []

    # Add system message if provided
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    # Add user message
    user_content = []
    if image_input is not None:
        # Add image placeholder - the actual image will be passed separately to the processor
        user_content.append({"type": "image"})

    user_content.append({"type": "text", "text": question})

    messages.append({"role": "user", "content": user_content})

    return messages


def generate_text_from_sample(
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    question: str,
    image_input: Optional[Union[str, Image.Image]] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> Tuple[str, str]:
    """
    Generate text response from a Qwen2.5-VL model for a given question and optional image.

    Args:
        model: The Qwen2.5-VL model.
        processor: The Qwen2.5-VL processor.
        question: The question to ask.
        image_input: Optional image (path, URL, or PIL Image).
        system_prompt: System prompt.
        max_new_tokens: Maximum number of new tokens to generate.
        do_sample: Whether to use sampling for generation.
        temperature: Temperature for generation.

    Returns:
        Tuple of (generated_text, formatted_prompt_string)
    """
    # Load image if provided
    image_object = None
    if image_input is not None:
        image_object = load_image_pil(image_input)

    # Format messages
    messages = format_qwen_vl_messages(question, image_input, system_prompt)

    # Apply chat template to get the formatted prompt string
    formatted_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        add_vision_id=(image_object is not None),
    )

    # Prepare inputs for the model
    if image_object is not None:
        inputs = processor(text=[formatted_prompt], images=[image_object], return_tensors="pt")
    else:
        inputs = processor(text=[formatted_prompt], return_tensors="pt")

    # Move inputs to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=processor.tokenizer.eos_token_id,
            temperature=temperature,
        )

    # Decode only the new tokens (exclude input tokens)
    input_token_len = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_token_len:]

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text, formatted_prompt


def format_data(sample: Dict) -> List[Dict]:
    """
    Format a dataset sample into the chat message format for Qwen2.5-VL.
    Based on the format_data function from the fine-tuning notebook.

    Args:
        sample: Dataset sample with image, query/question, and optionally label/answer

    Returns:
        List of message dictionaries for the chat template.
    """
    messages = []

    # Add system message
    messages.append(
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]}
    )

    # Add user message with image and question
    user_content = []

    # Add image if present
    if "image" in sample and sample["image"] is not None:
        user_content.append({"type": "image", "image": sample["image"]})

    # Add question text
    question_text = sample.get("query", sample.get("question", ""))
    user_content.append({"type": "text", "text": question_text})

    messages.append({"role": "user", "content": user_content})

    # Add assistant message if label/answer is present (for training)
    if "label" in sample and sample["label"] is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": sample["label"]}]}
        )
    elif "answer" in sample and sample["answer"] is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]}
        )

    return messages
