# Standard library imports
import argparse
import io
import json
import logging
import os
import pprint
from typing import Any, Dict, List, Optional, Tuple, TypedDict

# Third-party imports
import numpy as np
from PIL import Image

# Local imports
from src.dataops.common.s3_fetcher import fetch_s3_uri, is_s3_uri
from src.dataops.cov_segm.datamodel import ConversationItem, InstanceMask

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Type Definitions for Processed Output ---
class ProcessedMask(TypedDict):
    """Structure holding a loaded mask image and its metadata."""

    mask: Image.Image
    positive_value: int
    source: str  # The column name from which the mask was loaded


class ProcessedConversationItem(TypedDict):
    """Structure holding processed data for a single conversation item."""

    phrases: List[Dict[str, Any]]  # List of phrase dicts (e.g., from model_dump())
    type: str
    processed_instance_masks: List[ProcessedMask]
    processed_full_masks: List[ProcessedMask]


class ProcessedCovSegmSample(TypedDict):
    """Structure holding the fully processed cov_segm sample data."""

    id: str  # Sample ID (from input row or "unknown_id" if missing)
    image: Image.Image  # The main S3 image
    processed_conversations: List[ProcessedConversationItem]


# Add global variable to cache the last parsed ConversationItem objects
_last_parsed_conversations = []  # This will hold the most recently parsed conversation items


def parse_conversations(json_string: str) -> List[ConversationItem]:
    """Parses the 'conversations' JSON string into a list of ConversationItem objects.

    Args:
        json_string: A JSON string representing an array of conversation items,
                    conforming to the structure defined by ConversationItem.

    Returns:
        A list of ConversationItem objects parsed from the JSON string.

    Raises:
        ValueError: If the JSON is invalid or does not conform to the expected schema.
    """
    global _last_parsed_conversations
    try:
        parsed_data = json.loads(json_string)
        if not isinstance(parsed_data, list):
            raise ValueError("Conversations JSON must be a list.")

        parsed_items = [ConversationItem.model_validate(item) for item in parsed_data]
        # Store the parsed items for debugging
        _last_parsed_conversations = parsed_items

        return parsed_items
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    except Exception as e:
        raise ValueError(f"Invalid conversation data structure: {e}") from e


def get_last_parsed_conversations() -> List[Dict[str, Any]]:
    """Returns the most recently parsed conversation items as dictionaries.

    This function is intended for debugging purposes only.

    Returns:
        A list of dictionaries containing the raw parsed conversation items.
    """
    return [item.model_dump() for item in _last_parsed_conversations]


def _load_image_from_uri(uri: str) -> Optional[Image.Image]:
    """Loads an image from an S3 URI.

    Args:
        uri: The S3 URI of the image.

    Returns:
        A PIL Image object, or None if loading fails.
    """
    if not is_s3_uri(uri):
        logger.warning(f"Invalid S3 URI provided for image: {uri}")
        return None
    try:
        logger.info(f"Attempting to fetch image from: {uri}")
        content_bytes = fetch_s3_uri(uri)
        if content_bytes:
            image = Image.open(io.BytesIO(content_bytes))
            logger.info(
                f"Successfully loaded image from {uri}. Format: {image.format}, "
                f"Size: {image.size}, Mode: {image.mode}"
            )
            return image
        else:
            logger.warning(f"Failed to fetch content for image URI: {uri}")
            return None
    except Exception as e:
        logger.error(f"Error loading image from URI {uri}: {e}", exc_info=True)
        return None


def _resolve_mask_path(
    column_path: str, hf_cov_segm_row: Dict[str, Any]
) -> Tuple[Optional[Any], bool]:
    """Resolves a column path (possibly with path-style notation) to its data value.

    Handles both direct column access and path-style notation like 'masks_rest/0'.

    Args:
        column_path: The column path string (e.g., 'mask_0' or 'masks_rest/0')
        hf_cov_segm_row: The raw dictionary representing one row from the dataset

    Returns:
        A tuple containing (resolved_value, success_flag)
        - resolved_value: The data value if found, None otherwise
        - success_flag: True if resolution was successful, False otherwise
    """
    # Handle path-style column references like 'masks_rest/0'
    if "/" in column_path:
        # Split into parts (e.g., ['masks_rest', '0'])
        parts = column_path.split("/")
        base_column = parts[0]

        # Check if the base column exists in the row
        if base_column in hf_cov_segm_row:
            logger.debug(f"Found base column '{base_column}' - resolving path '{column_path}'")
            try:
                # Start with the base column value
                value = hf_cov_segm_row[base_column]

                # Navigate nested indices if it's a list
                for index_str in parts[1:]:
                    if isinstance(value, list) and index_str.isdigit():
                        index = int(index_str)
                        if 0 <= index < len(value):
                            value = value[index]
                            logger.debug(f"Accessed index {index} in list, continuing...")
                        else:
                            logger.warning(
                                f"Index {index} out of range for {base_column} "
                                f"list (length: {len(value)})"
                            )
                            return None, False
                    else:
                        logger.warning(
                            f"Cannot access {index_str} in {type(value).__name__} for {column_path}"
                        )
                        return None, False

                # We've successfully navigated to the value
                logger.debug(f"Successfully accessed nested data at path: '{column_path}'")
                return value, True

            except Exception as e:
                logger.error(f"Error accessing nested path '{column_path}': {e}")
                return None, False
        else:
            logger.warning(f"Base column '{base_column}' not found for path '{column_path}'")
            return None, False

    # Standard direct column access
    elif column_path in hf_cov_segm_row:
        logger.debug(f"Found column '{column_path}' directly in the dataset row")
        return hf_cov_segm_row[column_path], True

    else:
        logger.warning(f"Mask column '{column_path}' not found in dataset row")
        return None, False


def _load_mask(mask_info: InstanceMask, hf_cov_segm_row: Dict[str, Any]) -> Optional[ProcessedMask]:
    """Loads a mask from a specified column in a Hugging Face cov_segm row.

    Handles both direct column access and path-style notation (e.g., 'masks_rest/0').

    Args:
        mask_info: The InstanceMask object containing mask details.
        hf_cov_segm_row: The raw dictionary representing one row from the
                         Hugging Face cov_segm dataset.

    Returns:
        A ProcessedMask TypedDict containing the loaded mask, its positive value,
        and its source column name, or None if loading fails.
    """
    mask_image: Optional[Image.Image] = None
    source: Optional[str] = None

    if mask_info.column:
        column_path = mask_info.column
        source = column_path

        # Move detailed logging to debug level
        logger.debug(f"Attempting to load mask from path: '{column_path}'")

        # Resolve the mask data path to its actual value
        mask_data, success = _resolve_mask_path(column_path, hf_cov_segm_row)

        if not success:
            return None

        # Process the mask data once we have it
        try:
            if isinstance(mask_data, Image.Image):
                mask_image = mask_data
                logger.debug(f"Loaded mask from '{source}' as PIL.Image")
            elif isinstance(mask_data, np.ndarray):
                mask_image = Image.fromarray(mask_data)
                logger.debug(
                    f"Loaded mask from '{source}' as np.ndarray and converted to PIL.Image"
                )
            else:
                logger.warning(f"Unsupported mask type at '{source}': {type(mask_data)}")
                return None
        except Exception as e:
            logger.error(f"Error converting mask data from '{source}': {e}")
            return None
    else:
        logger.warning(f"InstanceMask provided without a 'column' field: {mask_info}")
        return None

    if mask_image and source is not None:
        processed_mask: ProcessedMask = {
            "mask": mask_image,
            "positive_value": mask_info.positive_value,
            "source": source,
        }
        logger.debug(
            f"Successfully loaded mask from '{source}' with "
            f"positive_value={mask_info.positive_value}"
        )
        return processed_mask
    else:
        if mask_info.column:
            logger.warning(f"Failed to load mask from column: {mask_info.column}")
        return None


def _process_mask_metadata(mask_metadata, hf_cov_segm_row) -> Optional[ProcessedMask]:
    """Process mask metadata and load the corresponding mask.

    Handles both direct column references (mask_0) and path-style notation (masks_rest/0).

    Args:
        mask_metadata: The InstanceMask object containing mask details.
        hf_cov_segm_row: The raw dictionary representing one row from the dataset.

    Returns:
        A ProcessedMask or None if loading fails.
    """
    if not mask_metadata.column:
        return None

    # Just directly use _load_mask which already handles both formats properly
    # through the _resolve_mask_path function
    return _load_mask(mask_metadata, hf_cov_segm_row)


def load_sample(hf_cov_segm_row: Dict[str, Any]) -> Optional[ProcessedCovSegmSample]:
    """Loads and processes a single row from the Hugging Face cov_segm dataset.

    Parses 'conversations' JSON, fetches the main image specified via S3 URI,
    and loads associated masks from columns within the input row.

    Args:
        hf_cov_segm_row: A dictionary representing a raw row obtained from
                         `datasets.load_dataset('lab42/cov-segm-v3')`.
                         Expected to have a 'conversations' key (JSON string)
                         and columns like 'mask_0', 'mask_1', etc., containing
                         mask image data (PIL.Image or np.ndarray).

    Returns:
        A ProcessedCovSegmSample TypedDict containing the main image and processed
        conversation data (including loaded mask images), or None if essential
        components (conversations, main image) cannot be loaded/parsed.
    """
    conversations_json = hf_cov_segm_row.get("conversations")
    if not conversations_json or not isinstance(conversations_json, str):
        logger.error("Row missing 'conversations' string or value is not a string.")
        return None

    try:
        parsed_conv_items: List[ConversationItem] = parse_conversations(conversations_json)
        if not parsed_conv_items:
            logger.error("Parsing conversations resulted in an empty list.")
            return None
    except ValueError as e:
        logger.error(f"Failed to parse conversations JSON: {e}", exc_info=True)
        return None

    # --- Debugging: Log raw parsed conversation items if requested ---
    if os.environ.get("VIBE_DEBUG_LOADER") == "1":
        logger.info("-- Debug Loader: Raw Parsed Conversation Items --")
        for idx, item in enumerate(parsed_conv_items):
            logger.info(
                f"Item {idx}: {pprint.pformat(item.model_dump())}"
            )  # Use model_dump for Pydantic obj
        logger.info("-- End Debug Loader --")
    # --- End Debugging ---

    first_item = parsed_conv_items[0]
    if not first_item.image_uri or not first_item.image_uri.jpg:
        logger.error("First conversation item lacks image_uri or image_uri.jpg field.")
        return None
    main_image_uri = first_item.image_uri.jpg

    main_image = _load_image_from_uri(main_image_uri)
    if not main_image:
        return None

    processed_conversations: List[ProcessedConversationItem] = []
    for idx, item in enumerate(parsed_conv_items):
        # For summary logging
        instance_masks_attempted = 0
        instance_masks_loaded = 0
        full_masks_attempted = 0
        full_masks_loaded = 0

        processed_instance_masks: List[ProcessedMask] = []
        if item.instance_masks:
            instance_masks_attempted = len(item.instance_masks)
            for mask_metadata in item.instance_masks:
                loaded_mask = _process_mask_metadata(mask_metadata, hf_cov_segm_row)
                if loaded_mask:
                    processed_instance_masks.append(loaded_mask)
                    instance_masks_loaded += 1
                else:
                    # Construct the potential path for logging if needed
                    potential_path = mask_metadata.column
                    # Log only if mask loading failed, not just missing columns/index
                    if (
                        mask_metadata.column
                        and _resolve_mask_path(potential_path, hf_cov_segm_row)[1]
                    ):
                        logger.warning(
                            f"Failed to load instance mask for sample {idx}, "
                            f"conversation {idx}, definition '{mask_metadata}' (path: {potential_path})"
                        )

        processed_full_masks: List[ProcessedMask] = []
        if item.instance_full_masks:
            full_masks_attempted = len(item.instance_full_masks)
            for mask_metadata in item.instance_full_masks:
                loaded_mask = _process_mask_metadata(mask_metadata, hf_cov_segm_row)
                if loaded_mask:
                    processed_full_masks.append(loaded_mask)
                    full_masks_loaded += 1
                else:
                    # Construct the potential path for logging if needed
                    potential_path = mask_metadata.column
                    # Log only if mask loading failed, not just missing columns/index
                    if (
                        mask_metadata.column
                        and _resolve_mask_path(potential_path, hf_cov_segm_row)[1]
                    ):
                        logger.warning(
                            f"Failed to load instance full mask for sample {idx}, "
                            f"conversation {idx}, definition '{mask_metadata}' (path: {potential_path})"
                        )

        # Log summary info at INFO level
        phrases_text = [p.text for p in item.phrases]
        logger.info(
            f"ConversationItem {idx} with phrases {phrases_text}: "
            + f"Loaded {instance_masks_loaded}/{instance_masks_attempted} instance masks, "
            + f"{full_masks_loaded}/{full_masks_attempted} full masks"
        )

        processed_item: ProcessedConversationItem = {
            "phrases": [p.model_dump() for p in item.phrases],
            "type": item.type,
            "processed_instance_masks": processed_instance_masks,
            "processed_full_masks": processed_full_masks,
        }
        processed_conversations.append(processed_item)

    # Extract sample ID or use default if missing
    sample_id = hf_cov_segm_row.get("id", "unknown_id")

    final_result: ProcessedCovSegmSample = {
        "id": sample_id,
        "image": main_image,
        "processed_conversations": processed_conversations,
    }
    return final_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test loading an image from an S3 URI.")
    parser.add_argument(
        "s3_uri", help="The S3 URI of the image to load (e.g., s3://bucket/image.jpg)"
    )
    args = parser.parse_args()

    print(f"Attempting to load image from: {args.s3_uri}")
    loaded_image = _load_image_from_uri(args.s3_uri)

    if loaded_image:
        print("-" * 20)
        print("Image loaded successfully!")
        print(f"  Format: {loaded_image.format}")
        print(f"  Size: {loaded_image.size}")
        print(f"  Mode: {loaded_image.mode}")
        print("-" * 20)
    else:
        print("-" * 20)
        print("Failed to load image.")
        print("-" * 20)
