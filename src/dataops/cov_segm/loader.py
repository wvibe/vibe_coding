# Standard library imports
import io
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
from PIL import Image

# Local imports
from src.dataops.common.s3_fetcher import fetch_s3_uri, is_s3_uri
from src.dataops.cov_segm.datamodel import (
    ClsSegment,
    ConversationItem,
    InstanceMask,
    SegmMask,
    SegmSample,
)

# Configure logging
logger = logging.getLogger(__name__)


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
    try:
        parsed_data = json.loads(json_string)
        if not isinstance(parsed_data, list):
            raise ValueError("Conversations JSON must be a list.")

        parsed_items = [ConversationItem.model_validate(item) for item in parsed_data]
        return parsed_items
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    except Exception as e:
        raise ValueError(f"Invalid conversation data structure: {e}") from e


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
        logger.debug(f"Attempting to fetch image from: {uri}")
        content_bytes = fetch_s3_uri(uri)
        if content_bytes:
            image = Image.open(io.BytesIO(content_bytes))
            logger.debug(
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


def _resolve_reference_path(
    column_path: str, hf_cov_segm_row: Dict[str, Any]
) -> Tuple[Optional[Any], bool]:
    """Resolves a column path (direct or nested like 'masks_rest/0') to its data value.

    Handles nested paths assuming a 'base_rest/index' structure.

    Args:
        column_path: The column path string (e.g., 'mask_0' or 'masks_rest/0')
        hf_cov_segm_row: The raw dictionary representing one row from the dataset

    Returns:
        A tuple containing (resolved_value, success_flag)
    """
    if "/" in column_path:
        parts = column_path.split("/", 1)  # Split only once
        if len(parts) == 2:
            base_column, index_str = parts
            # Optional check: if not base_column.endswith("_rest"):
            #    logger.warning(...) # Or raise error
            if base_column in hf_cov_segm_row and index_str.isdigit():
                base_value = hf_cov_segm_row[base_column]
                if isinstance(base_value, list):
                    try:
                        index = int(index_str)
                        resolved_value = base_value[index]
                        logger.debug(
                            f"Successfully resolved nested path: '{column_path}' to index {index}"
                        )
                        return resolved_value, True
                    except IndexError:
                        logger.warning(
                            f"Index {index_str} out of range for '{base_column}' "
                            f"(length: {len(base_value)}) in path '{column_path}'"
                        )
                    except Exception as e:  # Catch other potential errors
                        logger.error(f"Error accessing index {index_str} in '{base_column}': {e}")
                else:
                    logger.warning(f"Base '{base_column}' is not a list for path '{column_path}'")
            else:
                # Log conditions separately for clarity
                if base_column not in hf_cov_segm_row:
                    logger.warning(
                        f"Base column '{base_column}' not found for path '{column_path}'"
                    )
                if not index_str.isdigit():
                    logger.warning(f"Non-digit index '{index_str}' in path '{column_path}'")
        else:
            # Path contained '/' but didn't split into two parts
            logger.warning(f"Invalid nested path format: '{column_path}'")

    # Standard direct column access (or if nested lookup failed)
    elif column_path in hf_cov_segm_row:
        logger.debug(f"Successfully resolved direct path: '{column_path}'")
        return hf_cov_segm_row[column_path], True
    else:
        logger.warning(f"Column '{column_path}' not found in dataset row")

    # Default return if any failure occurs above
    return None, False


def _process_mask_list(
    mask_metadata_list: Optional[List[InstanceMask]],
    hf_row: Dict[str, Any],
    row_id: str,
    conv_idx: int,
    mask_type_desc: str,  # e.g., "visible" or "full"
) -> List[SegmMask]:
    """Processes a list of InstanceMask metadata to generate valid SegmMask objects.

    Args:
        mask_metadata_list: List of InstanceMask objects or None.
        hf_row: The raw dataset row dictionary.
        row_id: The ID of the current sample (for logging).
        conv_idx: The index of the current conversation item (for logging).
        mask_type_desc: Description of the mask type being processed (for logging).

    Returns:
        A list of valid SegmMask objects.
    """
    valid_masks: List[SegmMask] = []
    if not mask_metadata_list:
        return valid_masks  # Return empty list if input is None or empty

    for mask_metadata in mask_metadata_list:
        if not mask_metadata.column:
            logger.warning(
                f"Row {row_id}, Conv {conv_idx}: Skipping {mask_type_desc} InstanceMask "
                f"due to missing 'column': {mask_metadata}"
            )
            continue

        column_path = mask_metadata.column
        raw_data, success = _resolve_reference_path(column_path, hf_row)

        if success and isinstance(raw_data, (Image.Image, np.ndarray)):
            try:
                # Instantiate SegmMask - it handles parsing internally
                segm_mask = SegmMask(instance_mask_info=mask_metadata, raw_mask_data=raw_data)
                if segm_mask.is_valid:
                    valid_masks.append(segm_mask)
                    logger.debug(
                        f"Row {row_id}, Conv {conv_idx}: Parsed valid {mask_type_desc} mask "
                        f"from '{column_path}' (Area: {segm_mask.pixel_area})"
                    )
                else:
                    # Log if parsing happened but result was invalid (e.g., zero area)
                    logger.debug(
                        f"Row {row_id}, Conv {conv_idx}: Parsed {mask_type_desc} mask from "
                        f"'{column_path}' but it was invalid (e.g., zero area)."
                    )
            except Exception as e:
                # Catch errors during SegmMask instantiation/parsing
                logger.error(
                    f"Row {row_id}, Conv {conv_idx}: Error processing {mask_type_desc} mask "
                    f"from '{column_path}': {e}",
                    exc_info=True,
                )
        else:
            # Log if raw data couldn't be loaded/resolved or was wrong type
            logger.warning(
                f"Row {row_id}, Conv {conv_idx}: Failed load/resolve {mask_type_desc} mask "
                f"from '{column_path}'. Success: {success}, Type: {type(raw_data).__name__}"
            )

    return valid_masks


def load_sample(hf_cov_segm_row: Dict[str, Any]) -> Optional[SegmSample]:
    """Loads and processes a single row from the Hugging Face cov_segm dataset.

    Parses 'conversations' JSON, fetches the main image specified via S3 URI,
    loads associated raw masks, and uses SegmMask class to parse them.

    Args:
        hf_cov_segm_row: A dictionary representing a raw row obtained from
                         `datasets.load_dataset('lab42/cov-segm-v3')`.

    Returns:
        A SegmSample object containing the main image and processed segments
        (including parsed SegmMask objects), or None if essential components
        (conversations, main image) cannot be loaded/parsed or no valid
        segments are found.
    """
    row_id = hf_cov_segm_row.get("id")
    if not row_id:
        logger.error("Row missing 'id' field.")
        return None

    conversations_json = hf_cov_segm_row.get("conversations")
    if not conversations_json or not isinstance(conversations_json, str):
        logger.error(f"Row {row_id}: Missing 'conversations' string or value is not a string.")
        return None

    try:
        parsed_conv_items: List[ConversationItem] = parse_conversations(conversations_json)
        if not parsed_conv_items:
            logger.warning(f"Row {row_id}: Parsing conversations resulted in an empty list.")
            return None
    except ValueError as e:
        logger.error(f"Row {row_id}: Failed to parse conversations JSON: {e}", exc_info=False)
        return None

    first_item = parsed_conv_items[0]
    if not first_item.image_uri or not first_item.image_uri.jpg:
        logger.error(
            f"Row {row_id}: First conversation item lacks image_uri or image_uri.jpg field."
        )
        return None

    main_image_uri = first_item.image_uri.jpg
    main_image = _load_image_from_uri(main_image_uri)
    if not main_image:
        logger.error(f"Row {row_id}: Failed to load main image from URI: {main_image_uri}.")
        return None

    all_segments: List[ClsSegment] = []
    for idx, item in enumerate(parsed_conv_items):
        if item.image_uri.jpg != main_image_uri:
            logger.error(
                f"Row {row_id}: Conversation item {idx} has a different image_uri.jpg "
                f"('{item.image_uri.jpg}') than the main image ('{main_image_uri}')."
            )
            return None  # Skip sample if conversation items reference different main images

        # Use the helper function to process mask lists
        visible_segm_masks = _process_mask_list(
            item.instance_masks, hf_cov_segm_row, row_id, idx, "visible"
        )
        full_segm_masks = _process_mask_list(
            item.instance_full_masks, hf_cov_segm_row, row_id, idx, "full"
        )

        # Create ClsSegment only if there are phrases to describe it
        if item.phrases:
            cls_segment = ClsSegment(
                phrases=item.phrases,
                type=item.type,
                visible_masks=visible_segm_masks,
                full_masks=full_segm_masks,
            )
            all_segments.append(cls_segment)
            logger.debug(
                f"Row {row_id}, Conv {idx}: Created ClsSegment with {len(visible_segm_masks)} "
                f"visible and {len(full_segm_masks)} full masks."
            )
        else:
            logger.warning(
                f"Row {row_id}, Conv {idx}: Skipping segment creation due to empty phrases."
            )

    if not all_segments:
        logger.warning(
            f"Row {row_id}: No valid segments were created for this sample. Returning None."
        )
        return None

    segm_sample = SegmSample(id=row_id, image=main_image, segments=all_segments)
    logger.debug(f"Row {row_id}: Successfully loaded SegmSample with {len(all_segments)} segments.")
    return segm_sample
