from typing import Any, Dict, List, Optional, TypedDict

from PIL import Image
from pydantic import BaseModel


class Phrase(BaseModel):
    """Represents a phrase within a conversation item."""

    id: int
    text: str
    type: str


class ImageURI(BaseModel):
    """Represents the S3 URI for an image."""

    jpg: str
    format: str


class InstanceMask(BaseModel):
    """Represents information about an instance mask, which can be an S3 URI or a column name."""

    column: str  # Column name in the dataset if mask is not in S3
    image_uri: ImageURI  # S3 URI for the mask image
    positive_value: int  # Pixel value indicating the positive class


class ConversationItem(BaseModel):
    """Represents a single item in the 'conversations' list."""

    phrases: List[Phrase]
    image_uri: ImageURI
    instance_masks: Optional[List[InstanceMask]] = None
    instance_full_masks: Optional[List[InstanceMask]] = None
    type: str


# --- TypedDicts for Processed Data ---


class ProcessedMask(TypedDict):
    """Structure holding a loaded mask image and its metadata."""

    mask: Image.Image
    positive_value: int  # Keep original positive_value for reference/debugging
    source: str  # The column name from which the mask was loaded
    pixel_area: Optional[int]  # Calculated pixel count matching positive_value
    width: Optional[int]  # Calculated bounding box width of the mask pixels
    height: Optional[int]  # Calculated bounding box height of the mask pixels


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
