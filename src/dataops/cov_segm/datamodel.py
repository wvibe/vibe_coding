from typing import List, Optional

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
