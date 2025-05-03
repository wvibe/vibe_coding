# Standard library imports
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
from PIL import Image
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)


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
    """Represents information about an instance mask, used for loading.

    This Pydantic model holds the metadata needed to locate and interpret
    a specific instance mask from the raw dataset.
    """

    column: str  # Column name/path in the dataset (e.g., 'mask_0' or 'masks_rest/0')
    # image_uri field seems redundant here if mask is always in a column
    # Consider if this field is truly needed or if it was from an older design.
    # For now, keeping it based on original Pydantic model.
    image_uri: ImageURI  # S3 URI for the mask image (Potentially unused if column is primary)
    positive_value: int  # Pixel value indicating the positive class


class ConversationItem(BaseModel):
    """Represents a single item in the 'conversations' list from the raw dataset."""

    phrases: List[Phrase]
    image_uri: ImageURI  # URI of the main image for this conversation block
    instance_masks: Optional[List[InstanceMask]] = None
    instance_full_masks: Optional[List[InstanceMask]] = None
    type: str


# --- New OOP Classes for Processed Data ---


class SegmMask:
    """Represents a parsed segmentation mask with binary representation and geometry.

    This class encapsulates the logic to parse raw mask data (PIL Image or NumPy array)
    based on a specific positive value, generating a boolean binary mask and calculating
    relevant geometric properties like pixel area and bounding box.

    Attributes:
        source_info: The original InstanceMask metadata used for loading.
        binary_mask: Boolean NumPy array (True where pixels match the positive value),
                     or None if parsing failed.
        pixel_area: Number of True pixels in binary_mask, or None if parsing failed.
        bbox: Bounding box tuple (x_min, y_min, x_max, y_max) of the True pixels,
              or None if parsing failed or area is 0.
        is_valid: True if parsing succeeded and pixel_area > 0.
    """

    source_info: InstanceMask
    binary_mask: Optional[np.ndarray]
    pixel_area: Optional[int]
    bbox: Optional[Tuple[int, int, int, int]]  # (x_min, y_min, x_max, y_max)
    is_valid: bool

    def __init__(
        self, instance_mask_info: InstanceMask, raw_mask_data: Union[Image.Image, np.ndarray]
    ):
        """Initializes SegmMask, stores source info, and triggers parsing."""
        self.source_info = instance_mask_info
        # Initialize attributes to default invalid state
        self.binary_mask = None
        self.pixel_area = None
        self.bbox = None
        self.is_valid = False
        # Parse the raw data
        self._parse(raw_mask_data)

    def _parse(self, raw_mask_data: Union[Image.Image, np.ndarray]) -> None:
        """Parses raw mask data to generate binary mask and geometry."""
        positive_value = self.source_info.positive_value
        source_desc = self.source_info.column  # For logging

        try:
            # 1. Ensure NumPy array
            if isinstance(raw_mask_data, Image.Image):
                np_mask = np.array(raw_mask_data)
            elif isinstance(raw_mask_data, np.ndarray):
                np_mask = raw_mask_data
            else:
                logger.warning(
                    f"Unsupported raw mask type '{type(raw_mask_data)}' for source '{source_desc}'."
                )
                return

            # 2. Generate Boolean Mask based on positive_value
            # Handle potential type issues (e.g., boolean arrays from mode '1' images)
            if np_mask.dtype == bool:
                # If mask is already boolean, positive_value determines if we take True or False
                if positive_value == 1:
                    binary_mask_bool = np_mask
                elif positive_value == 0:
                    binary_mask_bool = ~np_mask
                else:
                    logger.warning(
                        f"Unexpected positive_value {positive_value} for boolean mask "
                        f"from source '{source_desc}'. Expected 0 or 1."
                    )
                    binary_mask_bool = np.zeros_like(np_mask, dtype=bool)
            else:
                # For non-boolean masks, directly compare with positive_value
                try:
                    binary_mask_bool = np_mask == positive_value
                except TypeError as te:
                    logger.warning(
                        f"Type error during comparison for mask from source '{source_desc}' "
                        f"(dtype: {np_mask.dtype}, positive_value: {positive_value}): {te}"
                    )
                    return

            # Ensure it's boolean
            if binary_mask_bool.dtype != bool:
                logger.warning(
                    f"Internal error: generated mask is not boolean for source '{source_desc}'."
                )
                # Attempt conversion, but this indicates an issue
                binary_mask_bool = binary_mask_bool.astype(bool)

            self.binary_mask = binary_mask_bool

            # 3. Calculate Geometry if mask is not empty
            if not np.any(self.binary_mask):
                logger.debug(
                    f"No pixels matched positive_value={positive_value} in source '{source_desc}'."
                )
                self.pixel_area = 0
                # is_valid remains False
                return

            # Calculate pixel area
            self.pixel_area = int(np.sum(self.binary_mask))

            # Calculate bounding box
            rows, cols = np.where(self.binary_mask)
            if rows.size > 0 and cols.size > 0:  # Should always be true if pixel_area > 0
                y_min, y_max = int(rows.min()), int(rows.max())
                x_min, x_max = int(cols.min()), int(cols.max())
                self.bbox = (x_min, y_min, x_max, y_max)
                # Successfully parsed and calculated non-empty mask
                self.is_valid = True
            else:
                # This case should ideally not be reached if pixel_area > 0
                logger.warning(
                    f"Inconsistency: Mask from source '{source_desc}' has area {self.pixel_area} "
                    f"but failed to find valid bbox coordinates."
                )
                self.pixel_area = 0  # Reset area if bbox fails

        except Exception as e:
            logger.error(
                f"Error parsing mask from source '{source_desc}' "
                f"with positive_value={positive_value}: {e}",
                exc_info=True,
            )
            # Ensure state remains invalid on any unexpected error
            self.binary_mask = None
            self.pixel_area = None
            self.bbox = None
            self.is_valid = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert SegmMask instance to a serializable dictionary.

        Uses numpy.packbits to efficiently store binary mask as bytes.
        """
        result = {
            "source_info": self.source_info.dict(),  # Pydantic model has dict() method
            "is_valid": self.is_valid,
            "pixel_area": self.pixel_area,
            "bbox": self.bbox,
        }

        # Special handling for binary mask - convert to compressed bytes
        if self.binary_mask is not None:
            # Pack boolean array into bits to save space
            result["binary_mask_bytes"] = np.packbits(self.binary_mask).tobytes()
            result["binary_mask_shape"] = self.binary_mask.shape
        else:
            result["binary_mask_bytes"] = None
            result["binary_mask_shape"] = None

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmMask":
        """Create a SegmMask instance from a serialized dictionary.

        Reconstructs the binary mask from packed bytes representation.
        """
        # Create empty instance (skip normal initialization)
        instance = cls.__new__(cls)

        # Restore Pydantic model
        instance.source_info = InstanceMask.parse_obj(data["source_info"])
        instance.is_valid = data["is_valid"]
        instance.pixel_area = data["pixel_area"]
        instance.bbox = data["bbox"]

        # Restore binary mask from packed bytes
        if data["binary_mask_bytes"] is not None:
            packed_bytes = data["binary_mask_bytes"]
            shape = data["binary_mask_shape"]
            # Unpack bytes to boolean array
            unpacked = np.unpackbits(np.frombuffer(packed_bytes, dtype=np.uint8))
            # Reshape to original dimensions
            total_size = shape[0] * shape[1]
            instance.binary_mask = unpacked[:total_size].reshape(shape).astype(bool)
        else:
            instance.binary_mask = None

        return instance

    @property
    def width(self) -> Optional[int]:
        """Calculated width from the bounding box, if available."""
        if self.bbox:
            return self.bbox[2] - self.bbox[0] + 1
        return None

    @property
    def height(self) -> Optional[int]:
        """Calculated height from the bounding box, if available."""
        if self.bbox:
            return self.bbox[3] - self.bbox[1] + 1
        return None


class ClsSegment:
    """Represents a single semantic class concept within a sample.

    Links synonymous descriptive phrases to their corresponding parsed masks
    (both visible instances and full segmentations).

    Attributes:
        phrases: List of Phrase objects describing this single class concept.
                 These are treated as equivalent descriptions for this segment.
        type: The type associated with this segment (e.g., 'description').
        visible_masks: List of SegmMask objects for visible instances.
        full_masks: List of SegmMask objects for full segmentations.
    """

    phrases: List[Phrase]
    type: str
    visible_masks: List[SegmMask]
    full_masks: List[SegmMask]

    def __init__(
        self,
        phrases: List[Phrase],
        type: str,
        visible_masks: List[SegmMask],
        full_masks: List[SegmMask],
    ):
        """Initializes a ClsSegment instance."""
        self.phrases = phrases
        self.type = type
        self.visible_masks = visible_masks
        self.full_masks = full_masks

    def to_dict(self) -> Dict[str, Any]:
        """Convert ClsSegment instance to a serializable dictionary."""
        return {
            "phrases": [p.dict() for p in self.phrases],  # Pydantic models have dict()
            "type": self.type,
            "visible_masks": [mask.to_dict() for mask in self.visible_masks],
            "full_masks": [mask.to_dict() for mask in self.full_masks],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClsSegment":
        """Create a ClsSegment instance from a serialized dictionary."""
        return cls(
            phrases=[Phrase.parse_obj(p) for p in data["phrases"]],
            type=data["type"],
            visible_masks=[SegmMask.from_dict(m) for m in data["visible_masks"]],
            full_masks=[SegmMask.from_dict(m) for m in data["full_masks"]],
        )


class SegmSample:
    """Represents a single, fully processed sample from the dataset.

    Contains the main image and a list of its associated class segments.

    Attributes:
        id: The unique identifier for the sample.
        image: The main PIL image associated with the sample.
        segments: The list of ClsSegment objects within this sample.
    """

    id: str
    image: Image.Image
    segments: List[ClsSegment]

    def __init__(self, id: str, image: Image.Image, segments: List[ClsSegment]):
        """Initializes a SegmSample instance."""
        self.id = id
        self.image = image
        self.segments = segments

    def find_segment_by_prompt(self, prompt: str) -> Optional[ClsSegment]:
        """Finds the first segment where a phrase matches the given prompt.

        Args:
            prompt: The exact phrase text to search for.

        Returns:
            The matching ClsSegment object, or None if not found.
        """
        for segment in self.segments:
            for phrase in segment.phrases:
                if phrase.text == prompt:
                    return segment
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert SegmSample instance to a serializable dictionary.

        Converts PIL Image to bytes for serialization.
        """
        # Convert PIL image to bytes
        image_bytes = BytesIO()
        self.image.save(image_bytes, format="JPEG")

        return {
            "id": self.id,
            "image_bytes": image_bytes.getvalue(),
            "segments": [seg.to_dict() for seg in self.segments],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmSample":
        """Create a SegmSample instance from a serialized dictionary.

        Reconstructs PIL Image from bytes.
        """
        # Convert bytes back to PIL Image
        image = Image.open(BytesIO(data["image_bytes"]))
        # Ensure image is in memory to avoid file handle issues
        image.load()

        return cls(
            id=data["id"],
            image=image,
            segments=[ClsSegment.from_dict(s) for s in data["segments"]],
        )
