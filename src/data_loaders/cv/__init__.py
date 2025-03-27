"""
Computer Vision data loaders
"""

from src.data_loaders.cv.dummy import DummyDetectionDataset

# Import main classes
from src.data_loaders.cv.voc import ImprovedVOCDataset, PascalVOCDataset

__all__ = ["PascalVOCDataset", "ImprovedVOCDataset", "DummyDetectionDataset"]
