"""
YOLOv11 detection and segmentation models.
"""

from src.models.ext.yolov11.predict_detect import predict_pipeline as predict_detect
from src.models.ext.yolov11.predict_segment import predict_pipeline as predict_segment

__all__ = ["predict_detect", "predict_segment"]
