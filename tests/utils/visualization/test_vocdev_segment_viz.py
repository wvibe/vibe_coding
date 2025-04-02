import numpy as np

from utils.data_converter.voc2yolo_utils import VOC_CLASSES
from utils.visualization import vocdev_segment_viz


def test_generate_instance_label_valid_class():
    # Test with a valid class mask that should map correctly
    instance_id = 1
    instance_mask = np.array([[1, 1], [1, 0]], dtype=np.uint8)
    class_mask = np.array([[7, 7], [7, 0]], dtype=np.uint8)
    expected_label = f"{VOC_CLASSES[7 - 1]}.{instance_id}"
    label = vocdev_segment_viz.generate_instance_label(
        instance_id, instance_mask, class_mask, image_id="test_img"
    )
    assert label == expected_label


def test_generate_instance_label_missing_class_mask():
    # If class mask is None, should default to 'Unknown'
    instance_id = 2
    instance_mask = np.array([[2, 0], [0, 2]], dtype=np.uint8)
    label = vocdev_segment_viz.generate_instance_label(
        instance_id, instance_mask, None, image_id="test_img"
    )
    assert label == f"Unknown.{instance_id}"


def test_generate_instance_label_invalid_class_id():
    # If the class mask contains an invalid class ID, it should return 'Unknown'
    instance_id = 3
    instance_mask = np.full((2, 2), 3, dtype=np.uint8)
    class_mask = np.full((2, 2), 99, dtype=np.uint8)  # Assuming 99 is invalid
    label = vocdev_segment_viz.generate_instance_label(
        instance_id, instance_mask, class_mask, image_id="test_img"
    )
    assert label == f"Unknown.{instance_id}"


def test_generate_instance_label_mixed_classes():
    # When there is a mix, the majority valid class should be chosen
    instance_id = 4
    instance_mask = np.array([[4, 4], [4, 0]], dtype=np.uint8)
    class_mask = np.array([[15, 15], [99, 0]], dtype=np.uint8)
    expected_label = f"{VOC_CLASSES[15 - 1]}.{instance_id}"
    label = vocdev_segment_viz.generate_instance_label(
        instance_id, instance_mask, class_mask, image_id="test_img"
    )
    assert label == expected_label
