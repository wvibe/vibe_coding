import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Module to test
from src.utils.visualization import yolo_segment_viz


class TestYoloSegmentVizPaths(unittest.TestCase):
    def setUp(self):
        """Set up common test variables."""
        self.mock_args = argparse.Namespace(
            years="2012",
            tags="val",
            image_id=None,
            sample_count=-1,
            voc_root="/fake/voc/root",
            output_root=None,  # Test default output root
            output_subdir="segment/visual",
            fill_polygons=False,
            alpha=0.3,
            percentiles=None,
            seed=42,
        )
        self.base_voc_root = Path("/fake/voc/root")
        self.output_base_dir = self.base_voc_root / "segment" / "visual"  # Default output
        self.tag_year = "val2012"
        self.image_dir = self.base_voc_root / "segment" / "images" / self.tag_year
        self.label_dir = self.base_voc_root / "segment" / "labels" / self.tag_year

    @patch("pathlib.Path.is_dir")
    def test_setup_paths_default_output(self, mock_is_dir):
        """Test _setup_paths uses voc_root as output_root by default."""
        mock_is_dir.return_value = True  # Simulate segment/images/ and segment/labels/ dirs exist
        voc_root_env = None  # Use args.voc_root

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            base_root, output_root, output_dir = yolo_segment_viz._setup_paths(
                self.mock_args, voc_root_env
            )

            self.assertEqual(base_root, self.base_voc_root)
            self.assertEqual(output_root, self.base_voc_root)  # Should default to base
            self.assertEqual(output_dir, self.output_base_dir)
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("pathlib.Path.is_dir")
    def test_setup_paths_custom_output(self, mock_is_dir):
        """Test _setup_paths uses specified output_root."""
        mock_is_dir.return_value = True
        custom_output = "/fake/output"
        self.mock_args.output_root = custom_output
        expected_output_dir = Path(custom_output) / "segment" / "visual"

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            base_root, output_root, output_dir = yolo_segment_viz._setup_paths(
                self.mock_args, voc_root_env=None
            )

            self.assertEqual(base_root, self.base_voc_root)
            self.assertEqual(output_root, Path(custom_output))
            self.assertEqual(output_dir, expected_output_dir)
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("pathlib.Path.is_dir")
    def test_setup_paths_missing_dirs(self, mock_is_dir):
        """Test _setup_paths warns when expected directories are missing."""
        # First check is for segment/images, second for segment/labels
        mock_is_dir.side_effect = [False, False]

        with patch("src.utils.visualization.yolo_segment_viz.logger.warning") as mock_warning:
            with patch("pathlib.Path.mkdir"):
                yolo_segment_viz._setup_paths(self.mock_args, voc_root_env=None)

                # Should warn about missing directories
                mock_warning.assert_called_once()
                self.assertIn("does not contain expected", mock_warning.call_args[0][0])


class TestYoloSegmentVizImageList(unittest.TestCase):
    def setUp(self):
        """Set up common test variables."""
        self.mock_args = argparse.Namespace(
            years="2012",
            tags="val",
            image_id=None,
            sample_count=-1,
            voc_root="/fake/voc/root",
            output_root=None,
            output_subdir="segment/visual",
            fill_polygons=False,
            alpha=0.3,
            percentiles=None,
            seed=42,
        )
        self.base_voc_root = Path("/fake/voc/root")
        self.tag_year = "val2012"
        self.image_dir = self.base_voc_root / "segment" / "images" / self.tag_year
        self.label_dir = self.base_voc_root / "segment" / "labels" / self.tag_year

    @patch("pathlib.Path.is_file")
    def test_get_target_image_list_single_mode_success(self, mock_is_file):
        """Test get_target_image_list in single image mode (files exist)."""
        image_id = "000001"
        self.mock_args.image_id = image_id
        mock_is_file.return_value = True  # Simulate image and label exist

        result = yolo_segment_viz.get_target_image_list(self.mock_args, self.base_voc_root)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (image_id, "2012", "val"))
        mock_is_file.assert_any_call()  # Checks both image and label paths
        self.assertEqual(mock_is_file.call_count, 2)

    @patch("pathlib.Path.is_file")
    def test_get_target_image_list_single_mode_image_missing(self, mock_is_file):
        """Test single image mode when image file is missing."""
        image_id = "000002"
        self.mock_args.image_id = image_id
        # First call (image check) returns False, second (label check) isn't reached
        mock_is_file.side_effect = [False, True]

        result = yolo_segment_viz.get_target_image_list(self.mock_args, self.base_voc_root)

        self.assertEqual(len(result), 0)
        mock_is_file.assert_called_once()  # Should stop after image check fails

    @patch("pathlib.Path.is_dir")
    @patch("pathlib.Path.glob")
    @patch("pathlib.Path.is_file")
    def test_get_target_image_list_batch_mode(self, mock_is_file, mock_glob, mock_is_dir):
        """Test get_target_image_list in batch mode."""
        # Simulate label dir exists
        mock_is_dir.return_value = True

        # Simulate label files found by glob
        mock_label_files = [
            self.label_dir / "000001.txt",
            self.label_dir / "000002.txt",
            self.label_dir / "000003.txt",
        ]
        mock_glob.return_value = mock_label_files

        # Set up a sequence of return values for is_file
        # The sequence will be [True, False, True] corresponding to
        # the file checks for 000001.jpg, 000002.jpg, 000003.jpg
        mock_is_file.side_effect = [True, False, True]

        result = yolo_segment_viz.get_target_image_list(self.mock_args, self.base_voc_root)

        # Should only include image IDs with existing image files (1 and 3)
        self.assertEqual(len(result), 2)
        self.assertIn(("000001", "2012", "val"), result)
        self.assertIn(("000003", "2012", "val"), result)
        self.assertNotIn(("000002", "2012", "val"), result)  # Image was missing

        mock_is_dir.assert_called_once_with()
        mock_glob.assert_called_once_with("*.txt")
        # is_file called once per label file to check corresponding image
        self.assertEqual(mock_is_file.call_count, len(mock_label_files))

    @patch("pathlib.Path.is_dir")
    @patch("pathlib.Path.glob")
    @patch("random.sample")
    def test_get_target_image_list_with_sampling(self, mock_sample, mock_glob, mock_is_dir):
        """Test get_target_image_list with sampling enabled."""
        # Setup
        self.mock_args.sample_count = 1  # Request only 1 sample
        mock_is_dir.return_value = True

        # Create mock files
        mock_label_files = [
            self.label_dir / "000001.txt",
            self.label_dir / "000002.txt",
        ]
        mock_glob.return_value = mock_label_files

        # Mock sampling function
        sampled_result = [("000001", "2012", "val")]
        mock_sample.return_value = sampled_result

        # Mock is_file to return True for all files
        with patch("pathlib.Path.is_file", return_value=True):
            result = yolo_segment_viz.get_target_image_list(self.mock_args, self.base_voc_root)

            self.assertEqual(result, sampled_result)
            mock_sample.assert_called_once()
            # Check the sample call received the right arguments
            sample_args, _ = mock_sample.call_args
            self.assertEqual(len(sample_args[0]), 2)  # All found IDs
            self.assertEqual(sample_args[1], 1)  # Sample count


class TestYoloSegmentVizParsing(unittest.TestCase):
    def setUp(self):
        self.image_width = 100
        self.image_height = 100
        self.class_names = ["background", "person", "bicycle", "car"]
        self.label_path = Path("/fake/path/000001.txt")

    def test_parse_valid_polygons(self):
        """Test parsing valid YOLO segmentation labels."""
        with patch(
            "builtins.open",
            unittest.mock.mock_open(
                read_data="1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"  # person polygon
                "2 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6"  # bicycle polygon
            ),
        ):
            objects_list = yolo_segment_viz.parse_yolo_segmentation_label(
                self.label_path, self.image_width, self.image_height, self.class_names
            )

            self.assertEqual(len(objects_list), 2)

            # Check first object (person)
            self.assertEqual(objects_list[0]["name"], "person")
            self.assertEqual(len(objects_list[0]["points"]), 4)  # 4 points
            # Check point values (converted from normalized to pixel coords)
            self.assertEqual(objects_list[0]["points"][0], (10, 10))  # 0.1 * 100, 0.1 * 100

            # Check second object (bicycle)
            self.assertEqual(objects_list[1]["name"], "bicycle")
            self.assertEqual(len(objects_list[1]["points"]), 4)

    def test_parse_invalid_coords(self):
        """Test parsing labels with invalid coordinates (outside 0-1 range)."""
        with patch(
            "builtins.open",
            unittest.mock.mock_open(
                read_data="1 -0.1 0.1 1.2 0.1 0.2 0.2 0.1 0.2\n"  # Invalid coords (-0.1, 1.2)
            ),
        ):
            with patch("src.utils.visualization.yolo_segment_viz.logger.warning") as mock_warning:
                objects_list = yolo_segment_viz.parse_yolo_segmentation_label(
                    self.label_path, self.image_width, self.image_height, self.class_names
                )

                # Should warn about invalid coordinates
                self.assertTrue(
                    any(
                        "Invalid normalized coordinates" in args[0]
                        for args, _ in mock_warning.call_args_list
                    )
                )

                # If the implementation filters out objects with invalid coordinates completely
                self.assertEqual(len(objects_list), 0)

    def test_parse_invalid_class_id(self):
        """Test parsing labels with invalid class ID."""
        with patch(
            "builtins.open",
            unittest.mock.mock_open(
                read_data="10 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"  # Invalid class ID (10)
            ),
        ):
            with patch("src.utils.visualization.yolo_segment_viz.logger.warning") as mock_warning:
                objects_list = yolo_segment_viz.parse_yolo_segmentation_label(
                    self.label_path, self.image_width, self.image_height, self.class_names
                )

                # Should warn about invalid class ID
                mock_warning.assert_any_call(
                    f"Invalid class ID 10 in {self.label_path}. Using 'Unknown'."
                )

                # Should still create the object but with "Unknown" class
                self.assertEqual(len(objects_list), 1)
                self.assertEqual(objects_list[0]["name"], "Unknown")

    def test_parse_too_few_points(self):
        """Test parsing labels with too few points (less than 3)."""
        with patch(
            "builtins.open",
            unittest.mock.mock_open(
                read_data="1 0.1 0.1 0.2 0.2\n"  # Only 2 points (need at least 3)
            ),
        ):
            with patch("src.utils.visualization.yolo_segment_viz.logger.warning") as mock_warning:
                objects_list = yolo_segment_viz.parse_yolo_segmentation_label(
                    self.label_path, self.image_width, self.image_height, self.class_names
                )

                # Match the exact phrase from the implementation
                self.assertTrue(
                    any(
                        "Skipping invalid line" in args[0] or
                        "Skipping polygon with < 3 points" in args[0] or
                        "Too few coordinates" in args[0]
                        for args, _ in mock_warning.call_args_list
                    ),
                    "Expected warning about insufficient points not found",
                )

                # Should not include objects with too few points
                self.assertEqual(len(objects_list), 0)


class TestYoloSegmentVizProcessing(unittest.TestCase):
    def setUp(self):
        self.image_id = "000001"
        self.year = "2012"
        self.tag = "val"
        self.tag_year = f"{self.tag}{self.year}"
        self.voc_root = Path("/fake/voc/root")
        self.output_dir = self.voc_root / "segment" / "visual"
        self.class_names = ["background", "person", "bicycle", "car"]

        # Paths used for tests
        self.image_path = (
            self.voc_root / "segment" / "images" / self.tag_year / f"{self.image_id}.jpg"
        )
        self.label_path = (
            self.voc_root / "segment" / "labels" / self.tag_year / f"{self.image_id}.txt"
        )
        self.expected_save_path = self.output_dir / self.tag_year / f"{self.image_id}.png"

    @patch("src.utils.visualization.yolo_segment_viz.cv2.imread")
    @patch("src.utils.visualization.yolo_segment_viz.parse_yolo_segmentation_label")
    @patch("pathlib.Path.mkdir")
    @patch("src.utils.visualization.yolo_segment_viz.cv2.imwrite")
    def test_process_image_path_construction(
        self, mock_imwrite, mock_mkdir, mock_parse, mock_imread
    ):
        """Test the path construction in process_and_visualize_image."""
        # Simulate successful image load and parse
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)  # Fake image
        mock_parse.return_value = []  # No objects needed for path test

        # Call function in save mode
        success, _, _, _, save_success, _ = yolo_segment_viz.process_and_visualize_image(
            image_id=self.image_id,
            year=self.year,
            tag=self.tag,
            voc_root=self.voc_root,
            output_dir=self.output_dir,
            class_names=self.class_names,
            do_save=True,
            do_display=False,
            fill_polygons=False,
            alpha=0.3,
        )

        # Verify paths
        mock_imread.assert_called_once_with(str(self.image_path))
        mock_parse.assert_called_once()
        # Check label path in parse function call
        label_path_arg = mock_parse.call_args[0][0]
        self.assertEqual(label_path_arg, self.label_path)

        # Check output path and saving
        mock_mkdir.assert_called_once()
        save_path_arg = mock_imwrite.call_args[0][0]
        self.assertEqual(save_path_arg, str(self.expected_save_path))

        # Check success flag
        self.assertTrue(success)
        self.assertTrue(save_success)

    @patch("src.utils.visualization.yolo_segment_viz.cv2.imread")
    @patch("src.utils.visualization.yolo_segment_viz.parse_yolo_segmentation_label")
    @patch("src.utils.visualization.yolo_segment_viz.draw_polygon")
    @patch("src.utils.visualization.yolo_segment_viz.overlay_mask")
    def test_process_image_drawing_outline(self, mock_overlay, mock_draw, mock_parse, mock_imread):
        """Test polygon drawing with fill_polygons=False."""
        # Setup mocks
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create a sample polygon object
        mock_objects = [
            {
                "name": "person",
                "points": [(10, 10), (20, 10), (20, 20), (10, 20)],
                "points_array": np.array([(10, 10), (20, 10), (20, 20), (10, 20)]),
            }
        ]
        mock_parse.return_value = mock_objects

        # Call function with fill_polygons=False
        with patch("src.utils.visualization.yolo_segment_viz.cv2.imwrite"):
            with patch("pathlib.Path.mkdir"):
                yolo_segment_viz.process_and_visualize_image(
                    image_id=self.image_id,
                    year=self.year,
                    tag=self.tag,
                    voc_root=self.voc_root,
                    output_dir=self.output_dir,
                    class_names=self.class_names,
                    do_save=True,
                    do_display=False,
                    fill_polygons=False,  # Draw outline only
                    alpha=0.3,
                )

        # Should call draw_polygon but not overlay_mask
        mock_draw.assert_called_once()
        mock_overlay.assert_not_called()

    @patch("src.utils.visualization.yolo_segment_viz.cv2.imread")
    @patch("src.utils.visualization.yolo_segment_viz.parse_yolo_segmentation_label")
    @patch("src.utils.visualization.yolo_segment_viz.draw_polygon")
    @patch("src.utils.visualization.yolo_segment_viz.overlay_mask")
    @patch("src.utils.visualization.yolo_segment_viz.cv2.fillPoly")
    def test_process_image_drawing_filled(
        self, mock_fill, mock_overlay, mock_draw, mock_parse, mock_imread
    ):
        """Test polygon drawing with fill_polygons=True."""
        # Setup mocks
        mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create a sample polygon object
        mock_objects = [
            {
                "name": "person",
                "points": [(10, 10), (20, 10), (20, 20), (10, 20)],
                "points_array": np.array([(10, 10), (20, 10), (20, 20), (10, 20)]),
            }
        ]
        mock_parse.return_value = mock_objects

        # Call function with fill_polygons=True
        with patch("src.utils.visualization.yolo_segment_viz.cv2.imwrite"):
            with patch("pathlib.Path.mkdir"):
                yolo_segment_viz.process_and_visualize_image(
                    image_id=self.image_id,
                    year=self.year,
                    tag=self.tag,
                    voc_root=self.voc_root,
                    output_dir=self.output_dir,
                    class_names=self.class_names,
                    do_save=True,
                    do_display=False,
                    fill_polygons=True,  # Fill polygons
                    alpha=0.3,
                )

        # Should call fillPoly and overlay_mask, but not draw_polygon
        mock_fill.assert_called_once()
        mock_overlay.assert_called_once()
        mock_draw.assert_not_called()


class TestYoloSegmentVizStatistics(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(
            percentiles="0.25,0.5,0.75",
            sample_count=10,
        )

        # Sample statistics
        self.stats = {
            "polygons_per_image": [1, 2, 3, 4, 5],
            "classes_per_image": [1, 1, 2, 2, 3],
            "points_per_polygon": [4, 5, 6, 7, 8, 9],
            "images_processed": 5,
            "label_read_success": 5,
            "images_saved": 5,
            "images_displayed": 0,
        }

    @patch("src.utils.visualization.yolo_segment_viz.logger.info")
    @patch("numpy.percentile")
    def test_report_statistics_with_percentiles(self, mock_percentile, mock_info):
        """Test statistics reporting with percentiles."""
        # Mock the percentile calculations
        mock_percentile.side_effect = [
            np.array([2, 3, 4]),  # Polygon count percentiles
            np.array([1, 2, 2]),  # Class count percentiles
            np.array([5, 6.5, 8]),  # Points per polygon percentiles
        ]

        # Call the function
        yolo_segment_viz._report_statistics(self.args, self.stats, 5)

        # Check percentile calculations
        self.assertEqual(mock_percentile.call_count, 3)

        # Verify logging
        self.assertTrue(
            any("Percentiles requested" in call[0][0] for call in mock_info.call_args_list)
        )
        self.assertTrue(
            any("Polygon count percentiles" in call[0][0] for call in mock_info.call_args_list)
        )

    @patch("src.utils.visualization.yolo_segment_viz.logger.info")
    def test_report_statistics_averages(self, mock_info):
        """Test statistics reporting with averages."""
        # Use a version without percentiles
        args_no_percentiles = argparse.Namespace(percentiles=None)

        # Call the function
        yolo_segment_viz._report_statistics(args_no_percentiles, self.stats, 5)

        # Should report averages instead of percentiles
        self.assertTrue(
            any("Average polygons per image" in call[0][0] for call in mock_info.call_args_list)
        )
        self.assertTrue(
            any(
                "Average unique classes per image" in call[0][0]
                for call in mock_info.call_args_list
            )
        )
        self.assertTrue(
            any("Average points per polygon" in call[0][0] for call in mock_info.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
