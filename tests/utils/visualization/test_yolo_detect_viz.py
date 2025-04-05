import argparse
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Module to test
from src.utils.visualization import yolo_detect_viz


class TestYoloDetectVizPaths(unittest.TestCase):
    def setUp(self):
        """Set up common test variables."""
        self.mock_args = argparse.Namespace(
            years="2012",
            tags="val",
            image_id=None,
            sample_count=-1,
            voc_root="/fake/voc/root",
            output_root=None,  # Test default output root
            output_subdir="detect/visual",
            percentiles=None,
            seed=42,
        )
        self.base_voc_root = Path("/fake/voc/root")
        self.output_base_dir = self.base_voc_root / "detect" / "visual"  # Default output
        self.tag_year = "val2012"
        self.image_dir = self.base_voc_root / "detect" / "images" / self.tag_year
        self.label_dir = self.base_voc_root / "detect" / "labels" / self.tag_year

    @patch("pathlib.Path.is_dir")
    def test_setup_paths_default_output(self, mock_is_dir):
        """Test _setup_paths uses voc_root as output_root by default."""
        mock_is_dir.return_value = True  # Simulate detect/images/ and detect/labels/ dirs exist
        voc_root_env = str(self.base_voc_root)

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            base_root, output_root, output_dir = yolo_detect_viz._setup_paths(
                self.mock_args, voc_root_env=None
            )  # Pass None, use args.voc_root

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
        expected_output_dir = Path(custom_output) / "detect" / "visual"

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            base_root, output_root, output_dir = yolo_detect_viz._setup_paths(
                self.mock_args, voc_root_env=None
            )

            self.assertEqual(base_root, self.base_voc_root)
            self.assertEqual(output_root, Path(custom_output))
            self.assertEqual(output_dir, expected_output_dir)
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("pathlib.Path.is_file")
    def test_get_target_image_list_single_mode_success(self, mock_is_file):
        """Test get_target_image_list in single image mode (files exist)."""
        image_id = "000001"
        self.mock_args.image_id = image_id
        mock_is_file.return_value = True  # Simulate image and label exist

        result = yolo_detect_viz.get_target_image_list(self.mock_args, self.base_voc_root)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (image_id, "2012", "val"))
        mock_is_file.assert_any_call()  # Checks both image and label paths
        self.assertEqual(mock_is_file.call_count, 2)

    @patch("pathlib.Path.is_file")
    def test_get_target_image_list_single_mode_image_missing(self, mock_is_file):
        """Test single image mode when image file is missing."""
        image_id = "000002"
        self.mock_args.image_id = image_id
        # First call (image check) returns False, second (label check) isn't reached/matters less
        mock_is_file.side_effect = [False, True]

        result = yolo_detect_viz.get_target_image_list(self.mock_args, self.base_voc_root)

        self.assertEqual(len(result), 0)
        mock_is_file.assert_called_once()  # Should stop after image check fails

    @patch("pathlib.Path.is_file")
    def test_get_target_image_list_single_mode_label_missing(self, mock_is_file):
        """Test single image mode when label file is missing."""
        image_id = "000003"
        self.mock_args.image_id = image_id
        # First call (image check) returns True, second (label check) returns False
        mock_is_file.side_effect = [True, False]

        result = yolo_detect_viz.get_target_image_list(self.mock_args, self.base_voc_root)

        self.assertEqual(len(result), 0)
        self.assertEqual(mock_is_file.call_count, 2)  # Checks both

    @patch("pathlib.Path.is_dir")
    @patch("pathlib.Path.glob")
    @patch("pathlib.Path.is_file", autospec=True)
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

        # Simulate corresponding images exist for 1 and 3, but not 2
        # Use a lambda function for the side effect
        mock_is_file.side_effect = lambda path_arg: (
            True
            if path_arg == self.image_dir / "000001.jpg"
            else False
            if path_arg == self.image_dir / "000002.jpg"
            else True
            if path_arg == self.image_dir / "000003.jpg"
            else False  # Default for other checks
        )

        result = yolo_detect_viz.get_target_image_list(self.mock_args, self.base_voc_root)

        self.assertEqual(len(result), 2)
        self.assertIn(("000001", "2012", "val"), result)
        self.assertIn(("000003", "2012", "val"), result)
        self.assertNotIn(("000002", "2012", "val"), result)  # Image was missing
        mock_is_dir.assert_called_once_with()
        mock_glob.assert_called_once_with("*.txt")
        # is_file called once per label file to check corresponding image
        self.assertEqual(mock_is_file.call_count, len(mock_label_files))

    @patch("pathlib.Path.is_dir", return_value=False)  # Label dir doesn't exist
    def test_get_target_image_list_batch_mode_label_dir_missing(self, mock_is_dir):
        """Test batch mode when label directory is missing."""
        result = yolo_detect_viz.get_target_image_list(self.mock_args, self.base_voc_root)
        self.assertEqual(len(result), 0)
        mock_is_dir.assert_called_once()

    @patch("pathlib.Path.is_dir", return_value=True)
    @patch("pathlib.Path.glob", return_value=[])  # No label files found
    def test_get_target_image_list_batch_mode_no_labels_found(self, mock_glob, mock_is_dir):
        """Test batch mode when label directory is empty."""
        result = yolo_detect_viz.get_target_image_list(self.mock_args, self.base_voc_root)
        self.assertEqual(len(result), 0)
        mock_is_dir.assert_called_once()
        mock_glob.assert_called_once()

    @patch("pathlib.Path.is_file", autospec=True)
    def test_is_file_patch_isolated(self, mock_is_file):
        """TEMPORARY: Test if patching Path.is_file works in isolation."""
        print("\nRunning isolated is_file patch test...")
        test_path_exists = Path("/fake/i_exist.jpg")
        test_path_missing = Path("/fake/i_do_not_exist.txt")

        # Define side effect: return True only for test_path_exists
        side_effect_func = lambda p: True if p == test_path_exists else False
        mock_is_file.side_effect = side_effect_func
        print(f"Side effect set to: {side_effect_func}")

        try:
            print(f"Calling is_file() on {test_path_exists}...")
            result_exists = test_path_exists.is_file()
            print(f"Result for existing path: {result_exists}")
            self.assertTrue(result_exists, "is_file should return True for mocked existing path")

            print(f"Calling is_file() on {test_path_missing}...")
            result_missing = test_path_missing.is_file()
            print(f"Result for missing path: {result_missing}")
            self.assertFalse(result_missing, "is_file should return False for mocked missing path")

            # Verify the mock was called with the correct arguments
            print("Checking mock calls...")
            mock_is_file.assert_any_call(test_path_exists)
            mock_is_file.assert_any_call(test_path_missing)
            print("Isolated test passed.")

        except Exception as e:
            print(f"ERROR in isolated test: {e}")
            raise  # Re-raise the exception to fail the test clearly

    # Although process_and_visualize_image is complex, we can test its path logic
    @patch("src.utils.visualization.yolo_detect_viz.cv2.imread")
    @patch("src.utils.visualization.yolo_detect_viz.parse_yolo_detection_label")
    @patch("pathlib.Path.mkdir")  # Mock mkdir for saving
    @patch("src.utils.visualization.yolo_detect_viz.cv2.imwrite")  # Mock save
    def test_process_image_path_construction(
        self, mock_imwrite, mock_mkdir, mock_parse, mock_imread
    ):
        """Verify paths used within process_and_visualize_image."""
        image_id = "000004"
        year = "2012"
        tag = "val"
        tag_year = f"{tag}{year}"

        # Simulate successful image load and label parse
        mock_imread.return_value = MagicMock(shape=(100, 100, 3))  # Fake image shape
        mock_parse.return_value = []  # No objects needed for path test

        # Call the function in save mode
        yolo_detect_viz.process_and_visualize_image(
            image_id=image_id,
            year=year,
            tag=tag,
            voc_root=self.base_voc_root,
            output_dir=self.output_base_dir,  # Pass <output_root>/detect/visual
            class_names=[],
            do_save=True,
            do_display=False,
        )

        # Check image read path
        expected_image_path = self.base_voc_root / "detect" / "images" / tag_year / f"{image_id}.jpg"
        mock_imread.assert_called_once_with(str(expected_image_path))

        # Check label parse path
        expected_label_path = self.base_voc_root / "detect" / "labels" / tag_year / f"{image_id}.txt"
        mock_parse.assert_called_once()
        call_args, _ = mock_parse.call_args
        self.assertEqual(call_args[0], expected_label_path)

        # Check output path
        expected_save_subdir = self.output_base_dir / tag_year
        expected_save_path = expected_save_subdir / f"{image_id}.png"
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_imwrite.assert_called_once()
        save_call_args, _ = mock_imwrite.call_args
        self.assertEqual(save_call_args[0], str(expected_save_path))


if __name__ == "__main__":
    unittest.main()
