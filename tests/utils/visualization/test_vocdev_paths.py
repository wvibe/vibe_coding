import unittest
from argparse import Namespace
from pathlib import Path
from typing import Optional
from unittest.mock import call, patch

# Also import utils directly if needed for calculating expected paths
from src.utils.data_converter import voc2yolo_utils
from src.utils.visualization.vocdev_detect_viz import (
    _get_batch_image_ids as get_batch_ids_detect,
)

# Functions under test - import the helper function directly
from src.utils.visualization.vocdev_detect_viz import (
    _setup_paths as setup_paths_detect,
)
from src.utils.visualization.vocdev_detect_viz import (
    get_target_image_list as get_target_list_detect,
)
from src.utils.visualization.vocdev_segment_viz import (
    _get_batch_image_ids as get_batch_ids_segment,
)
from src.utils.visualization.vocdev_segment_viz import (
    _setup_paths as setup_paths_segment,
)
from src.utils.visualization.vocdev_segment_viz import (
    get_target_image_list as get_target_list_segment,
)


class TestVocdevPathSetup(unittest.TestCase):
    def setUp(self):
        # Common mock args setup
        self.mock_args_base = Namespace(
            voc_root=None,
            output_root=None,
            output_subdir="visual_test",
            image_id=None,  # Ensure image_id is None for batch tests
            year="2007",
            tag="train",  # Add default year/tag for convenience
        )

    def _run_path_setup_test(
        self,
        setup_func,
        args: Namespace,
        mock_getenv_return: Optional[str],
        mock_exists_return: bool,
        expected_base_root: Optional[Path],
        expected_devkit_root: Optional[Path],
        expected_output_dir: Optional[Path],
        expect_exception: Optional[type] = None,
    ):
        """Helper to run a test case for _setup_paths with mocks."""
        # Use autospec=True for Path.exists to better mimic real behavior if needed
        with (
            patch("os.getenv") as mock_getenv,
            patch("pathlib.Path.exists", autospec=True) as mock_exists,
            patch("pathlib.Path.mkdir"),
        ):
            mock_getenv.return_value = mock_getenv_return

            # FIX 1: Ensure correct exists_side_effect signature and logic
            def exists_side_effect(instance_path):  # instance_path is the Path object
                if instance_path.name == "VOCdevkit":
                    return mock_exists_return
                return True  # Assume other paths exist

            # Assign the corrected side effect function
            mock_exists.side_effect = exists_side_effect

            if expect_exception:
                with self.assertRaises(expect_exception):
                    # Pass the env value correctly here
                    setup_func(args, mock_getenv_return)
            else:
                base_root, devkit_root, _, output_dir = setup_func(args, mock_getenv_return)
                self.assertEqual(base_root, expected_base_root)
                self.assertEqual(devkit_root, expected_devkit_root)
                self.assertEqual(output_dir, expected_output_dir)

    # --- Tests for setup_paths_detect ---

    def test_detect_voc_root_arg_no_output_arg(self):
        args = self.mock_args_base
        args.voc_root = "/test/voc"
        args.output_subdir = "visual_detect"
        self._run_path_setup_test(
            setup_paths_detect,
            args,
            mock_getenv_return=None,
            mock_exists_return=True,
            expected_base_root=Path("/test/voc"),
            expected_devkit_root=Path("/test/voc/VOCdevkit"),
            expected_output_dir=Path("/test/voc/VOCdevkit/visual_detect"),
        )

    def test_detect_env_var_no_output_arg(self):
        args = self.mock_args_base
        args.output_subdir = "visual_detect"
        self._run_path_setup_test(
            setup_paths_detect,
            args,
            mock_getenv_return="/env/voc",
            mock_exists_return=True,
            expected_base_root=Path("/env/voc"),
            expected_devkit_root=Path("/env/voc/VOCdevkit"),
            expected_output_dir=Path("/env/voc/VOCdevkit/visual_detect"),
        )

    def test_detect_voc_root_arg_with_output_arg(self):
        args = self.mock_args_base
        args.voc_root = "/test/voc"
        args.output_root = "/custom/output"
        args.output_subdir = "visual_detect"
        self._run_path_setup_test(
            setup_paths_detect,
            args,
            mock_getenv_return=None,
            mock_exists_return=True,
            expected_base_root=Path("/test/voc"),
            expected_devkit_root=Path("/test/voc/VOCdevkit"),
            expected_output_dir=Path("/custom/output/visual_detect"),
        )

    def test_detect_env_var_with_output_arg(self):
        args = self.mock_args_base
        args.output_root = "/another/output"
        args.output_subdir = "visual_detect"
        self._run_path_setup_test(
            setup_paths_detect,
            args,
            mock_getenv_return="/env/voc",
            mock_exists_return=True,
            expected_base_root=Path("/env/voc"),
            expected_devkit_root=Path("/env/voc/VOCdevkit"),
            expected_output_dir=Path("/another/output/visual_detect"),
        )

    def test_detect_error_no_voc_root(self):
        args = self.mock_args_base
        self._run_path_setup_test(
            setup_paths_detect,
            args,
            mock_getenv_return=None,
            mock_exists_return=True,  # Doesn't matter
            expected_base_root=None,
            expected_devkit_root=None,
            expected_output_dir=None,
            expect_exception=ValueError,
        )

    def test_detect_error_devkit_not_found(self):
        args = self.mock_args_base
        args.voc_root = "/test/voc"
        self._run_path_setup_test(
            setup_paths_detect,
            args,
            mock_getenv_return=None,
            mock_exists_return=False,  # Mock devkit not found
            expected_base_root=None,
            expected_devkit_root=None,
            expected_output_dir=None,
            expect_exception=FileNotFoundError,
        )

    # --- Tests for setup_paths_segment (identical logic for now) ---

    def test_segment_voc_root_arg_no_output_arg(self):
        args = self.mock_args_base
        args.voc_root = "/test/voc_seg"
        args.output_subdir = "visual_segment"
        self._run_path_setup_test(
            setup_paths_segment,
            args,
            mock_getenv_return=None,
            mock_exists_return=True,
            expected_base_root=Path("/test/voc_seg"),
            expected_devkit_root=Path("/test/voc_seg/VOCdevkit"),
            expected_output_dir=Path("/test/voc_seg/VOCdevkit/visual_segment"),
        )

    def test_segment_env_var_no_output_arg(self):
        args = self.mock_args_base
        args.output_subdir = "visual_segment"
        self._run_path_setup_test(
            setup_paths_segment,
            args,
            mock_getenv_return="/env/voc_seg",
            mock_exists_return=True,
            expected_base_root=Path("/env/voc_seg"),
            expected_devkit_root=Path("/env/voc_seg/VOCdevkit"),
            expected_output_dir=Path("/env/voc_seg/VOCdevkit/visual_segment"),
        )

    def test_segment_voc_root_arg_with_output_arg(self):
        args = self.mock_args_base
        args.voc_root = "/test/voc_seg"
        args.output_root = "/custom/output_seg"
        args.output_subdir = "visual_segment"
        self._run_path_setup_test(
            setup_paths_segment,
            args,
            mock_getenv_return=None,
            mock_exists_return=True,
            expected_base_root=Path("/test/voc_seg"),
            expected_devkit_root=Path("/test/voc_seg/VOCdevkit"),
            expected_output_dir=Path("/custom/output_seg/visual_segment"),
        )

    def test_segment_env_var_with_output_arg(self):
        args = self.mock_args_base
        args.output_root = "/another/output_seg"
        args.output_subdir = "visual_segment"
        self._run_path_setup_test(
            setup_paths_segment,
            args,
            mock_getenv_return="/env/voc_seg",
            mock_exists_return=True,
            expected_base_root=Path("/env/voc_seg"),
            expected_devkit_root=Path("/env/voc_seg/VOCdevkit"),
            expected_output_dir=Path("/another/output_seg/visual_segment"),
        )

    def test_segment_error_no_voc_root(self):
        args = self.mock_args_base
        self._run_path_setup_test(
            setup_paths_segment,
            args,
            mock_getenv_return=None,
            mock_exists_return=True,  # Doesn't matter
            expected_base_root=None,
            expected_devkit_root=None,
            expected_output_dir=None,
            expect_exception=ValueError,
        )

    def test_segment_error_devkit_not_found(self):
        args = self.mock_args_base
        args.voc_root = "/test/voc_seg"
        self._run_path_setup_test(
            setup_paths_segment,
            args,
            mock_getenv_return=None,
            mock_exists_return=False,  # Mock devkit not found
            expected_base_root=None,
            expected_devkit_root=None,
            expected_output_dir=None,
            expect_exception=FileNotFoundError,
        )

    # --- Tests for ImageSet Path Logic (_get_batch_image_ids) ---

    # Patch read_image_ids where it's looked up (in the viz script)
    # AND patch Path.exists to prevent internal FileNotFoundError in real function
    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.utils.visualization.vocdev_detect_viz.read_image_ids")
    def test_detect_batch_uses_main_imageset(self, mock_read_ids, mock_exists):
        # Arrange
        mock_read_ids.return_value = ["000001"]
        base_voc_root = Path("/fake/voc")
        years = ["2007"]
        tags = ["train"]

        # Calculate the expected path (used for assertion)
        expected_voc_dir = voc2yolo_utils.get_voc_dir(base_voc_root, years[0])
        expected_imageset_path = voc2yolo_utils.get_image_set_path(
            expected_voc_dir, set_type="detect", tag=tags[0]
        )

        # Act: Call the function being tested
        get_batch_ids_detect(years, tags, base_voc_root)

        # Assert: Check that our mock read_image_ids was called with the correct path
        mock_read_ids.assert_called_once_with(expected_imageset_path)
        # Verify exists *could* have been called on the path (optional)
        # mock_exists.assert_any_call(expected_imageset_path)

    # Patch read_image_ids where it's looked up (in the viz script)
    # AND patch Path.exists to prevent internal FileNotFoundError in real function
    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.utils.visualization.vocdev_segment_viz.read_image_ids")
    def test_segment_batch_uses_segmentation_imageset(self, mock_read_ids, mock_exists):
        # Arrange
        mock_read_ids.return_value = ["000002"]
        base_voc_root = Path("/fake/voc2")
        years = ["2012"]
        tags = ["val"]

        # Calculate the expected path (used for assertion)
        expected_voc_dir = voc2yolo_utils.get_voc_dir(base_voc_root, years[0])
        expected_imageset_path = voc2yolo_utils.get_image_set_path(
            expected_voc_dir, set_type="segment", tag=tags[0]
        )

        # Act: Call the function being tested
        get_batch_ids_segment(years, tags, base_voc_root)

        # Assert: Check that our mock read_image_ids was called with the correct path
        mock_read_ids.assert_called_once_with(expected_imageset_path)
        # Verify exists *could* have been called on the path (optional)
        # mock_exists.assert_any_call(expected_imageset_path)

    # --- Tests for Single Image Annotation/Mask Path Logic (get_target_image_list) ---

    @patch("pathlib.Path.exists", autospec=True)
    def test_detect_single_image_checks_annotation_path(self, mock_exists):
        # Arrange
        mock_exists.return_value = True
        args = Namespace(year="2007", tag="train", image_id="000123", sample_count=-1)
        base_voc_root = Path("/base/voc_detect")
        voc_devkit_dir = base_voc_root / "VOCdevkit"

        # Calculate the expected paths *without* mocking the get_* functions
        expected_year_voc_dir = voc2yolo_utils.get_voc_dir(base_voc_root, args.year)
        expected_img_path = voc2yolo_utils.get_image_path(expected_year_voc_dir, args.image_id)
        expected_anno_path = voc2yolo_utils.get_annotation_path(
            expected_year_voc_dir, args.image_id
        )

        # Act
        # Need to use the imported function name from the module
        get_target_list_detect(args, base_voc_root, voc_devkit_dir)

        # Assert
        # Check that Path.exists was called on the specific expected Path objects
        mock_exists.assert_has_calls(
            [
                call(expected_img_path),  # Check exists called on the expected image path
                call(expected_anno_path),  # Check exists called on the expected annotation path
            ],
            any_order=True,
        )

    @patch("pathlib.Path.exists", autospec=True)
    def test_segment_single_image_checks_mask_path(self, mock_exists):
        # Arrange
        mock_exists.return_value = True
        args = Namespace(year="2012", tag="val", image_id="000456", sample_count=-1)
        base_voc_root = Path("/base/voc_segment")
        voc_devkit_dir = base_voc_root / "VOCdevkit"

        # Calculate the expected paths
        expected_year_voc_dir = voc2yolo_utils.get_voc_dir(base_voc_root, args.year)
        expected_img_path = voc2yolo_utils.get_image_path(expected_year_voc_dir, args.image_id)
        expected_mask_path = voc2yolo_utils.get_segmentation_mask_path(
            expected_year_voc_dir, args.image_id
        )

        # Act
        # Need to use the imported function name from the module
        get_target_list_segment(args, base_voc_root, voc_devkit_dir)

        # Assert
        mock_exists.assert_has_calls(
            [call(expected_img_path), call(expected_mask_path)], any_order=True
        )


if __name__ == "__main__":
    unittest.main()
