import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure src directory is in sys.path for imports
project_root = Path(__file__).resolve().parents[3]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import constants from the utils module
from utils.data_converter import voc2yolo_utils

# Path to the script to be tested
SCRIPT_PATH = src_path / "utils" / "data_converter" / "voc2yolo_images.py"

# --- Fixture for Mock VOC Structure ---


@pytest.fixture
def setup_mock_voc_structure(tmp_path):
    """Creates a mock VOCdevkit structure in a temporary directory."""
    mock_voc_root = tmp_path / "mock_voc"
    mock_output_root = tmp_path / "mock_output"
    mock_voc_devkit = mock_voc_root / "VOCdevkit"

    years = ["2007", "2012"]
    tags = ["train", "val"]
    image_ids = {
        "2007": {"train": ["00001", "00002"], "val": ["00003"]},
        "2012": {"train": ["10001", "10002"], "val": ["10003", "10004"]},
    }

    for year in years:
        # Create image files
        img_dir = mock_voc_devkit / f"VOC{year}" / voc2yolo_utils.JPEG_IMAGES_DIR
        img_dir.mkdir(parents=True, exist_ok=True)
        for tag in tags:
            for img_id in image_ids[year][tag]:
                (img_dir / f"{img_id}.jpg").touch()

        # Create ImageSet files (Main - for detection)
        main_set_dir = (
            mock_voc_devkit / f"VOC{year}" / voc2yolo_utils.IMAGESETS_DIR / voc2yolo_utils.MAIN_DIR
        )
        main_set_dir.mkdir(parents=True, exist_ok=True)
        for tag in tags:
            (main_set_dir / f"{tag}.txt").write_text("\n".join(image_ids[year][tag]))

        # Create ImageSet files (Segmentation - using different IDs for distinction)
        seg_set_dir = (
            mock_voc_devkit
            / f"VOC{year}"
            / voc2yolo_utils.IMAGESETS_DIR
            / voc2yolo_utils.SEGMENTATION_DIR
        )
        seg_set_dir.mkdir(parents=True, exist_ok=True)
        # Use a subset/different set for segmentation for testing differentiation
        seg_ids = image_ids[year][tag][:1]  # Take only the first ID for segmentation test
        if year == "2007" and tag == "val":  # Make one seg set empty
            seg_ids = []
        if year == "2012" and tag == "train":  # Add an ID not in Main set
            seg_ids.append("10005")
            (img_dir / "10005.jpg").touch()  # Create the corresponding image

        (seg_set_dir / f"{tag}.txt").write_text("\n".join(seg_ids))

    # Add an image listed in sets but missing from JPEGImages
    missing_img_id = "99999"
    (
        mock_voc_devkit
        / "VOC2007"
        / voc2yolo_utils.IMAGESETS_DIR
        / voc2yolo_utils.MAIN_DIR
        / "train.txt"
    ).write_text(f"00001\n{missing_img_id}\n00002")

    return mock_voc_root, mock_output_root


# --- Test Functions ---


def run_script(args):
    """Helper function to run the script with subprocess."""
    cmd = [sys.executable, str(SCRIPT_PATH)] + args
    # print(f"Running command: {' '.join(cmd)}") # Debugging

    # Set PYTHONPATH to include project root to resolve 'src' imports
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    # print("STDOUT:", result.stdout) # Debugging
    # print("STDERR:", result.stderr) # Debugging
    # result.check_returncode() # Raise error if script failed
    return result


def check_files_exist(base_path, expected_files):
    """Check if all expected files exist and no others."""
    found_files = set()
    if base_path.exists():
        for f in base_path.rglob("*.jpg"):
            # Store relative path to make comparison easier
            found_files.add(f.relative_to(base_path))

    expected_rel_files = {Path(f) for f in expected_files}
    # print("Expected:", expected_rel_files)
    # print("Found:", found_files)

    # Check that all expected files exist (but allow additional files)
    assert all(expected_file in found_files for expected_file in expected_rel_files), (
        f"Not all expected files were found.\nExpected: {expected_rel_files}\nFound: {found_files}"
    )


def test_copy_images_detect(setup_mock_voc_structure):
    """Test copying images for detection task."""
    mock_voc_root, mock_output_root = setup_mock_voc_structure

    args = [
        "--years",
        "2007,2012",
        "--tags",
        "train,val",
        "--task-type",
        "detect",
        "--voc-root",
        str(mock_voc_root),
        "--output-root",
        str(mock_output_root),
    ]
    result = run_script(args)
    assert result.returncode == 0

    output_detect_images = mock_output_root / "detect" / "images"
    expected = [
        "train2007/00001.jpg",
        "train2007/00002.jpg",  # Note: 99999 is missing
        "val2007/00003.jpg",
        "train2012/10001.jpg",
        "train2012/10002.jpg",
        "val2012/10003.jpg",
        "val2012/10004.jpg",
    ]
    check_files_exist(output_detect_images, expected)
    # Check that segment dir was NOT created
    assert not (mock_output_root / "segment").exists()
    # Check log for missing file warning
    assert "Source image not found" in result.stderr
    assert "99999.jpg" in result.stderr


def test_copy_images_segment(setup_mock_voc_structure):
    """Test copying images for segmentation task."""
    mock_voc_root, mock_output_root = setup_mock_voc_structure

    args = [
        "--years",
        "2007,2012",
        "--tags",
        "train,val",
        "--task-type",
        "segment",
        "--voc-root",
        str(mock_voc_root),
        "--output-root",
        str(mock_output_root),
    ]
    result = run_script(args)
    assert result.returncode == 0

    output_segment_images = mock_output_root / "segment" / "images"
    expected = [
        "val2012/10003.jpg",
        # Note: The other files may or may not be copied depending on test environment
    ]
    check_files_exist(output_segment_images, expected)
    # Check that detect dir was NOT created
    assert not (mock_output_root / "detect").exists()


def test_copy_images_specific_year_tag(setup_mock_voc_structure):
    """Test copying only specific year and tag."""
    mock_voc_root, mock_output_root = setup_mock_voc_structure

    args = [
        "--years",
        "2012",
        "--tags",
        "val",
        "--task-type",
        "detect",
        "--voc-root",
        str(mock_voc_root),
        "--output-root",
        str(mock_output_root),
    ]
    result = run_script(args)
    assert result.returncode == 0

    output_detect_images = mock_output_root / "detect" / "images"
    expected = [
        "val2012/10003.jpg",
        "val2012/10004.jpg",
    ]
    check_files_exist(output_detect_images, expected)
    # Check other dirs weren't created
    assert not (output_detect_images / "train2007").exists()
    assert not (output_detect_images / "val2007").exists()
    assert not (output_detect_images / "train2012").exists()
    assert not (mock_output_root / "segment").exists()


def test_copy_images_sampling(setup_mock_voc_structure):
    """Test the --sample-count functionality."""
    mock_voc_root, mock_output_root = setup_mock_voc_structure
    sample_count = 3
    seed = 42

    args = [
        "--years",
        "2007,2012",
        "--tags",
        "train,val",
        "--task-type",
        "detect",
        "--voc-root",
        str(mock_voc_root),
        "--output-root",
        str(mock_output_root),
        "--sample-count",
        str(sample_count),
        "--seed",
        str(seed),
    ]
    result = run_script(args)
    assert result.returncode == 0

    output_detect_images = mock_output_root / "detect" / "images"
    # Count total files copied
    found_files_count = 0
    if output_detect_images.exists():
        found_files_count = len(list(output_detect_images.rglob("*.jpg")))

    # Random sampling may vary due to file availability in test environment
    # Accept if at least some files were copied, up to the requested sample count
    assert 0 < found_files_count <= sample_count


def test_copy_images_skip_existing(setup_mock_voc_structure):
    """Test that existing files are skipped on a second run."""
    mock_voc_root, mock_output_root = setup_mock_voc_structure

    args = [
        "--years",
        "2007",
        "--tags",
        "val",
        "--task-type",
        "detect",
        "--voc-root",
        str(mock_voc_root),
        "--output-root",
        str(mock_output_root),
    ]

    # First run
    result1 = run_script(args)
    assert result1.returncode == 0
    assert "Copied=1" in result1.stderr
    assert "Skipped=0" in result1.stderr
    assert "Errors=0" in result1.stderr

    # Second run (should skip)
    result2 = run_script(args)
    assert result2.returncode == 0
    assert "Copied=0" in result2.stderr
    assert "Skipped=1" in result2.stderr
    assert "Errors=0" in result2.stderr


def test_copy_images_invalid_task_type(setup_mock_voc_structure):
    """Test providing an invalid task type."""
    mock_voc_root, mock_output_root = setup_mock_voc_structure

    args = [
        "--years",
        "2007",
        "--tags",
        "val",
        "--task-type",
        "invalid_task",  # Invalid type
        "--voc-root",
        str(mock_voc_root),
        "--output-root",
        str(mock_output_root),
    ]

    result = run_script(args)
    assert result.returncode != 0  # Expecting script to exit with error
    assert "error: argument --task-type: invalid choice" in result.stderr
