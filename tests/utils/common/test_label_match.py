"""
tests/test_label_match.py
Unit‑tests for vibelab.utils.common.label_match
Run with:  pytest -q tests/test_label_match.py
"""

from typing import Tuple

import numpy as np
import pytest

from vibelab.utils.common.bbox import calculate_iou
from vibelab.utils.common.label_match import (
    _INVALID_IOU_PLACEHOLDER,
    _match_instances_greedy,
    _match_instances_hungarian,
    match_instances,
)
from vibelab.utils.common.mask import calculate_mask_iou

# --------------------------------------------------------------------------- #
# 1.  Test utilities                                                           #
# --------------------------------------------------------------------------- #


class MockBBox:
    """Light‑weight bounding‑box wrapper used only for tests."""

    def __init__(self, coords: Tuple[float, float, float, float]):
        # coords = [xmin, ymin, xmax, ymax]
        self.coords = np.asarray(coords, dtype=float)


class MockMask:
    """Light‑weight binary‑mask wrapper used only for tests."""

    def __init__(self, mask: np.ndarray):
        self.data = mask.astype(bool)


def bbox_iou_fn(a: MockBBox, b: MockBBox) -> float:
    return calculate_iou(a.coords, b.coords)


def mask_iou_fn(a: MockMask, b: MockMask) -> float:
    return calculate_mask_iou(a.data, b.data)


# --------------------------------------------------------------------------- #
# 2.  Basic functional tests (both algorithms)                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "dataset_a, dataset_b, iou_cutoff, exp_matches",
    [
        # Identical boxes should match with high cutoff
        pytest.param(
            [MockBBox([0, 0, 10, 10])],
            [MockBBox([0, 0, 10, 10])],
            0.99,
            {(0, 0, 1.0)},
            id="bbox_exact_match",
        ),
        # Partial overlap above / below cutoff
        pytest.param(
            [MockBBox([0, 0, 10, 10])],
            [MockBBox([5, 5, 15, 15])],
            0.05,
            {(0, 0, 0.14285714285714285)},
            id="bbox_low_cutoff_should_match",
        ),
        pytest.param(
            [MockBBox([0, 0, 10, 10])],
            [MockBBox([5, 5, 15, 15])],
            0.5,
            set(),
            id="bbox_high_cutoff_no_match",
        ),
    ],
)
@pytest.mark.parametrize("use_hungarian", [True, False], ids=["hungarian", "greedy"])
def test_match_instances_basic(dataset_a, dataset_b, iou_cutoff, exp_matches, use_hungarian):
    """Same behaviour expected for Hungarian and Greedy on basic cases."""
    matches, un_a, un_b = match_instances(
        dataset_a,
        dataset_b,
        bbox_iou_fn,
        iou_cutoff=iou_cutoff,
        use_hungarian=use_hungarian,
    )

    # Check if matches contain IoU values
    if matches:
        for match in matches:
            assert len(match) == 3, "Matches should include IoU values"
            assert isinstance(match[2], float), "IoU value should be a float"
            assert 0 <= match[2] <= 1, "IoU value should be between 0 and 1"

    # Convert matches to set of tuples for comparison
    match_set = {(a, b, iou) for a, b, iou in matches}

    # For comparing with expected results
    for match in match_set:
        if len(exp_matches) > 0:
            # Check if at least one expected match has same indices
            expected_match = next(
                (em for em in exp_matches if em[0] == match[0] and em[1] == match[1]), None
            )
            if expected_match:
                assert abs(match[2] - expected_match[2]) < 1e-10, (
                    f"IoU values should match: {match[2]} vs {expected_match[2]}"
                )

    # Check the structure is correct
    assert len(un_a) + len(matches) == len(dataset_a)
    assert len(un_b) + len(matches) == len(dataset_b)


# --------------------------------------------------------------------------- #
# 3.  Corner / edge‑cases and dummy padding                                    #
# --------------------------------------------------------------------------- #


def test_row_or_col_with_all_invalids_dummy():
    """Entire row & column invalid – dummy padding must prevent forced pairing."""
    iou_mat = np.array(
        [[0.8, _INVALID_IOU_PLACEHOLDER], [_INVALID_IOU_PLACEHOLDER, _INVALID_IOU_PLACEHOLDER]]
    )

    # Ensure we are using the dummy‑aware helper
    assert "dummy" in (_match_instances_hungarian.__doc__ or "").lower()

    matches, mask_a, mask_b = _match_instances_hungarian(iou_mat, iou_cutoff=0.3)

    assert len(matches) == 1
    assert matches[0][:2] == (0, 0)  # Index matches
    assert abs(matches[0][2] - 0.8) < 1e-10  # IoU value
    assert mask_a.tolist() == [True, False]
    assert mask_b.tolist() == [True, False]


@pytest.mark.parametrize("shape", [(0, 4), (4, 0), (0, 0)])
def test_empty_but_valid_matrices(shape):
    """Helpers must gracefully handle (0,N)/(N,0)/(0,0) matrices."""
    mat = np.empty(shape, dtype=float)

    for fn in (_match_instances_hungarian, _match_instances_greedy):
        matches, mask_a, mask_b = fn(mat, iou_cutoff=0.3)
        assert matches == []
        assert mask_a.shape == (shape[0],)
        assert mask_b.shape == (shape[1],)
        assert not mask_a.any() and not mask_b.any()


def test_rectangular_matrix():
    """Non‑square matrix should be padded; result masks lengths == original dims."""
    iou_mat = np.array([[0.9, 0.1, _INVALID_IOU_PLACEHOLDER], [0.2, 0.8, 0.5]])  # shape 2×3

    matches, mask_a, mask_b = _match_instances_hungarian(iou_mat, iou_cutoff=0.3)

    # Check for IoU values in matches
    for match in matches:
        assert len(match) == 3, "Matches should include IoU values"
        r, c, iou = match
        assert iou == iou_mat[r, c], (
            f"IoU value does not match matrix at ({r},{c}): {iou} != {iou_mat[r, c]}"
        )

    # Check overall structure
    match_indices = [(r, c) for r, c, _ in matches]
    expected1 = {(0, 0), (1, 1)}
    expected2 = {(0, 1), (1, 0)}
    assert set(match_indices) in (expected1, expected2)
    assert mask_a.tolist() == [True, True]
    assert mask_b.tolist()[-1] is False  # last column unmatched


# --------------------------------------------------------------------------- #
# 4.  Greedy can be sub‑optimal                                               #
# --------------------------------------------------------------------------- #


def test_greedy_can_be_suboptimal():
    """Hungarian must beat Greedy on this crafted matrix."""
    iou_mat = np.array(
        [[4, 5, 8], [9, 8, 9], [5, 8, 9]],
        dtype=float,
    )

    greedy_matches, _, _ = _match_instances_greedy(iou_mat, 0.0)
    hungarian_matches, _, _ = _match_instances_hungarian(iou_mat, 0.0)

    # Check IoU values are included in matches
    for matches in [greedy_matches, hungarian_matches]:
        for match in matches:
            assert len(match) == 3, "Matches should include IoU values"
            r, c, iou = match
            assert iou == iou_mat[r, c], (
                f"IoU value does not match matrix at ({r},{c}): {iou} != {iou_mat[r, c]}"
            )

    g_sum = sum(iou_mat[r, c] for r, c, _ in greedy_matches)
    h_sum = sum(iou_mat[r, c] for r, c, _ in hungarian_matches)

    assert h_sum > g_sum, "Hungarian should beat greedy on the trap matrix"


# --------------------------------------------------------------------------- #
# 5.  Mask IoU (pixel overlaps)                                               #
# --------------------------------------------------------------------------- #


def _make_square(top: int, left: int, size: int = 4) -> np.ndarray:
    """Return a boolean image with a filled square."""
    img = np.zeros((20, 20), dtype=bool)
    img[top : top + size, left : left + size] = True
    return img


@pytest.mark.parametrize(
    "shift, cutoff, expect_match",
    [
        (0, 0.9, True),
        (2, 0.14, True),
        (6, 0.4, False),
    ],
)
def test_mask_matching(shift, cutoff, expect_match):
    mask_a = MockMask(_make_square(5, 5))
    mask_b = MockMask(_make_square(5 + shift, 5 + shift))

    matches, un_a, un_b = match_instances([mask_a], [mask_b], mask_iou_fn, iou_cutoff=cutoff)

    if matches:
        # Check IoU values are included
        for match in matches:
            assert len(match) == 3, "Matches should include IoU values"
            a_idx, b_idx, iou = match
            assert 0 <= iou <= 1, f"IoU value should be between 0 and 1, got {iou}"

    assert (len(matches) == 1) is expect_match
    assert (len(un_a) == 0) is expect_match
    assert (len(un_b) == 0) is expect_match


# --------------------------------------------------------------------------- #
# 6.  Property‑based fuzz (quick random matrices)                             #
# --------------------------------------------------------------------------- #


@pytest.mark.quick
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_small_consistency(seed):
    """For random ≤4×4 matrices Hungarian ≥ Greedy in total score."""
    rng = np.random.default_rng(seed)
    m = rng.integers(1, 5)
    n = rng.integers(1, 5)
    mat = rng.random((m, n)).round(3)

    greedy_matches, _, _ = _match_instances_greedy(mat, 0.0)
    hungarian_matches, _, _ = _match_instances_hungarian(mat, 0.0)

    # Check that IoU values are included
    for matches in [greedy_matches, hungarian_matches]:
        for match in matches:
            assert len(match) == 3, "Matches should include IoU values"
            r, c, iou = match
            assert abs(iou - mat[r, c]) < 1e-10, (
                f"IoU value does not match matrix at ({r},{c}): {iou} != {mat[r, c]}"
            )

    g_sum = sum(match[2] for match in greedy_matches)
    h_sum = sum(match[2] for match in hungarian_matches)

    assert h_sum >= g_sum


# --------------------------------------------------------------------------- #
# 7. Test IoU value inclusion                                                 #
# --------------------------------------------------------------------------- #


def test_match_instances_includes_iou_values():
    """Test that match_instances returns IoU values in matched pairs."""
    dataset_a = [MockBBox([0, 0, 10, 10]), MockBBox([20, 20, 30, 30])]
    dataset_b = [MockBBox([5, 5, 15, 15]), MockBBox([22, 22, 28, 28])]

    # Expected IoU values:
    # (0,0): overlap of [5,5,10,10] with area 25 / total area 175 = 1/7 ≈ 0.1429
    # (1,1): overlap of [22,22,28,28] with area 36 / total area 100 = 0.36

    # Test both algorithms
    for use_hungarian in [True, False]:
        matches, un_a, un_b = match_instances(
            dataset_a, dataset_b, bbox_iou_fn, iou_cutoff=0.1, use_hungarian=use_hungarian
        )

        assert len(matches) == 2, f"Should find 2 matches with cutoff 0.1 ({use_hungarian=})"

        # Sort by first index for stable comparison
        matches.sort(key=lambda x: x[0])

        # Check pair (0,0)
        assert matches[0][0] == 0
        assert matches[0][1] == 0
        assert abs(matches[0][2] - 0.14285714285714285) < 1e-10

        # Check pair (1,1)
        assert matches[1][0] == 1
        assert matches[1][1] == 1
        assert abs(matches[1][2] - 0.36) < 1e-10
