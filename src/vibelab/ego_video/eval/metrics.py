"""Placeholder metrics for future motion-analysis evaluation."""

from __future__ import annotations

import numpy as np


def translation_magnitude(dx: float, dy: float) -> float:
    """Compute the Euclidean magnitude of a 2D translation."""

    return float(np.sqrt(dx**2 + dy**2))
