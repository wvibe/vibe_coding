"""
Global pytest configuration and fixtures
"""

import os
import sys
from pathlib import Path

import pytest


def pytest_configure(config):
    """
    Pytest configuration hook to add project root to sys.path
    This allows imports to work the same way in tests as in the main code
    """
    project_root = str(Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory as a Path object"""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def data_root():
    """Return the data root directory from environment variable"""
    return Path(os.getenv("DATA_ROOT", "data"))


@pytest.fixture(scope="session")
def voc_root():
    """Return the VOC dataset root directory from environment variable"""
    return Path(
        os.getenv("VOC_ROOT", os.path.join(os.getenv("DATA_ROOT", "data"), "VOCdevkit"))
    )
