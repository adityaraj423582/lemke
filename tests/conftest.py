"""
Shared pytest fixtures for the lemke package tests.
Provides temporary LCP and game files for file-based tests.
"""
from pathlib import Path

import pytest


@pytest.fixture
def examples_dir():
    """Path to the examples directory (project root / examples)."""
    return Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def example_lcp_path(examples_dir):
    """Path to the bundled examples/lcp file."""
    p = examples_dir / "lcp"
    if not p.exists():
        pytest.skip("examples/lcp not found")
    return str(p)


@pytest.fixture
def example_game_path(examples_dir):
    """Path to the bundled examples/game file."""
    p = examples_dir / "game"
    if not p.exists():
        pytest.skip("examples/game not found")
    return str(p)


@pytest.fixture
def temp_lcp_file(tmp_path):
    """
    Yields a callable that writes LCP content to a temp file and returns its path.
    Usage: path = temp_lcp_file(content_string)
    """
    def _write(content: str):
        path = tmp_path / "test.lcp"
        path.write_text(content, encoding="utf-8")
        return str(path)
    return _write


@pytest.fixture
def temp_game_file(tmp_path):
    """
    Yields a callable that writes game file content to a temp file and returns its path.
    Usage: path = temp_game_file(content_string)
    """
    def _write(content: str):
        path = tmp_path / "test.game"
        path.write_text(content, encoding="utf-8")
        return str(path)
    return _write
