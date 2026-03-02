# shared fixtures for temp lcp/game files
from pathlib import Path

import pytest


@pytest.fixture
def examples_dir():
    return Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def example_lcp_path(examples_dir):
    p = examples_dir / "lcp"
    if not p.exists():
        pytest.skip("examples/lcp not found")
    return str(p)


@pytest.fixture
def example_game_path(examples_dir):
    p = examples_dir / "game"
    if not p.exists():
        pytest.skip("examples/game not found")
    return str(p)


@pytest.fixture
def temp_lcp_file(tmp_path):
    # write lcp content to a temp file, return path
    def _write(content: str):
        path = tmp_path / "test.lcp"
        path.write_text(content, encoding="utf-8")
        return str(path)
    return _write


@pytest.fixture
def temp_game_file(tmp_path):
    def _write(content: str):
        path = tmp_path / "test.game"
        path.write_text(content, encoding="utf-8")
        return str(path)
    return _write
