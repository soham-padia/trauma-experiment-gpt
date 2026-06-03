"""
Shared pytest fixtures and import-path setup for the anxiety-in-LLMs test suite.

These tests import functions from `src/` directly. We add the repo's `src/`
directory to ``sys.path`` here so individual test files can `import` the
analysis modules without any package shenanigans.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make `src/` importable as a top-level package directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def stai_questions() -> list[dict]:
    """The 20-item STAI question bank, exactly as the experiment uses it."""
    path = SRC_DIR / "STAI" / "questionnaires.json"
    with open(path) as f:
        data = json.load(f)
    questions = data["STAI"]["questions"]
    assert len(questions) == 20, "STAI bank should have exactly 20 items"
    return questions


@pytest.fixture(scope="session")
def stai_preamble() -> str:
    path = SRC_DIR / "STAI" / "questionnaires.json"
    with open(path) as f:
        data = json.load(f)
    return data["STAI"]["preamble"]


@pytest.fixture
def trauma_military_answers() -> list[int]:
    """
    Raw answers (1-indexed labels: 1=almost never ... 4=almost always)
    from the actual Llama-3.1-70B `trauma_stai__military__none` session.
    Source: src/results/hidden_states/Meta-Llama-3.1-70B-Instruct/metadata.json
    Properly scored, this should total 79 (very anxious, max=80).
    """
    return [1, 1, 4, 4, 1, 4, 3, 1, 4, 1, 1, 4, 4, 4, 1, 1, 4, 4, 1, 1]


@pytest.fixture
def all_anxious_answers() -> list[int]:
    """
    The 'maximally anxious' answer pattern.
    Reverse-scored items (1,2,5,8,10,11,15,16,19,20 1-indexed) get raw=1
        → contribution 5 - 1 = 4
    Direct-scored items get raw=4
        → contribution 4
    Total = 20 * 4 = 80
    """
    REVERSE = {1, 2, 5, 8, 10, 11, 15, 16, 19, 20}
    return [1 if (i + 1) in REVERSE else 4 for i in range(20)]


@pytest.fixture
def all_calm_answers() -> list[int]:
    """
    The 'maximally calm' answer pattern. Mirror of all_anxious.
    Reverse items raw=4 → 5 - 4 = 1; direct items raw=1 → 1. Total = 20.
    """
    REVERSE = {1, 2, 5, 8, 10, 11, 15, 16, 19, 20}
    return [4 if (i + 1) in REVERSE else 1 for i in range(20)]
