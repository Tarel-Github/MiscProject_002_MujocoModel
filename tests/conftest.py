import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HUMANOID = ROOT / "examples" / "models" / "humanoid.xml"


@pytest.fixture
def humanoid_path():
    return HUMANOID


@pytest.fixture
def tmpdir_path(tmp_path):
    return tmp_path
