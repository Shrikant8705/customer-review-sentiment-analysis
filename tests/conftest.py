from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture(scope="session", autouse=True)
def trained_model() -> None:
    from src.train import train_model

    train_model()
