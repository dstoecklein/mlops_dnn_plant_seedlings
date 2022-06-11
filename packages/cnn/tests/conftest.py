import pytest

from model.config import config


@pytest.fixture
def charlock_dir():
    test_data_dir = config.TEST_DATA_FOLDER
    charlock_dir = test_data_dir / "Charlock"
    return charlock_dir


@pytest.fixture
def scentless_mayweed_dir():
    test_data_dir = config.TEST_DATA_FOLDER
    scentless_mayweed_dir = test_data_dir / "Scentless Mayweed"
    return scentless_mayweed_dir
