from pathlib import Path

from model.config import config

path = Path(config.MODEL_PATH) / "VERSION"
with path.open() as version_file:
    __version__ = version_file.readline().strip()
