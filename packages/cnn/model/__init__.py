from pathlib import Path
from model.config import core


path = Path(core.ROOT) / "VERSION"
with path.open() as version_file:
    __version__ = version_file.readline().strip()
    print(__version__)
