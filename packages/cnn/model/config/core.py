from pathlib import Path
from pydantic import BaseModel
from strictyaml import YAML, load
from typing import List

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent
DATA_PATH = ROOT.parent / "data"
CONFIG_PATH = ROOT / "config"
ARTIFACTS_PATH = ROOT / "artifacts"
CONFIG_FILE = CONFIG_PATH / "config.yml"


class AppConfig(BaseModel):
    package_name: str
    pipeline_name: str
    data_folder_name: str
    pipeline_save_file: str
    model_save_file: str
    classes_save_file: str
    encoder_save_file: str


class ModelConfig(BaseModel):
    test_size: float
    seed: int
    image_size: int
    valid_image_extensions: List[str]
    data_columns: List[str]
    batch_size: int
    epochs: int


class MasterConfig(BaseModel):
    app_config: AppConfig
    model_config: ModelConfig


def get_config_path() -> str:
    return CONFIG_FILE


def read_config_file(config_path: str=None) -> YAML:
    if not config_path:
        config_path = get_config_path()

    if config_path:
        with open(config_path, "r") as f:
            config_file = load(f.read())
            return config_file
    else:
        raise Exception(f"No config file found at {config_path}")


def create_and_validate_config(config_file: YAML=None) -> MasterConfig:
    if config_file is None:
        config_file = read_config_file()
    
    _config = MasterConfig(
        app_config=AppConfig(**config_file.data),
        model_config=ModelConfig(**config_file.data)
    )
    return _config


config = create_and_validate_config()
