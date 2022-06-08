from pathlib import Path
from typing import List

from pydantic import BaseModel
from strictyaml import YAML, load

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent
DATA_PATH = ROOT.parent / "data"
ARTIFACTS_PATH = ROOT.parent / "artifacts"
CONFIG_PATH = ROOT / "config"
CONFIG_FILE = CONFIG_PATH / "config.yml"


class AppConfig(BaseModel):
    package_name: str
    pipeline_name: str
    pipeline_save_file: str
    model_save_file: str
    classes_save_file: str
    encoder_save_file: str


class DataConfig(BaseModel):
    data_folder_name: str
    image_size: int
    valid_image_extensions: List[str]
    test_size: float
    seed: int
    data_columns: List[str]


class ModelConfig(BaseModel):
    batch_size: int
    epochs: int
    learning_rate: float
    loss: str
    metrics: List[str]
    validation_split: int
    checkpoint_monitor: str
    checkpoint_mode: str
    reducelr_monitor: str
    reducelr_factor: float
    reducelr_patience: int
    reducelr_mode: str
    reducelr_minlr: float


class MasterConfig(BaseModel):
    app_config: AppConfig
    data_config: DataConfig
    model_config: ModelConfig


def get_config_path() -> Path:
    return CONFIG_FILE


def read_config_file(config_path: Path = None) -> YAML:
    if not config_path:
        config_path = get_config_path()

    if config_path:
        with open(config_path, "r") as f:
            config_file = load(f.read())
            return config_file
    else:
        raise Exception(f"No config file found at {config_path}")


def create_and_validate_config(config_file: YAML = None) -> MasterConfig:
    if config_file is None:
        config_file = read_config_file()

    _config = MasterConfig(
        app_config=AppConfig(**config_file.data),
        data_config=DataConfig(**config_file.data),
        model_config=ModelConfig(**config_file.data),
    )
    return _config


config = create_and_validate_config()
