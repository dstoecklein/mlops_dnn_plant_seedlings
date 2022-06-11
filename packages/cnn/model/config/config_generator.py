from configparser import ConfigParser
from pathlib import Path

config_file = ConfigParser()
CONFIG_FILE_NAME = "config.cfg"
CWD = Path(__file__).resolve().parent
MODEL_PATH = CWD.parent
ROOT_PATH = MODEL_PATH.parent

with open(MODEL_PATH / "VERSION") as version_file:
    _version = version_file.read().strip()

config_file["AppConfig"] = {
    "package_name": "cnn",
    "root_path": ROOT_PATH,
    "data_path": ROOT_PATH / "data",
    "artifacts_path": ROOT_PATH / "artifacts",
    "config_path": CWD,
    "config_file": CWD / "config.yml",
    "pipeline_save_file": ROOT_PATH
    / "artifacts"
    / f"cnn_pipeline_output_{_version}.pkl",
    "model_save_file": ROOT_PATH / "artifacts" / f"cnn_model_{_version}.h5",
    "classes_save_file": ROOT_PATH / "artifacts" / f"classes_{_version}.pkl",
    "encoder_save_file": ROOT_PATH / "artifacts" / f"encoder_{_version}.pkl",
}

config_file["DataConfig"] = {
    "data_folder_name": "plant_seedlings_v2",
    "image_size": 150,
    "valid_image_extensions": [".png", ".jpg"],
    "test_size": 0.2,
    "seed": 101,
    "data_columns": ["image", "target"],
}

config_file["ModelConfig"] = {
    "version": _version,
    "batch_size": 10,
    "epochs": 10,
    "learning_rate": 0.0001,
    "loss": "binary_crossentropy",
    "metrics": ["accuracy"],
    "validation_split": 10,
    "checkpoint_monitor": "accuracy",
    "checkpoint_mode": "max",
    "reducelr_monitor": "accuracy",
    "reducelr_factor": 0.5,
    "reducelr_patience": 2,
    "reducelr_mode": "max",
    "reducelr_minlr": 0.00001,
}


def write_config() -> None:
    with open(CWD / CONFIG_FILE_NAME, "w") as configfileObj:
        configfileObj.seek(0)
        config_file.write(configfileObj)
        configfileObj.truncate()
        configfileObj.flush()
        configfileObj.close()
    print("Config file created")


def read_config() -> ConfigParser:
    config_file = ConfigParser()
    config_file.read(CWD / CONFIG_FILE_NAME)
    return config_file


if __name__ == "__main__":
    import ast

    write_config()
    config_file = read_config()
    print(type(config_file))
    print(config_file.sections())
    print(config_file["AppConfig"].get("package_name"))
    print(config_file["AppConfig"].get("pipeline_save_file"))
    print(config_file["AppConfig"].get("model_save_file"))
    print(config_file["AppConfig"].get("classes_save_file"))
    print(config_file["AppConfig"].get("encoder_save_file"))
    print(ast.literal_eval(config_file["DataConfig"].get("valid_image_extensions"))[0])
