from pathlib import Path

# PATHS
PWD = Path(__file__).resolve().parent
MODEL_PATH = PWD.parent
PACKAGE_PATH = MODEL_PATH.parent

# VERSION
with open(MODEL_PATH / "VERSION") as version_file:
    _version = version_file.read().strip()

# FOLDERS
DATASET_FOLDER = PACKAGE_PATH / "data"
DATA_FOLDER = DATASET_FOLDER / "plant_seedlings_v2"
TEST_DATA_FOLDER = DATASET_FOLDER / "test_data"
ARTIFACTS_FOLDER = PACKAGE_PATH / "artifacts"

# MODEl PERSISTING
MODEL_NAME = "cnn_model"
PIPELINE_NAME = "cnn_pipeline_output"
CLASSES_NAME = "classes"
ENCODER_NAME = "encoder"

# ARTIFACTS
MODEL_SAVE_FILE = ARTIFACTS_FOLDER / f"{MODEL_NAME}_{_version}.h5"
PIPELINE_SAVE_FILE = ARTIFACTS_FOLDER / f"{PIPELINE_NAME}_{_version}.pkl"
CLASSES_SAVE_FILE = ARTIFACTS_FOLDER / f"{CLASSES_NAME}_{_version}.pkl"
ENCODER_SAVE_FILE = ARTIFACTS_FOLDER / f"{ENCODER_NAME}_{_version}.pkl"

# DATA CONFIG
IMAGE_SIZE = 150
TEST_SIZE = 0.2
SEED = 101
VALID_IMG_EXTENSIONS = [".png", ".jpg"]
DF_COLS = ["image", "target"]

# MODEL FITTING
BATCH_SIZE = 10
EPOCHS = 10
LEARNING_RATE = 0.0001
LOSS = "binary_crossentropy"
METRICS = ["accuracy"]
VAL_SPLIT = 10
CP_MONITOR = "accuracy"  # checkpoints
CP_MODE = "max"
REDUCELR_MONITOR = "accuracy"
REDUCELR_FACTOR = 0.5
REDUCELR_PATIENCE = 2
REDUCELR_MODE = "max"
REDUCELR_MINLR = 0.00001
