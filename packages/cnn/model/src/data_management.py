import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import model_definition
import pandas as pd
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from model.config import config

_logger = logging.getLogger(__name__)


def load_single_image(image_path: Path, filename: str) -> pd.DataFrame:
    """
    Creates dataframe with image path and target
    """
    image = image_path / filename
    if image.is_file():
        images_df = pd.DataFrame([image, "unkown"]).T
    else:
        raise RuntimeError(f"No such file {image}")

    images_df.columns = config.DF_COLS
    return images_df


def load_images(data_path: Path) -> pd.DataFrame:
    """
    Creates dataframe with image path and target
    """
    images_df = list()  # list with dataframes (path, target)

    for class_folder_path in Path.iterdir(data_path):  # iter subdirectories
        if class_folder_path.is_dir():  # check if directory
            for image in Path.iterdir(class_folder_path):  # iter files
                if image.is_file():  # check if file
                    tmp = pd.DataFrame([str(image), str(class_folder_path.name)]).T
                    tmp.columns = config.DF_COLS
                    images_df.append(tmp)
                else:
                    raise RuntimeError(f"No such file {image}")
        else:
            raise RuntimeError(f"No such path {class_folder_path}")

    images_df = pd.concat(images_df, axis=0, ignore_index=True)
    return images_df


def get_train_test_split(images_df: pd.DataFrame) -> Tuple:
    """
    Performs train test split
    """
    X_train, X_test, y_train, y_test = train_test_split(
        images_df[config.DF_COLS[0]],
        images_df[config.DF_COLS[1]],
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
    )

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test


def save_pipeline(model: Pipeline) -> None:
    """
    Saves the training pipeline artifacts
    """
    joblib.dump(model.named_steps["dataset"], config.PIPELINE_SAVE_FILE)
    joblib.dump(model.named_steps["cnn_model"].classes_, config.CLASSES_SAVE_FILE)
    model.named_steps["cnn_model"].model.save(config.MODEL_SAVE_FILE)

    remove_old_pipelines(
        files_to_keep=[
            config.MODEL_SAVE_FILE.name,
            config.ENCODER_SAVE_FILE.name,
            config.PIPELINE_SAVE_FILE.name,
            config.CLASSES_SAVE_FILE.name,
        ]
    )


def load_pipeline() -> Pipeline:
    """
    Loads the training pipeline artifacts
    """
    dataset = joblib.load(config.PIPELINE_SAVE_FILE)

    def _build_model():
        return load_model(config.MODEL_SAVE_FILE)

    classifier = KerasClassifier(
        build_fn=_build_model,
        batch_size=config.BATCH_SIZE,
        # validation_split=config.model_config.validation_split,
        epochs=config.EPOCHS,
        verbose=2,
        callbacks=model_definition.callbacks_list,
    )

    classifier.classes_ = joblib.load(config.CLASSES_SAVE_FILE)
    classifier.model = _build_model()

    return Pipeline([("dataset", dataset), ("cnn_model", classifier)])


def load_encoder() -> LabelEncoder:
    encoder = joblib.load(config.ENCODER_SAVE_FILE)
    return encoder


def remove_old_pipelines(*, files_to_keep: List[str]) -> None:
    """
    Removes old pipelines, models, encoders and classes.
    To ensure there is a 1:1 mapping between the package
    version and the model version to be imported and
    used by other applications.
    """
    do_not_delete = files_to_keep + ["__int__.py"]
    for model_file in Path(config.ARTIFACTS_FOLDER).iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


if __name__ == "__main__":
    df = load_images(config.DATA_FOLDER)
    print(df.head())
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    print(X_train, X_train.shape)
