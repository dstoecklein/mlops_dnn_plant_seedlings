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

from model.config import core
from model.config.core import config

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

    images_df.columns = config.data_config.data_columns
    return images_df


def load_bulk_images(data_path: Path) -> pd.DataFrame:
    """
    Creates dataframe with image path and target
    """
    images_df = list()  # list with dataframes (path, target)

    for class_folder_path in Path.iterdir(data_path):  # iter subdirectories
        if class_folder_path.is_dir():  # check if directory
            for image in Path.iterdir(class_folder_path):  # iter files
                if image.is_file():  # check if file
                    tmp = pd.DataFrame([str(image), str(class_folder_path.name)]).T
                    tmp.columns = config.data_config.data_columns
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
        images_df[config.data_config.data_columns[0]],
        images_df[config.data_config.data_columns[1]],
        test_size=config.data_config.test_size,
        random_state=config.data_config.seed,
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
    joblib.dump(
        model.named_steps["dataset"],
        core.ARTIFACTS_PATH / config.app_config.pipeline_save_file,
    )
    joblib.dump(
        model.named_steps["cnn_model"].classes_,
        core.ARTIFACTS_PATH / config.app_config.classes_save_file,
    )
    model.named_steps["cnn_model"].model.save(
        core.ARTIFACTS_PATH / config.app_config.model_save_file
    )

    remove_old_pipelines(
        files_to_keep=[
            config.app_config.model_save_file,
            config.app_config.encoder_save_file,
            config.app_config.pipeline_save_file,
            config.app_config.classes_save_file,
        ]
    )


def load_pipeline() -> Pipeline:
    """
    Loads the training pipeline artifacts
    """
    dataset = joblib.load(core.ARTIFACTS_PATH / config.app_config.pipeline_save_file)

    def _build_model():
        return load_model(core.ARTIFACTS_PATH / config.app_config.model_save_file)

    classifier = KerasClassifier(
        build_fn=_build_model,
        batch_size=config.model_config.batch_size,
        # validation_split=config.model_config.validation_split,
        epochs=config.model_config.epochs,
        verbose=2,
        callbacks=model_definition.callbacks_list,
    )

    classifier.classes_ = joblib.load(
        core.ARTIFACTS_PATH / config.app_config.classes_save_file
    )

    classifier.model = _build_model()

    return Pipeline([("dataset", dataset), ("cnn_model", classifier)])


def load_encoder() -> LabelEncoder:
    encoder = joblib.load(core.ARTIFACTS_PATH / config.app_config.encoder_save_file)
    return encoder


def remove_old_pipelines(*, files_to_keep: List[str]) -> None:
    """
    Removes old pipelines, models, encoders and classes.
    To ensure there is a 1:1 mapping between the package
    version and the model version to be imported and
    used by other applications.
    """
    do_not_delete = files_to_keep + ["__int__.py"]
    for model_file in Path(core.ARTIFACTS_PATH).iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
