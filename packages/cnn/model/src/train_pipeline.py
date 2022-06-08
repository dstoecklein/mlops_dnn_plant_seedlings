import data_management as dm
import joblib
import pipeline as pipe
import preprocessors as pp

from model.config import core
from model.config.core import config


def run_training(save_pipeline: bool = True):
    images_df = dm.load_images(core.DATA_PATH / config.data_config.data_folder_name)
    X_train, X_test, y_train, y_test = dm.get_train_test_split(images_df)

    encoder = pp.TargetEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)

    pipe.pipeline.fit(X_train, y_train)

    if save_pipeline:
        joblib.dump(encoder, core.ARTIFACTS_PATH / config.app_config.encoder_save_file)
        dm.save_pipeline(pipe.pipeline)


if __name__ == "__main__":
    run_training(save_pipeline=True)
