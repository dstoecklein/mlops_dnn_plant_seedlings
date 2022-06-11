import joblib

import model.src.data_management as dm
import model.src.pipeline as pipe
import model.src.preprocessors as pp
from model.config import config


def run_training(save_pipeline: bool = True):
    """
    Train the Convolutional Neural Network
    """
    images_df = dm.load_images(config.DATA_FOLDER)
    X_train, X_test, y_train, y_test = dm.get_train_test_split(images_df)

    # one-hot encode the target
    encoder = pp.TargetEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)

    # train the pipeline
    pipe.pipeline.fit(X_train, y_train)

    # save artifacts
    if save_pipeline:
        joblib.dump(encoder, config.ENCODER_SAVE_FILE)
        dm.save_pipeline(pipe.pipeline)


if __name__ == "__main__":
    run_training(save_pipeline=True)
