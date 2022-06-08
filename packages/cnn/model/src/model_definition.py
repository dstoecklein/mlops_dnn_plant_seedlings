from typing import Tuple

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier

from model.config import core
from model.config.core import config


def cnn_model(
    kernel_size: Tuple = (3, 3),
    pool_size: Tuple = (2, 2),
    first_filters: int = 32,
    second_filters: int = 64,
    third_filters: int = 128,
    dropout_conv: float = 0.3,
    dropout_dense: float = 0.3,
    image_size: int = 50,
) -> Sequential:

    model = Sequential()
    model.add(
        Conv2D(
            first_filters,
            kernel_size,
            activation="relu",
            input_shape=(image_size, image_size, 3),
        )
    )
    model.add(Conv2D(first_filters, kernel_size, activation="relu"))
    model.add(MaxPooling2D(pool_size=pool_size, padding="same"))
    model.add(Dropout(dropout_conv))

    model.add(Conv2D(second_filters, kernel_size, activation="relu"))
    model.add(Conv2D(second_filters, kernel_size, activation="relu"))
    model.add(MaxPooling2D(pool_size=pool_size, padding="same"))
    model.add(Dropout(dropout_conv))

    model.add(Conv2D(third_filters, kernel_size, activation="relu"))
    model.add(Conv2D(third_filters, kernel_size, activation="relu"))
    model.add(MaxPooling2D(pool_size=pool_size, padding="same"))
    model.add(Dropout(dropout_conv))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(dropout_dense))
    model.add(Dense(12, activation="softmax"))

    model.compile(
        Adam(learning_rate=config.model_config.learning_rate),
        loss=config.model_config.loss,
        metrics=config.model_config.metrics,
    )

    return model


checkpoint = ModelCheckpoint(
    core.ARTIFACTS_PATH / config.app_config.model_save_file,
    monitor=config.model_config.checkpoint_monitor,
    verbose=1,
    save_best_only=True,
    mode=config.model_config.checkpoint_mode,
)

reduce_lr = ReduceLROnPlateau(
    monitor=config.model_config.reducelr_monitor,
    factor=config.model_config.reducelr_factor,
    patience=config.model_config.reducelr_patience,
    verbose=1,
    mode=config.model_config.reducelr_mode,
    min_lr=config.model_config.reducelr_minlr,
)


callbacks_list = [checkpoint, reduce_lr]

cnn_clf = KerasClassifier(
    build_fn=cnn_model,
    batch_size=config.model_config.batch_size,
    # validation_split=config.model_config.validation_split,
    epochs=config.model_config.epochs,
    verbose=2,
    callbacks=callbacks_list,
    image_size=config.data_config.image_size,
)


if __name__ == "__main__":
    model = cnn_model()
    model.summary()
