import model_definition
import preprocessors as pp
from sklearn.pipeline import Pipeline

from model.config import core
from model.config.core import config

pipeline = Pipeline(
    [
        ("dataset", pp.CreateDataset(config.data_config.image_size)),
        ("cnn_model", model_definition.cnn_clf),
    ]
)

if __name__ == "__main__":
    import data_management as dm
    from sklearn.metrics import accuracy_score

    images_df = dm.load_images(core.DATA_PATH / config.data_config.data_folder_name)
    X_train, X_test, y_train, y_test = dm.get_train_test_split(images_df)

    encoder = pp.TargetEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)

    pipeline.fit(X_train, y_train)

    test_y = encoder.transform(y_test)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        encoder.encoder.transform(y_test),
        predictions,
        normalize=True,
        sample_weight=None,
    )

    print("Accuracy: ", accuracy)
