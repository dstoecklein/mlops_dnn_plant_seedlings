from sklearn.pipeline import Pipeline

import model.src.preprocessors as pp
from model.config import config
from model.src import model_definition

pipeline = Pipeline(
    [
        ("dataset", pp.CreateDataset(config.IMAGE_SIZE)),
        ("cnn_model", model_definition.cnn_clf),
    ]
)
