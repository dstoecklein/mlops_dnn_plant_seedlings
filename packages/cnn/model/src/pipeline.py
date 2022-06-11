import model_definition
import preprocessors as pp
from sklearn.pipeline import Pipeline

from model.config import config

pipeline = Pipeline(
    [
        ("dataset", pp.CreateDataset(config.IMAGE_SIZE)),
        ("cnn_model", model_definition.cnn_clf),
    ]
)
