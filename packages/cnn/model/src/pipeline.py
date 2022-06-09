import model_definition
import preprocessors as pp
from sklearn.pipeline import Pipeline

from model.config.core import config

pipeline = Pipeline(
    [
        ("dataset", pp.CreateDataset(config.data_config.image_size)),
        ("cnn_model", model_definition.cnn_clf),
    ]
)
