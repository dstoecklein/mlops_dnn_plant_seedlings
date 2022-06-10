import logging
from pathlib import Path

from model import __version__ as _version
from model.src import data_management as dm

_logger = logging.getLogger(__name__)

MODEL_PIPELINE = dm.load_pipeline()
ENCODER = dm.load_encoder()


def make_single_prediction(*, image_path: Path, filename: str):
    """
    Make a single prediction using the saved model pipeline.
    Args:
        image_path: Location of the image to classify
        image_name: Filename of the image to classify

    Returns:
        Dictionary with both raw predictions and readable values.
    """
    image_df = dm.load_single_image(image_path=image_path, filename=filename)
    prepared_df = image_df["image"].astype(str).reset_index(drop=True)
    _logger.info(
        f'received input array: {prepared_df},'
        f'filename: {filename}'
    )

    predictions = MODEL_PIPELINE.predict(prepared_df)
    readable_predictions = ENCODER.encoder.inverse_transform(predictions)
    _logger.info(
        f'Made prediction: {predictions} '
        f'with model version: {_version}'
    )

    result_dict = dict(
        predictions=predictions,
        readable_predictions=readable_predictions,
        version=_version,
    )
    return result_dict


if __name__ == "__main__":
    from model.config import core
    from model.config.core import config

    result = make_single_prediction(
        image_path=core.DATA_PATH / config.data_config.data_folder_name / "Cleavers",
        filename="9.png",
    )
    print(result)
