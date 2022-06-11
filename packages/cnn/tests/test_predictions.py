from model import __version__ as _version
from model.src.predict import make_single_prediction


def test_make_prediction_on_single_sample_charlock(charlock_dir):
    # Given
    filename = "1.png"
    expected_class = "Charlock"

    # When
    results = make_single_prediction(
        image_path=charlock_dir, 
        filename=filename
    )

    # Then
    assert results["predictions"] is not None
    assert results["readable_predictions"][0] == expected_class
    assert results["version"] == _version


def test_make_prediction_on_single_sample_scentless_mayweed(scentless_mayweed_dir):
    # Given
    filename = "1.png"
    expected_class = "Scentless Mayweed"

    # When
    results = make_single_prediction(
        image_path=scentless_mayweed_dir, 
        filename=filename
    )

    # Then
    assert results["predictions"] is not None
    assert results["readable_predictions"][0] == expected_class
    assert results["version"] == _version
