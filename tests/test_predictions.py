from src.predict import predict_sentiment, predict_with_details
from src.train import train_model


def setup_module():
    train_model()


def test_bad_product_is_negative():
    sentiment, confidence = predict_sentiment("The product is bad!")
    assert sentiment == "negative"
    assert confidence > 0.45


def test_terrible_product_is_negative():
    sentiment, _ = predict_sentiment("terrible product")
    assert sentiment == "negative"


def test_excellent_product_is_positive():
    sentiment, _ = predict_sentiment("excellent product")
    assert sentiment == "positive"


def test_neutral_example_is_neutral():
    sentiment, _ = predict_sentiment("It is okay, nothing special.")
    assert sentiment == "neutral"


def test_detailed_prediction_has_probabilities_and_signals():
    result = predict_with_details("Terrible product... broke in 2 days!!!")

    assert result["sentiment"] == "negative"
    assert isinstance(result["probabilities"], dict)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6
    assert "negative" in result["signals"]
