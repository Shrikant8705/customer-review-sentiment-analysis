from pathlib import Path
from functools import lru_cache
import sys

import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.sentiment_model import extract_signal_words

MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.joblib"


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model was not found. Run `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_with_details(review: str) -> dict[str, object]:
    model = _load_model()
    probabilities = model.predict_proba([review])[0]
    classes = list(model.classes_)

    best_index = int(probabilities.argmax())
    sentiment = classes[best_index]
    confidence = float(probabilities[best_index])

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": {
            label: float(probability)
            for label, probability in zip(classes, probabilities)
        },
        "signals": extract_signal_words(review),
    }


def predict_sentiment(review: str) -> tuple[str, float]:
    result = predict_with_details(review)
    return result["sentiment"], float(result["confidence"])


if __name__ == "__main__":
    examples = [
        "The customer service was extremely helpful.",
        "The product broke after one day.",
        "The item is average for the price.",
    ]

    for review in examples:
        result = predict_with_details(review)
        print(f"Review: {review}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence estimate: {result['confidence']:.2%}")
        print(f"Signals: {result['signals']}\n")
