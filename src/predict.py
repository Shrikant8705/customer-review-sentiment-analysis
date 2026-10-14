from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.joblib"


def predict_sentiment(review: str) -> tuple[str, float]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model was not found. Run `python src/train.py` first."
        )

    model = joblib.load(MODEL_PATH)
    sentiment = model.predict([review])[0]
    probabilities = model.predict_proba([review])[0]
    confidence = float(max(probabilities))
    return sentiment, confidence


if __name__ == "__main__":
    examples = [
        "The customer service was extremely helpful.",
        "The product broke after one day.",
        "The item is average for the price.",
    ]

    for review in examples:
        sentiment, confidence = predict_sentiment(review)
        print(f"Review: {review}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2%}\n")
