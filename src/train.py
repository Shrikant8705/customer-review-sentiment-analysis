from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.sentiment_model import build_model_pipeline

DATA_PATH = ROOT_DIR / "data" / "reviews.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"



def train_model() -> None:
    df = pd.read_csv(DATA_PATH).dropna(subset=["review", "sentiment"])
    df["review"] = df["review"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df = df[df["review"] != ""]

    class_counts = df["sentiment"].value_counts()
    if len(class_counts) < 2:
        raise ValueError("The dataset must contain at least two sentiment classes.")
    if (class_counts < 2).any():
        raise ValueError(
            "Each sentiment class must have at least 2 samples for a stratified split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["sentiment"],
        test_size=0.25,
        random_state=42,
        stratify=df["sentiment"],
    )

    model = build_model_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Dataset size: {len(df)}")
    print(f"Class distribution: {class_counts.to_dict()}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
