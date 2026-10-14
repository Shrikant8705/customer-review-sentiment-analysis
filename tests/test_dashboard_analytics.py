import pandas as pd
import pytest

from src.analytics import (
    analyze_reviews_dataframe,
    filter_reviews,
    priority_negative_reviews,
    sentiment_score_0_to_10,
    validate_review_dataframe,
)
from src.aspect_sentiment import detect_aspect_sentiment
from src.emotions import detect_emotions


def test_csv_validation_requires_review_column():
    invalid_df = pd.DataFrame({"text_body": ["good"]})
    valid, message, review_column = validate_review_dataframe(invalid_df)

    assert not valid
    assert "review" in message.lower()
    assert review_column is None


def test_analysis_schema_includes_required_columns():
    df = pd.DataFrame(
        {
            "review": [
                "Delivery was fast, but product quality was terrible.",
                "Excellent support and great value.",
            ],
            "product": ["Widget A", "Widget B"],
            "category": ["Gadgets", "Gadgets"],
            "date": ["2026-10-01", "2026-10-02"],
        }
    )

    result = analyze_reviews_dataframe(df)

    expected_columns = {
        "review",
        "sentiment",
        "confidence",
        "sentiment_score_0_10",
        "aspect_labels",
        "emotion_scores",
        "priority_score",
        "product",
        "category",
    }
    assert expected_columns.issubset(set(result.columns))


def test_sentiment_score_mapping():
    score = sentiment_score_0_to_10({"positive": 0.2, "neutral": 0.4, "negative": 0.4})
    assert score == pytest.approx(4.0)


def test_aspect_detection_for_delivery_vs_quality():
    review = "Delivery was fast, but product quality was terrible."
    aspects = detect_aspect_sentiment(review)

    assert aspects["delivery_shipping"] == "positive"
    assert aspects["product_quality"] == "negative"


def test_emotion_detection_negative_and_positive_examples():
    neg = detect_emotions("I am angry and frustrated with this broken product")
    pos = detect_emotions("Thanks, I am happy and satisfied with this excellent product")

    assert "anger" in neg
    assert "frustration" in neg
    assert "gratitude" in pos
    assert "happiness" in pos


def test_filtering_and_priority_helpers():
    df = pd.DataFrame(
        {
            "review": [
                "Worst purchase ever, broken item and I want a refund",
                "Excellent value and very satisfied",
                "Average quality, okay experience",
            ],
            "product": ["A", "B", "C"],
            "category": ["X", "Y", "X"],
        }
    )

    analyzed = analyze_reviews_dataframe(df)
    filtered = filter_reviews(analyzed, sentiments=["negative"], search_text="refund")

    assert len(filtered) == 1
    assert filtered.iloc[0]["sentiment"] == "negative"

    prioritized = priority_negative_reviews(analyzed, limit=1)
    assert len(prioritized) == 1
    assert prioritized.iloc[0]["sentiment"] == "negative"
