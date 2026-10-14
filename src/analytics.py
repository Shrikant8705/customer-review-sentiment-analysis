from __future__ import annotations

from collections import Counter
from io import BytesIO

import pandas as pd

from src.aspect_sentiment import ASPECT_KEYWORDS, aggregate_aspect_scores, detect_aspect_sentiment
from src.emotions import detect_emotions, dominant_emotion
from src.predict import predict_with_details

REVIEW_COLUMN_CANDIDATES = {"review", "reviews", "text", "comment", "feedback"}
OPTIONAL_COLUMN_CANDIDATES = {
    "date": {"date", "datetime", "timestamp", "created_at", "time"},
    "product": {"product", "product_name", "item", "sku"},
    "category": {"category", "segment", "department"},
    "rating": {"rating", "stars", "score"},
    "customer": {"customer", "customer_id", "user", "client"},
}
SEVERITY_CUES = {"refund", "cancel", "lawsuit", "unacceptable", "broken", "worst", "angry", "frustrating", "damaged"}


def _canonical(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _find_column(columns: list[str], candidates: set[str]) -> str | None:
    canonical_map = {_canonical(column): column for column in columns}
    for candidate in candidates:
        if candidate in canonical_map:
            return canonical_map[candidate]
    return None


def load_csv_input(uploaded_content: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(uploaded_content))


def validate_review_dataframe(df: pd.DataFrame) -> tuple[bool, str, str | None]:
    review_column = _find_column(list(df.columns), REVIEW_COLUMN_CANDIDATES)
    if not review_column:
        return False, "CSV must include a `review` column (case-insensitive).", None
    return True, "CSV looks valid.", review_column


def sentiment_score_0_to_10(probabilities: dict[str, float]) -> float:
    return float(
        (probabilities.get("positive", 0.0) * 10.0)
        + (probabilities.get("neutral", 0.0) * 5.0)
        + (probabilities.get("negative", 0.0) * 0.0)
    )


def _optional_column_map(columns: list[str]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for key, candidates in OPTIONAL_COLUMN_CANDIDATES.items():
        column = _find_column(columns, candidates)
        if column:
            mapped[key] = column
    return mapped


def _priority_score(sentiment: str, confidence: float, severity_hits: int) -> float:
    negative_factor = 1.0 if sentiment == "negative" else 0.2
    return round((negative_factor * 0.6) + (confidence * 0.25) + (min(severity_hits, 3) * 0.15), 4)


def analyze_reviews_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    valid, message, review_column = validate_review_dataframe(df)
    if not valid or not review_column:
        raise ValueError(message)

    optional_columns = _optional_column_map(list(df.columns))

    work_df = df.copy()
    work_df[review_column] = work_df[review_column].astype(str).str.strip()
    work_df = work_df[work_df[review_column] != ""].reset_index(drop=True)
    if work_df.empty:
        raise ValueError("CSV has no non-empty reviews to analyze.")

    if "date" in optional_columns:
        work_df["parsed_date"] = pd.to_datetime(work_df[optional_columns["date"]], errors="coerce")

    rows: list[dict[str, object]] = []
    for _, row in work_df.iterrows():
        review_text = str(row[review_column])
        prediction = predict_with_details(review_text)
        aspect_labels = detect_aspect_sentiment(review_text)
        emotion_scores = detect_emotions(review_text)
        dominant = dominant_emotion(emotion_scores)
        review_tokens = set(review_text.lower().split())
        severity_hits = sum(cue in review_tokens for cue in SEVERITY_CUES)

        item = {
            "review": review_text,
            "sentiment": prediction["sentiment"],
            "confidence": float(prediction["confidence"]),
            "sentiment_score_0_10": sentiment_score_0_to_10(prediction["probabilities"]),
            "positive_probability": float(prediction["probabilities"].get("positive", 0.0)),
            "neutral_probability": float(prediction["probabilities"].get("neutral", 0.0)),
            "negative_probability": float(prediction["probabilities"].get("negative", 0.0)),
            "signals": prediction["signals"],
            "aspect_labels": aspect_labels,
            "aspect_tags": [aspect for aspect, label in aspect_labels.items() if label != "not_mentioned"],
            "emotion_scores": emotion_scores,
            "emotion_tags": list(emotion_scores.keys()),
            "dominant_emotion": dominant,
            "severity_hits": int(severity_hits),
        }

        for aspect, label in aspect_labels.items():
            item[f"aspect_{aspect}"] = label

        item["priority_score"] = _priority_score(item["sentiment"], item["confidence"], item["severity_hits"])

        for optional_key, column_name in optional_columns.items():
            item[optional_key] = row[column_name]

        if "parsed_date" in work_df.columns:
            item["parsed_date"] = row.get("parsed_date")

        rows.append(item)

    return pd.DataFrame(rows)


def compute_kpis(analysis_df: pd.DataFrame) -> dict[str, float]:
    total = int(len(analysis_df))
    if total == 0:
        return {
            "total_reviews": 0,
            "avg_sentiment_score": 0.0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "avg_confidence": 0.0,
        }

    positive = float((analysis_df["sentiment"] == "positive").mean() * 100)
    negative = float((analysis_df["sentiment"] == "negative").mean() * 100)
    return {
        "total_reviews": total,
        "avg_sentiment_score": float(analysis_df["sentiment_score_0_10"].mean()),
        "positive_pct": positive,
        "negative_pct": negative,
        "avg_confidence": float(analysis_df["confidence"].mean() * 100),
    }


def sentiment_distribution(analysis_df: pd.DataFrame) -> pd.Series:
    return analysis_df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)


def emotion_distribution(analysis_df: pd.DataFrame) -> pd.Series:
    counts = Counter()
    for tags in analysis_df["emotion_tags"]:
        counts.update(tags)
    if not counts:
        return pd.Series(dtype="int64")
    return pd.Series(dict(counts)).sort_values(ascending=False)


def aspect_distribution_scores(analysis_df: pd.DataFrame) -> pd.Series:
    scores = aggregate_aspect_scores(analysis_df["aspect_labels"].tolist())
    if not scores:
        return pd.Series(dtype="float64")
    return pd.Series(scores).sort_values(ascending=False)


def build_time_trend(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if "parsed_date" not in analysis_df.columns:
        return pd.DataFrame()

    time_df = analysis_df.dropna(subset=["parsed_date"]).copy()
    if time_df.empty:
        return pd.DataFrame()

    time_df["date"] = pd.to_datetime(time_df["parsed_date"]).dt.date
    grouped = (
        time_df.groupby("date", as_index=False)
        .agg(avg_sentiment_score=("sentiment_score_0_10", "mean"), review_count=("review", "count"))
        .sort_values("date")
    )
    return grouped


def filter_reviews(
    analysis_df: pd.DataFrame,
    search_text: str = "",
    sentiments: list[str] | None = None,
    aspects: list[str] | None = None,
    emotions: list[str] | None = None,
    products: list[str] | None = None,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    filtered = analysis_df.copy()

    if sentiments:
        filtered = filtered[filtered["sentiment"].isin(sentiments)]

    if search_text:
        needle = search_text.strip().lower()
        if needle:
            filtered = filtered[filtered["review"].str.lower().str.contains(needle, na=False)]

    if aspects:
        filtered = filtered[
            filtered["aspect_tags"].apply(lambda tags: any(aspect in tags for aspect in aspects))
        ]

    if emotions:
        filtered = filtered[
            filtered["emotion_tags"].apply(lambda tags: any(emotion in tags for emotion in emotions))
        ]

    if products and "product" in filtered.columns:
        filtered = filtered[filtered["product"].isin(products)]

    if categories and "category" in filtered.columns:
        filtered = filtered[filtered["category"].isin(categories)]

    return filtered.reset_index(drop=True)


def priority_negative_reviews(analysis_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    negatives = analysis_df[analysis_df["sentiment"] == "negative"].copy()
    if negatives.empty:
        return negatives
    return negatives.sort_values(["priority_score", "confidence"], ascending=[False, False]).head(limit)
