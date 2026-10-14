import re
from collections import OrderedDict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

SIGNAL_WORDS = {
    "positive": {
        "amazing",
        "awesome",
        "best",
        "excellent",
        "fantastic",
        "good",
        "great",
        "happy",
        "love",
        "perfect",
        "recommend",
        "reliable",
        "satisfied",
        "superb",
        "worth",
    },
    "negative": {
        "awful",
        "bad",
        "broke",
        "broken",
        "disappointed",
        "hate",
        "horrible",
        "late",
        "poor",
        "refund",
        "terrible",
        "unusable",
        "waste",
        "worst",
    },
    "neutral": {
        "acceptable",
        "average",
        "fine",
        "mixed",
        "normal",
        "okay",
        "ok",
        "standard",
        "typical",
    },
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"n['’]t\\b", " not", text)
    text = re.sub(r"['’]re\\b", " are", text)
    text = re.sub(r"['’]ll\\b", " will", text)
    text = re.sub(r"['’]ve\\b", " have", text)
    text = re.sub(r"['’]m\\b", " am", text)
    text = re.sub(r"[^a-z0-9!?.,'\\s-]", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def build_model_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    token_pattern=r"(?u)\\b[\\w']+\\b|[!?]+",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
        ]
    )

    classifier = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    return Pipeline(
        [
            ("features", features),
            ("classifier", classifier),
        ]
    )


def extract_signal_words(text: str) -> dict[str, list[str]]:
    tokens = re.findall(r"[a-z']+", normalize_text(text))
    matched: dict[str, list[str]] = {}

    for label, cues in SIGNAL_WORDS.items():
        unique = list(OrderedDict.fromkeys(token for token in tokens if token in cues))
        if unique:
            matched[label] = unique

    return matched
