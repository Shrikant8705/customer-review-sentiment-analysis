import re
from collections import Counter

ASPECT_KEYWORDS = {
    "product_quality": {"quality", "durable", "durability", "material", "build", "defect", "defective"},
    "price_value": {"price", "value", "expensive", "cheap", "cost", "worth", "pricing"},
    "delivery_shipping": {"delivery", "shipping", "ship", "arrived", "arrival", "courier", "late"},
    "packaging": {"packaging", "package", "box", "packed", "seal", "wrapping"},
    "customer_support_service": {"support", "service", "help", "agent", "staff", "customer service"},
    "usability": {"easy", "difficult", "usability", "setup", "interface", "use", "usable"},
    "reliability": {"reliable", "reliability", "stable", "crash", "fails", "failed", "issue", "issues"},
}

POSITIVE_CUES = {
    "good", "great", "excellent", "amazing", "fast", "quick", "helpful", "smooth", "easy", "reliable", "satisfied", "perfect", "love", "worth",
}
NEGATIVE_CUES = {
    "bad", "poor", "terrible", "awful", "slow", "late", "broken", "broke", "damaged", "expensive", "unhelpful", "hard", "difficult", "crash", "failed", "frustrating", "worst",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _clause_sentiment(tokens: list[str]) -> int:
    counts = Counter(tokens)
    positive = sum(counts[word] for word in POSITIVE_CUES if word in counts)
    negative = sum(counts[word] for word in NEGATIVE_CUES if word in counts)
    if positive > negative:
        return 1
    if negative > positive:
        return -1
    return 0


def detect_aspect_sentiment(review: str, aspect_keywords: dict[str, set[str]] | None = None) -> dict[str, str]:
    """Heuristic/demo aspect sentiment detection.

    Returns labels per configurable aspect:
    positive, negative, neutral, or not_mentioned.
    """
    text = review.lower()
    config = aspect_keywords or ASPECT_KEYWORDS
    aspect_scores = {aspect: [] for aspect in config}

    clauses = [c.strip() for c in re.split(r"[.!?;]|\bbut\b|\bhowever\b", text) if c.strip()]
    for clause in clauses:
        tokens = _tokenize(clause)
        clause_score = _clause_sentiment(tokens)
        for aspect, keywords in config.items():
            if any(keyword in clause for keyword in keywords):
                aspect_scores[aspect].append(clause_score)

    labels: dict[str, str] = {}
    for aspect, scores in aspect_scores.items():
        if not scores:
            labels[aspect] = "not_mentioned"
            continue
        score = sum(scores)
        if score > 0:
            labels[aspect] = "positive"
        elif score < 0:
            labels[aspect] = "negative"
        else:
            labels[aspect] = "neutral"

    return labels


def aggregate_aspect_scores(aspect_rows: list[dict[str, str]]) -> dict[str, float]:
    """Aggregate aspect sentiment into 0-10 demo scores."""
    aggregated: dict[str, float] = {}
    for aspect in ASPECT_KEYWORDS:
        mentioned = [row.get(aspect, "not_mentioned") for row in aspect_rows if row.get(aspect, "not_mentioned") != "not_mentioned"]
        if not mentioned:
            continue
        pos = sum(label == "positive" for label in mentioned)
        neg = sum(label == "negative" for label in mentioned)
        neu = sum(label == "neutral" for label in mentioned)
        score = ((pos * 10) + (neu * 5) + (neg * 0)) / len(mentioned)
        aggregated[aspect] = round(float(score), 2)
    return aggregated
