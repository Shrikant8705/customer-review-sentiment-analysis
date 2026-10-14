import re

EMOTION_LEXICON = {
    "anger": {"angry", "furious", "outraged", "annoyed", "mad"},
    "disappointment": {"disappointed", "letdown", "regret", "regretful"},
    "frustration": {"frustrated", "frustrating", "irritating", "unusable", "stuck"},
    "happiness": {"happy", "delighted", "joyful", "pleased"},
    "satisfaction": {"satisfied", "content", "works", "reliable", "solid"},
    "gratitude": {"grateful", "thankful", "thanks", "appreciate"},
    "admiration": {"amazing", "excellent", "impressive", "admire", "awesome"},
    "concern": {"concern", "worried", "uncertain", "risk", "issue"},
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def detect_emotions(review: str) -> dict[str, int]:
    """Local lexicon-based emotion estimates (demo heuristic)."""
    tokens = _tokens(review)
    emotions: dict[str, int] = {}
    for emotion, lexicon in EMOTION_LEXICON.items():
        score = sum(token in lexicon for token in tokens)
        if score > 0:
            emotions[emotion] = int(score)
    return emotions


def dominant_emotion(emotions: dict[str, int]) -> str:
    if not emotions:
        return "none"
    return max(sorted(emotions), key=lambda key: emotions[key])
