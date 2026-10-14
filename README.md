# Customer Review Sentiment Analysis

A lightweight sentiment analysis demo that classifies customer reviews as **positive**, **negative**, or **neutral**.

> ⚠️ This repository is a demo project. The included dataset is curated for local experimentation, not production-grade coverage.

## What's improved

- Larger, more balanced demo dataset in `data/reviews.csv` (positive/negative/neutral, short phrases, punctuation-heavy text)
- Stronger model pipeline in `src/train.py`:
  - custom text normalization with negation handling
  - word n-grams + character n-grams TF-IDF features
  - class-balanced Logistic Regression
- Better training output:
  - dataset size + class distribution
  - test accuracy
  - classification report
- Polished Streamlit UI with:
  - modern layout and sentiment result cards
  - confidence estimate display and per-class probabilities
  - cue-word explanation when available
  - example review buttons and empty/loading/error states
- Added tests for representative examples in `tests/test_predictions.py`

## Project structure

```text
customer-review-sentiment-analysis/
├── app.py
├── data/
│   └── reviews.csv
├── models/
├── requirements.txt
├── src/
│   ├── predict.py
│   ├── sentiment_model.py
│   └── train.py
└── tests/
    └── test_predictions.py
```

## Setup

```bash
git clone https://github.com/Shrikant8705/customer-review-sentiment-analysis.git
cd customer-review-sentiment-analysis

python -m venv .venv
```

Activate the environment:

### macOS/Linux

```bash
source .venv/bin/activate
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model

Run from repository root:

```bash
python src/train.py
```

This command creates `models/sentiment_model.joblib`.

## Run the Streamlit app

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Run tests

```bash
pytest -q
```

## Why the old prediction was unreliable

The earlier dataset was very small and sparse, so the model had weak coverage of obvious sentiment cues (for example, short negative phrases like "The product is bad!").

## Confidence note

The app shows a **model confidence estimate** (class probability from this trained classifier). It is helpful for ranking certainty, but it is **not a guarantee** that a prediction is correct.
