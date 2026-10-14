# Customer Sentiment Intelligence Dashboard

An original, local-first customer sentiment intelligence app built with Streamlit.

- Preserves individual review sentiment prediction.
- Adds CSV upload analytics with KPIs, trends, filters, and a priority queue.
- Adds transparent heuristic/demo layers for aspect sentiment and emotion estimates.

> ⚠️ Demo quality notice: model confidence is an estimate from a local TF-IDF + Logistic Regression model. Aspect/emotion outputs are lexicon/rule heuristics and should not be treated as production-grade NLP.

## Project structure

```text
customer-review-sentiment-analysis/
├── app.py
├── data/
│   └── reviews.csv
├── models/
├── requirements.txt
├── src/
│   ├── analytics.py
│   ├── aspect_sentiment.py
│   ├── emotions.py
│   ├── predict.py
│   ├── sentiment_model.py
│   └── train.py
└── tests/
    ├── conftest.py
    ├── test_dashboard_analytics.py
    └── test_predictions.py
```

## Setup (from repository root)

```bash
git clone https://github.com/Shrikant8705/customer-review-sentiment-analysis.git
cd customer-review-sentiment-analysis
python -m venv .venv
```

Activate environment:

- macOS/Linux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train model

```bash
python src/train.py
```

Creates `models/sentiment_model.joblib` locally.

## Run app

```bash
streamlit run app.py
```

## Run tests

```bash
pytest -q
```

## CSV schema

### Required

- `review` (case-insensitive match)

### Optional (auto-detected when present)

- date/time: `date`, `datetime`, `timestamp`, `created_at`, `time`
- product: `product`, `product_name`, `item`, `sku`
- category: `category`, `segment`, `department`
- rating: `rating`, `stars`, `score`
- customer: `customer`, `customer_id`, `user`, `client`

### Example CSV

```csv
review,date,product,category,rating,customer
"Delivery was fast, but product quality was terrible.",2026-10-01,Widget A,Gadgets,2,C001
"Excellent support and good value for money.",2026-10-02,Widget B,Gadgets,5,C002
"Packaging was fine, setup took time.",2026-10-03,Widget C,Accessories,3,C003
```

If `review` is missing, the app shows a clear validation error.

## Dashboard capabilities

- KPI cards:
  - total reviews
  - average sentiment score (0-10)
  - positive %
  - negative %
  - average confidence estimate
- Sentiment distribution chart (positive/neutral/negative)
- Time trend chart (when a usable date column is detected)
- Review explorer with filters:
  - sentiment
  - aspect tags
  - emotion tags
  - free-text search
  - optional product/category filters
- Highest-priority negative reviews list (transparent heuristic)

## Transparent heuristics

### Aspect sentiment (demo)

Configurable aspects:

- product quality
- price/value
- delivery/shipping
- packaging
- customer support/service
- usability
- reliability

Each review is tagged per aspect as `positive`, `negative`, `neutral`, or `not_mentioned`.

### Emotion estimates (demo)

Lexicon-based estimates for:

- anger
- disappointment
- frustration
- happiness
- satisfaction
- gratitude
- admiration
- concern

These are separate from model sentiment output.

### Priority queue heuristic (demo)

Priority increases with:

1. negative sentiment
2. higher model confidence
3. severity cue hits (e.g., refund, broken, damaged)

## Limitations

- Small local dataset and classic ML baseline can miss nuance (sarcasm, domain drift, mixed sentiment).
- Aspect and emotion outputs are intentionally transparent but simplistic heuristics.
- Use this dashboard for exploration/prototyping, not automated high-stakes decisions.

## Roadmap

- Replace heuristic aspect/emotion layers with stronger pretrained transformer models.
- Add calibration and monitoring for confidence quality.
- Add explainability views (token-level contributions) and richer trend diagnostics.
- Add ingestion/validation profiles per business domain.
