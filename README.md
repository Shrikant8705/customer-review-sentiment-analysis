# Customer Review Sentiment Analysis

A small NLP project that classifies customer reviews as **positive**, **negative**, or **neutral** using TF-IDF features and logistic regression.

## Project structure

```text
customer-review-sentiment-analysis/
├── app.py
├── data/
│   └── reviews.csv
├── models/
├── requirements.txt
└── src/
    ├── predict.py
    └── train.py
```

## Requirements

- Python 3.10 or newer
- `pip`

## Setup

```bash
git clone https://github.com/Shrikant8705/customer-review-sentiment-analysis.git
cd customer-review-sentiment-analysis

python -m venv .venv
```

Activate the virtual environment:

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

```bash
python src/train.py
```

This creates `models/sentiment_model.joblib`.

## Run the Streamlit app

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Run command-line predictions

```bash
python src/predict.py
```

## Add your own data

Replace `data/reviews.csv` with a CSV containing these columns:

```csv
review,sentiment
"The product is excellent",positive
"The item arrived damaged",negative
"It is acceptable for the price",neutral
```

Then retrain the model:

```bash
python src/train.py
```

## Important note

The included dataset is intentionally small and is provided for demonstration. For a useful production model, train on a much larger, representative, and carefully labeled dataset.
