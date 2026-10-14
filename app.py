from pathlib import Path

import pandas as pd
import streamlit as st

from src.analytics import (
    analyze_reviews_dataframe,
    aspect_distribution_scores,
    build_time_trend,
    compute_kpis,
    emotion_distribution,
    filter_reviews,
    load_csv_input,
    priority_negative_reviews,
    sentiment_distribution,
)
from src.predict import predict_with_details

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.joblib"

EXAMPLE_REVIEWS = [
    "The product is bad!",
    "Terrible product... broke in 2 days!!!",
    "Excellent product, worth every rupee.",
    "Delivery was fast, but product quality was terrible.",
]

CARD_STYLE = {
    "positive": "#0f766e",
    "negative": "#b91c1c",
    "neutral": "#475569",
}

st.set_page_config(
    page_title="Customer Sentiment Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
      .subtitle {color:#475569; margin-top:-8px;}
      .card {
        border-radius: 14px;
        padding: 14px;
        border: 1px solid #e2e8f0;
        background: #ffffff;
      }
      .sentiment-card {
        border-radius: 14px;
        padding: 18px;
        color: #ffffff;
        margin-top: 10px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.18);
      }
      .note-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Customer Sentiment Intelligence Dashboard")
st.markdown(
    "<p class='subtitle'>Original, local-first sentiment analytics with transparent heuristic demo layers for aspects and emotions.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Quick commands")
    st.code("python src/train.py", language="bash")
    st.code("pytest -q", language="bash")
    st.caption("Run from repository root.")
    st.info(
        "Sentiment confidence is a model estimate from a local demo classifier. Aspect and emotion analysis is heuristic/demo only."
    )

if not MODEL_PATH.exists():
    st.warning("No trained model found yet.")
    st.markdown("Train first with `python src/train.py` from the repository root.")
    st.stop()


@st.cache_data(show_spinner=False)
def _cached_batch_analysis(file_bytes: bytes) -> pd.DataFrame:
    source_df = load_csv_input(file_bytes)
    return analyze_reviews_dataframe(source_df)


tab_single, tab_dashboard = st.tabs(["Single Review Analysis", "CSV Intelligence Dashboard"])

with tab_single:
    if "review_input" not in st.session_state:
        st.session_state.review_input = ""

    input_col, output_col = st.columns([1.1, 1.0], gap="large")

    with input_col:
        st.subheader("Analyze one review")
        st.session_state.review_input = st.text_area(
            "Review text",
            value=st.session_state.review_input,
            placeholder="Example: Delivery was fast, but product quality was terrible.",
            height=180,
        )

        st.markdown("**Try examples**")
        button_cols = st.columns(2)
        for i, example in enumerate(EXAMPLE_REVIEWS):
            if button_cols[i % 2].button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.review_input = example
                st.rerun()

        analyze_clicked = st.button("Analyze sentiment", type="primary", use_container_width=True)

    with output_col:
        st.subheader("Result")
        if not st.session_state.review_input.strip() and not analyze_clicked:
            st.markdown(
                "<div class='note-box'>✍️ Enter text or choose an example to see sentiment, aspect cues, and emotion estimates.</div>",
                unsafe_allow_html=True,
            )

        if analyze_clicked:
            review = st.session_state.review_input.strip()
            if not review:
                st.error("Please enter a review before analyzing.")
            else:
                try:
                    with st.spinner("Analyzing review..."):
                        result = predict_with_details(review)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                else:
                    sentiment = str(result["sentiment"])
                    confidence = float(result["confidence"])
                    probabilities = result["probabilities"]
                    signals = result["signals"]

                    color = CARD_STYLE.get(sentiment, "#1d4ed8")
                    st.markdown(
                        f"""
                        <div class='sentiment-card' style='background:{color};'>
                          <h3 style='margin:0;'>Sentiment: {sentiment.title()}</h3>
                          <p style='margin:8px 0 0 0;'>Model confidence estimate: {confidence:.1%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.progress(min(max(confidence, 0.0), 1.0))
                    prob_cols = st.columns(3)
                    for i, label in enumerate(["positive", "neutral", "negative"]):
                        value = float(probabilities.get(label, 0.0))
                        prob_cols[i].metric(label.title(), f"{value:.1%}")

                    if signals:
                        flat_signals = []
                        for label in ["positive", "neutral", "negative"]:
                            words = signals.get(label)
                            if words:
                                flat_signals.append(f"{label}: {', '.join(words)}")
                        st.caption("Detected sentiment cues → " + " | ".join(flat_signals))
                    else:
                        st.caption("No clear cue words matched; prediction relies on broader text patterns.")

                    st.markdown(
                        "<div class='note-box'>Aspect and emotion layers are available in the CSV dashboard tab as transparent heuristic/demo analytics.</div>",
                        unsafe_allow_html=True,
                    )

with tab_dashboard:
    st.subheader("Upload reviews CSV")
    st.caption(
        "Required column: `review` (case-insensitive). Optional columns recognized when present: date/time, product, category, rating, customer."
    )
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], accept_multiple_files=False)

    if uploaded_file is None:
        st.info("Upload a CSV to generate dashboard analytics.")
    else:
        try:
            bytes_data = uploaded_file.getvalue()
            analysis_df = _cached_batch_analysis(bytes_data)
        except Exception as exc:
            st.error(str(exc))
        else:
            kpis = compute_kpis(analysis_df)

            kpi_cols = st.columns(5)
            kpi_cols[0].metric("Total Reviews", f"{kpis['total_reviews']}")
            kpi_cols[1].metric("Avg Sentiment (0-10)", f"{kpis['avg_sentiment_score']:.2f}")
            kpi_cols[2].metric("Positive %", f"{kpis['positive_pct']:.1f}%")
            kpi_cols[3].metric("Negative %", f"{kpis['negative_pct']:.1f}%")
            kpi_cols[4].metric("Avg Confidence", f"{kpis['avg_confidence']:.1f}%")

            chart_col_1, chart_col_2 = st.columns(2)
            with chart_col_1:
                st.markdown("**Sentiment distribution**")
                dist = sentiment_distribution(analysis_df)
                st.bar_chart(dist)
            with chart_col_2:
                st.markdown("**Emotion distribution (heuristic/demo)**")
                emotions = emotion_distribution(analysis_df)
                if emotions.empty:
                    st.info("No emotion cues detected in current data.")
                else:
                    st.bar_chart(emotions)

            aspect_scores = aspect_distribution_scores(analysis_df)
            st.markdown("**Aspect sentiment scores (heuristic/demo, 0-10)**")
            if aspect_scores.empty:
                st.info("No aspect mentions detected in current data.")
            else:
                st.bar_chart(aspect_scores)

            trend_df = build_time_trend(analysis_df)
            st.markdown("**Time trend**")
            if trend_df.empty:
                st.caption("No usable date/time column detected, so trend view is unavailable.")
            else:
                st.line_chart(trend_df.set_index("date")["avg_sentiment_score"])

            st.markdown("---")
            st.markdown("### Review explorer")

            filter_col_1, filter_col_2, filter_col_3, filter_col_4 = st.columns(4)
            selected_sentiments = filter_col_1.multiselect(
                "Sentiment filter",
                options=["positive", "neutral", "negative"],
            )

            aspect_options = sorted(
                {
                    aspect
                    for tags in analysis_df["aspect_tags"]
                    for aspect in tags
                }
            )
            selected_aspects = filter_col_2.multiselect("Aspect filter", options=aspect_options)

            emotion_options = sorted(
                {
                    emotion
                    for tags in analysis_df["emotion_tags"]
                    for emotion in tags
                }
            )
            selected_emotions = filter_col_3.multiselect("Emotion filter", options=emotion_options)

            search_text = filter_col_4.text_input("Search reviews", placeholder="Search text")

            product_selection: list[str] = []
            category_selection: list[str] = []
            if "product" in analysis_df.columns or "category" in analysis_df.columns:
                filter_col_5, filter_col_6 = st.columns(2)
                if "product" in analysis_df.columns:
                    product_values = sorted(analysis_df["product"].dropna().astype(str).unique().tolist())
                    product_selection = filter_col_5.multiselect("Product filter", options=product_values)
                if "category" in analysis_df.columns:
                    category_values = sorted(analysis_df["category"].dropna().astype(str).unique().tolist())
                    category_selection = filter_col_6.multiselect("Category filter", options=category_values)

            filtered_df = filter_reviews(
                analysis_df,
                search_text=search_text,
                sentiments=selected_sentiments,
                aspects=selected_aspects,
                emotions=selected_emotions,
                products=product_selection,
                categories=category_selection,
            )

            st.caption(f"Showing {len(filtered_df)} of {len(analysis_df)} reviews")

            explorer_columns = [
                "review",
                "sentiment",
                "sentiment_score_0_10",
                "confidence",
                "dominant_emotion",
                "priority_score",
            ]
            for optional in ["product", "category", "rating", "customer", "date"]:
                if optional in filtered_df.columns:
                    explorer_columns.append(optional)

            st.dataframe(filtered_df[explorer_columns], use_container_width=True)

            st.markdown("### Highest-priority negative reviews")
            st.caption(
                "Transparent priority heuristic = stronger negative sentiment + model confidence + severity cue hits (demo only)."
            )
            priority_df = priority_negative_reviews(filtered_df, limit=10)
            if priority_df.empty:
                st.success("No negative reviews match current filters.")
            else:
                st.dataframe(
                    priority_df[["review", "priority_score", "confidence", "severity_hits", "dominant_emotion"]],
                    use_container_width=True,
                )
