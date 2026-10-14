from pathlib import Path
import streamlit as st

from src.predict import predict_with_details

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.joblib"

EXAMPLE_REVIEWS = [
    "The product is bad!",
    "Terrible product... broke in 2 days!!!",
    "Excellent product, worth every rupee.",
    "It is okay, nothing special.",
    "Not great, not terrible — just average.",
]

CARD_STYLE = {
    "positive": "#0f766e",
    "negative": "#b91c1c",
    "neutral": "#475569",
}

st.set_page_config(
    page_title="Customer Review Sentiment Studio",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    """
    <style>
      .subtitle {color:#475569; margin-top:-8px;}
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

st.title("💬 Customer Review Sentiment Studio")
st.markdown(
    "<p class='subtitle'>Analyze review tone as positive, negative, or neutral using a lightweight ML demo model.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Run setup")
    st.code("python src/train.py", language="bash")
    st.caption("Train the model after editing data/reviews.csv.")

    st.subheader("Confidence note")
    st.info(
        "Confidence is a model probability estimate from this demo dataset, not a guarantee of correctness."
    )

if not MODEL_PATH.exists():
    st.warning("No trained model found yet.")
    st.markdown("Train first with `python src/train.py` from the repository root.")
    st.stop()

if "review_input" not in st.session_state:
    st.session_state.review_input = ""

input_col, output_col = st.columns([1.15, 1.0], gap="large")

with input_col:
    st.subheader("Enter a customer review")
    st.session_state.review_input = st.text_area(
        "Review text",
        value=st.session_state.review_input,
        placeholder="Example: The product quality is excellent!",
        height=180,
        label_visibility="collapsed",
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
            "<div class='note-box'>✍️ Enter text or choose an example to see sentiment output.</div>",
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
                    "<div class='note-box'>⚠️ This app is a demonstration model trained on a small curated dataset. Use a larger, representative dataset for production decisions.</div>",
                    unsafe_allow_html=True,
                )
