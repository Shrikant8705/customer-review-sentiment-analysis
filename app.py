from pathlib import Path

import joblib
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.joblib"

st.set_page_config(page_title="Customer Sentiment Analysis", page_icon="💬")
st.title("💬 Customer Review Sentiment Analysis")
st.write("Enter a review to classify it as positive, negative, or neutral.")

if not MODEL_PATH.exists():
    st.warning("The model has not been trained yet.")
    st.code("python src/train.py", language="bash")
    st.stop()

model = joblib.load(MODEL_PATH)
review = st.text_area(
    "Customer review",
    placeholder="Example: The product quality is excellent!",
    height=140,
)

if st.button("Analyze sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        sentiment = model.predict([review])[0]
        probabilities = model.predict_proba([review])[0]
        confidence = float(max(probabilities))

        st.subheader(f"Sentiment: {sentiment.title()}")
        st.metric("Confidence", f"{confidence:.2%}")

        if sentiment == "positive":
            st.success("This review is positive.")
        elif sentiment == "negative":
            st.error("This review is negative.")
        else:
            st.info("This review is neutral.")
