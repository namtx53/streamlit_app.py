import streamlit as st
import pandas as pd
import sys
sys.path.append(".")

from src.data.cleaner import clean_text
from src.features.builder import build_tfidf
from src.models.supervised import train_classifier

st.set_page_config(page_title="Hotel Mining", layout="wide")

st.title("🏨 Hotel Review Analysis")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview Data")
    st.dataframe(df.head())

    if "review" not in df.columns:
        st.error("Dataset must contain 'review' column")
    else:
        df["clean"] = df["review"].apply(clean_text)
        X, vec = build_tfidf(df["clean"])

        if "sentiment" in df.columns:
            model = train_classifier(X, df["sentiment"])
            st.success("Model trained successfully!")

            user_input = st.text_area("Enter a review to predict sentiment:")

            if user_input:
                clean = clean_text(user_input)
                X_input = vec.transform([clean])
                pred = model.predict(X_input)

                st.write("### Prediction:", pred[0])
        else:
            st.warning("No 'sentiment' column found for training.")