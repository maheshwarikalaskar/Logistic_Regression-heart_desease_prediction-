from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
DATA_CANDIDATES = [BASE_DIR / "heart.csv", BASE_DIR / "data" / "heart.csv"]
TARGET = "target"

st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀")
st.title("🫀 Heart Disease Prediction")
st.caption("Educational demo of a logistic regression model. It is not a medical device and must not be used for diagnosis.")

data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
if data_path is None:
    st.error("heart.csv was not found next to app.py.")
    st.stop()


@st.cache_resource
def train_model(path: str):
    """Fit scaler + logistic regression on the dataset. Takes well under a second."""
    df = pd.read_csv(path)
    X, y = df.drop(columns=TARGET), df[TARGET]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X, y)
    return model, X


model, X = train_model(str(data_path))

st.sidebar.header("Patient details")
values = {}
for col in X.columns:
    series = X[col]
    if series.nunique() <= 5:  # coded categories (sex, cp, fbs, ...): offer only values seen in training
        options = sorted(series.unique().tolist())
        values[col] = st.sidebar.selectbox(col, options)
    elif pd.api.types.is_float_dtype(series):
        values[col] = st.sidebar.number_input(
            col, float(series.min()), float(series.max()), float(series.median()), step=0.1
        )
    else:
        values[col] = st.sidebar.number_input(
            col, int(series.min()), int(series.max()), int(series.median()), step=1
        )

input_df = pd.DataFrame([values])[list(X.columns)]
st.subheader("Values entered")
st.dataframe(input_df, hide_index=True)

if st.button("Predict"):
    class_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(input_df)[0, class_index])
    st.subheader("Model output")
    st.metric("Estimated probability of class 1 (the positive label in the training data)", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption("This is a statistical estimate from a small public dataset, not medical advice.")
