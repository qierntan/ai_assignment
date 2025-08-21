from pathlib import Path
import json
import joblib
import streamlit as st
import pandas as pd


MODELS_DIR = Path("models")
FEATURE_INFO_PATH = MODELS_DIR / "feature_info.json"


@st.cache_resource
def load_models():
    models = {}
    for name in ["svm", "random_forest", "logistic_regression"]:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    return models


@st.cache_resource
def load_feature_info():
    if FEATURE_INFO_PATH.exists():
        with open(FEATURE_INFO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"features": [
        "Gender",
        "Age",
        "Academic_Pressure",
        "Study_Satisfaction",
        "Sleep_Duration",
        "Dietary_Habits",
        "Suicidal_Thoughts",
        "Study_Hours",
        "Financial_Stress",
        "Family_History",
    ]}


def sidebar_inputs(features: list[str]) -> pd.DataFrame:
    st.sidebar.header("Input Features")
    inputs = {}
    inputs["Gender"] = st.sidebar.selectbox("Gender (0=Male,1=Female)", [0, 1], index=0)
    inputs["Age"] = st.sidebar.number_input("Age", min_value=15, max_value=100, value=25)
    inputs["Academic_Pressure"] = st.sidebar.slider("Academic Pressure (1-5)", 1.0, 5.0, 3.0, step=1.0)
    inputs["Study_Satisfaction"] = st.sidebar.slider("Study Satisfaction (1-5)", 1.0, 5.0, 3.0, step=1.0)
    inputs["Sleep_Duration"] = st.sidebar.slider("Sleep Duration (hours)", 3.0, 10.0, 7.5, step=0.5)
    inputs["Dietary_Habits"] = st.sidebar.selectbox("Dietary Habits (0=Healthy,1=Moderate,2=Unhealthy)", [0, 1, 2], index=1)
    inputs["Suicidal_Thoughts"] = st.sidebar.selectbox("Suicidal Thoughts (0/1)", [0, 1], index=0)
    inputs["Study_Hours"] = st.sidebar.number_input("Daily Study Hours", min_value=0.0, max_value=16.0, value=4.0, step=0.5)
    inputs["Financial_Stress"] = st.sidebar.slider("Financial Stress (0-5)", 0.0, 5.0, 2.0, step=1.0)
    inputs["Family_History"] = st.sidebar.selectbox("Family History Mental Illness (0/1)", [0, 1], index=0)
    row = {f: inputs.get(f, 0) for f in features}
    return pd.DataFrame([row])


def main():
    st.title("Student Mental Health Prediction")
    st.write("Predict likelihood of depression using SVM, Random Forest, and Logistic Regression.")

    feature_info = load_feature_info()
    features = feature_info.get("features", [])
    models = load_models()

    if not models:
        st.warning("No trained models found. Please run training first: python -m src.app.train_eval --csv 'Depression Student Dataset.csv'")

    X_input = sidebar_inputs(features)

    st.subheader("Prediction")
    selected_model = st.selectbox("Model", list(models.keys()) or ["logistic_regression"])

    if st.button("Predict"):
        if selected_model in models:
            model = models[selected_model]
            proba = 0.5
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(X_input)[:, 1][0])
            pred = int(model.predict(X_input)[0])
            st.write(f"Predicted class: {'Depressed' if pred == 1 else 'Not Depressed'}")
            st.write(f"Probability of depression: {proba:.3f}")
        else:
            st.info("Selected model not available; please train and refresh.")

    # Show metrics if available
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        st.subheader("Model Metrics (Test Set)")
        st.dataframe(pd.DataFrame(metrics).T)


if __name__ == "__main__":
    main()


