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
        "Academic_Pressure",
        "Study_Satisfaction",
        "Sleep_Duration",
        "Dietary_Habit",
        "Financial_Stress",
        "Family_History",
    ]}


def convert_sleep_duration(sleep_str: str) -> float:
    """Convert sleep duration string to numeric value"""
    mapping = {
        "< 5": 4.5,
        "5-6": 5.5,
        "7-8": 7.5,
        "> 8": 9.0,
    }
    return mapping.get(sleep_str, 7.5)  # default to 7.5 if not found


def sidebar_inputs(features: list[str]) -> pd.DataFrame:
    st.sidebar.header("Input Features")
    inputs = {}
    
    # Gender: 0=Male, 1=Female
    inputs["Gender"] = st.sidebar.selectbox("Gender", ["Male", "Female"], index=0)
    inputs["Gender"] = 0 if inputs["Gender"] == "Male" else 1
    
    # Academic Pressure: 1.0 to 5.0
    inputs["Academic_Pressure"] = st.sidebar.slider("Academic Pressure (1-5)", 1.0, 5.0, 3.0, step=1.0)
    
    # Study Satisfaction: 1.0 to 5.0
    inputs["Study_Satisfaction"] = st.sidebar.slider("Study Satisfaction (1-5)", 1.0, 5.0, 3.0, step=1.0)
    
    # Sleep Duration: categorical values from dataset
    sleep_options = ["< 5", "5-6", "7-8", "> 8"]
    inputs["Sleep_Duration"] = st.sidebar.selectbox("Sleep Duration (Hours)", sleep_options, index=2)
    
    # Dietary Habit: 0=Healthy, 1=Moderate, 2=Unhealthy (use numeric values directly)
    dietary_options = ["Healthy (0)", "Moderate (1)", "Unhealthy (2)"]
    inputs["Dietary_Habit"] = st.sidebar.selectbox("Dietary Habit", dietary_options, index=1)
    inputs["Dietary_Habit"] = dietary_options.index(inputs["Dietary_Habit"])
    
    # Financial Stress: 0 to 5
    inputs["Financial_Stress"] = st.sidebar.slider("Financial Stress (0-5)", 0, 5, 2, step=1)
    
    # Family History: 0=No, 1=Yes
    family_options = ["No", "Yes"]
    inputs["Family_History"] = st.sidebar.selectbox("Family History of Mental Illness", family_options, index=0)
    inputs["Family_History"] = 0 if inputs["Family_History"] == "No" else 1
    
    # Convert sleep duration to numeric
    inputs["Sleep_Duration"] = convert_sleep_duration(inputs["Sleep_Duration"])
    
    row = {f: inputs.get(f, 0) for f in features}
    return pd.DataFrame([row])


def main():
    st.title("Student Mental Health Prediction")
    st.write("Predict likelihood of depression using SVM, Random Forest, and Logistic Regression.")
    st.write("Based on the Depression Student Dataset with features: Gender, Academic Pressure, Study Satisfaction, Sleep Duration, Dietary Habit, Financial Stress, and Family History.")

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


