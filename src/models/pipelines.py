from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_numeric_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    """Create a ColumnTransformer that imputes and scales numeric features.

    Keeps only the listed numeric columns and drops the rest.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
    )
    return preprocessor


def build_models(numeric_features: list[str]) -> Dict[str, Pipeline]:
    """Build sklearn Pipelines that include preprocessing + classifier.

    Returns SVM, RandomForest, and LogisticRegression pipelines with class_weight
    set to "balanced" to mitigate class imbalance.
    """
    preprocessor = build_numeric_preprocessor(numeric_features)

    svm_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", SVC(kernel="rbf", probability=True, C=1.0, gamma="scale",
            random_state=42, class_weight="balanced")),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42,
            n_jobs=-1, class_weight="balanced")),
        ]
    )

    lr_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")),
        ]
    )

    return {
        "svm": svm_pipeline,
        "random_forest": rf_pipeline,
        "logistic_regression": lr_pipeline,
    }


def evaluate(model: Pipeline, X_test, y_test) -> Dict[str, float]:
    """Compute standard classification metrics for a fitted model.

    If the estimator exposes predict_proba, also compute ROC-AUC.
    """
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, proba)
        except Exception:
            pass
    return metrics


def save_model(model: Pipeline, out_path: str | Path) -> None:
    """Persist a fitted Pipeline to disk using joblib."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def load_model(path: str | Path) -> Pipeline:
    """Load a previously saved Pipeline from disk."""
    return joblib.load(path)


