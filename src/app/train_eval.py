import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from src.data.preprocess import load_dataset, train_test_split_dataset, save_feature_info
from src.models.pipelines import build_models, evaluate, save_model
from src.utils.io import save_json, ensure_dir


def train_all(csv_path: str, out_dir: str = "models") -> Dict[str, Dict[str, float]]:
    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y)

    feature_names = list(X.columns)
    save_feature_info(Path(out_dir) / "feature_info.json", feature_names)

    models = build_models(feature_names)
    metrics: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        m = evaluate(model, X_test, y_test)
        metrics[name] = m
        save_model(model, Path(out_dir) / f"{name}.joblib")

    save_json(Path(out_dir) / "metrics.json", metrics)
    return metrics


def train_single(csv_path: str, out_dir: str, model_name: str) -> Dict[str, Dict[str, float]]:
    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y)

    feature_names = list(X.columns)
    save_feature_info(Path(out_dir) / "feature_info.json", feature_names)

    models = build_models(feature_names)
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(models.keys())}")

    model = models[model_name]
    model.fit(X_train, y_train)
    m = evaluate(model, X_test, y_test)
    save_model(model, Path(out_dir) / f"{model_name}.joblib")

    # Merge/update metrics.json
    metrics_path = Path(out_dir) / "metrics.json"
    existing: Dict[str, Dict[str, float]] = {}
    if metrics_path.exists():
        try:
            import json

            with open(metrics_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing[model_name] = m
    save_json(metrics_path, existing)
    return {model_name: m}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate student depression models")
    parser.add_argument("--csv", type=str, default="Depression_Student_Dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--out", type=str, default="models", help="Output directory for models and metrics")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "svm", "random_forest", "logistic_regression"],
        help="Train either all models or a single model",
    )
    args = parser.parse_args()

    ensure_dir(args.out)
    if args.model == "all":
        metrics = train_all(args.csv, args.out)
        print(pd.DataFrame(metrics).T)
    else:
        metrics = train_single(args.csv, args.out, args.model)
        print(pd.DataFrame(metrics).T)


if __name__ == "__main__":
    main()


