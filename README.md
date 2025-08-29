## Student Mental Health Prediction (Depression)

This project builds a supervised machine learning system to predict the likelihood of student depression using SVM, Random Forest, and Logistic Regression. It includes:

- Data preprocessing and feature engineering
- Model training and evaluation (accuracy, precision, recall, F1, ROC AUC)
- A Streamlit app for interactive predictions and viewing metrics

### Dataset
Place the dataset file at the project root (already included):
`Depression Student Dataset.csv`

### Requirements
- Python 3.10+ (Windows: install from Microsoft Store or `python.org`)
- Pip

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

### Train and Evaluate
This trains SVM, Random Forest, and Logistic Regression and saves models and metrics to `models/`.
```bash
python -m src.app.train_eval --csv "Depression Student Dataset.csv" --out models
```

Train a single model (e.g., Random Forest) only:
```bash
python -m src.app.train_eval --csv "Depression Student Dataset.csv" --out models --model random_forest
```
On success, you will see a metrics table printed and the following files created:
- `models/svm.joblib`
- `models/random_forest.joblib`
- `models/logistic_regression.joblib`
- `models/feature_info.json`
- `models/metrics.json`

### Run the Streamlit App
```bash
python -m streamlit run src/app/streamlit_app.py
```
In the sidebar, set feature values and choose a model to predict. If models aren’t found, run the training command above first.

### Project Structure
```
src/
  app/
    streamlit_app.py      # Streamlit UI
    train_eval.py         # CLI: train and evaluate models
  data/
    preprocess.py         # Loading, cleaning, splitting
  models/
    pipelines.py          # Pipelines for SVM, RF, LR
  utils/
    io.py                 # JSON and directory helpers
models/                    # Saved models and metrics (after training)
requirements.txt
Depression Student Dataset.csv
```

### Notes
- Sleep duration strings (e.g., "< 5", "5-6", "7-8", "> 8") are mapped to representative numeric values.
- Numeric features are imputed (median) and standardized within pipelines.
- Class probability is shown when the model supports it (`predict_proba`).
