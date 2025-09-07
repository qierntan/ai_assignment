from pathlib import Path
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from src.utils.io import save_json
import io


MODELS_DIR = Path("models")
FEATURE_INFO_PATH = MODELS_DIR / "feature_info.json"

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []


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


@st.cache_data
def load_full_dataset():
    """Load the full dataset for visualization and full dataset evaluation"""
    try:
        from src.data.preprocess import load_dataset
        X, y = load_dataset("Depression_Student_Dataset.csv")
        return X, y
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None, None


@st.cache_data
def evaluate_full_dataset():
    """Evaluate models on the full dataset (100%)"""
    X, y = load_full_dataset()
    if X is None or y is None:
        return {}
    
    # Load models directly in this function
    models = load_models()
    if not models:
        return {}
    
    full_metrics = {}
    for name, model in models.items():
        try:
            y_pred = model.predict(X)
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[:, 1]
            
            metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, zero_division=0),
                "recall": recall_score(y, y_pred, zero_division=0),
                "f1": f1_score(y, y_pred, zero_division=0),
            }
            if proba is not None:
                try:
                    from sklearn.metrics import roc_auc_score
                    metrics["roc_auc"] = roc_auc_score(y, proba)
                except Exception:
                    pass
            full_metrics[name] = metrics
        except Exception as e:
            st.error(f"Error evaluating {name} on full dataset: {e}")
    
    return full_metrics


def convert_sleep_duration(sleep_str: str) -> float:
    """Convert sleep duration string to numeric value"""
    mapping = {
        "< 5": 4.5,
        "5-6": 5.5,
        "7-8": 7.5,
        "> 8": 9.0,
    }
    return mapping.get(sleep_str, 7.5)  # default to 7.5 if not found


def get_input_features(features: list[str]) -> pd.DataFrame:
    """Get input features from user interface"""
    inputs = {}
    
    # Gender: 0=Male, 1=Female
    inputs["Gender"] = st.selectbox("Gender", ["Male", "Female"], index=0)
    inputs["Gender"] = 0 if inputs["Gender"] == "Male" else 1
    
    # Academic Pressure: 1.0 to 5.0
    inputs["Academic_Pressure"] = st.slider("Academic Pressure (1-5)", 1.0, 5.0, 3.0, step=1.0)
    
    # Study Satisfaction: 1.0 to 5.0
    inputs["Study_Satisfaction"] = st.slider("Study Satisfaction (1-5)", 1.0, 5.0, 3.0, step=1.0)
    
    # Sleep Duration: categorical values from dataset
    sleep_options = ["< 5", "5-6", "7-8", "> 8"]
    inputs["Sleep_Duration"] = st.selectbox("Sleep Duration (Hours)", sleep_options, index=2)
    
    # Dietary Habit: 0=Healthy, 1=Moderate, 2=Unhealthy (use numeric values directly)
    dietary_options = ["Healthy (0)", "Moderate (1)", "Unhealthy (2)"]
    inputs["Dietary_Habit"] = st.selectbox("Dietary Habit", dietary_options, index=1)
    inputs["Dietary_Habit"] = dietary_options.index(inputs["Dietary_Habit"])
    
    # Financial Stress: 0 to 5
    inputs["Financial_Stress"] = st.slider("Financial Stress (0-5)", 0, 5, 2, step=1)
    
    # Family History: 0=No, 1=Yes
    family_options = ["No", "Yes"]
    inputs["Family_History"] = st.selectbox("Family History of Mental Illness", family_options, index=0)
    inputs["Family_History"] = 0 if inputs["Family_History"] == "No" else 1
    
    # Convert sleep duration to numeric
    inputs["Sleep_Duration"] = convert_sleep_duration(inputs["Sleep_Duration"])
    
    row = {f: inputs.get(f, 0) for f in features}
    return pd.DataFrame([row])


def create_data_overview_visualizations(X, y):
    """Create data overview visualizations"""
    # Combine features and target for visualization
    df_viz = X.copy()
    df_viz['Depression'] = y
    
    # 1. Depression distribution
    fig1 = px.pie(
        values=df_viz['Depression'].value_counts().values,
        names=['Not Depressed', 'Depressed'],
        title='Distribution of Depression Cases'
    )
    
    # 2. Feature distributions by depression status
    fig2 = go.Figure()
    
    # Add box plots for each numeric feature
    for feature in X.columns:
        if feature != 'Depression':
            fig2.add_trace(go.Box(
                y=df_viz[feature],
                x=df_viz['Depression'].map({0: 'Not Depressed', 1: 'Depressed'}),
                name=feature.replace('_', ' '),
                boxpoints='outliers'
            ))
    
    fig2.update_layout(
        title='Feature Distributions by Depression Status',
        yaxis_title='Value',
        xaxis_title='Depression Status',
        height=600
    )
    
    # 3. Correlation heatmap
    corr_matrix = df_viz.corr()
    fig3 = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title='Feature Correlation Matrix',
        color_continuous_scale='RdBu_r'
    )
    
    # 4. Academic Pressure vs Study Satisfaction scatter plot
    fig4 = px.scatter(
        df_viz,
        x='Academic_Pressure',
        y='Study_Satisfaction',
        color='Depression',
        color_discrete_map={0: 'blue', 1: 'red'},
        title='Academic Pressure vs Study Satisfaction',
        labels={'Depression': 'Depression Status'}
    )
    
    return fig1, fig2, fig3, fig4


def create_metrics_comparison_chart(metrics_dict, metric_name, title):
    """Create comparison charts for metrics"""
    models = list(metrics_dict.keys())
    values = [metrics_dict[model].get(metric_name, 0) for model in models]
    
    fig = px.bar(
        x=models,
        y=values,
        title=title,
        labels={'x': 'Model', 'y': metric_name.upper()},
        color=values,
        color_continuous_scale='Viridis',
        text=[f'{val:.4f}' for val in values]  # Add text labels directly
    )
    
    # Update text position and styling
    fig.update_traces(
        textposition='outside',
        textfont_size=12,
        textfont_color='black'
    )
    
    # Calculate max value and set y-axis range to accommodate text labels
    max_value = max(values) if values else 1
    y_range_max = max_value * 1.15  # Add 15% more space above the highest bar
    
    # Adjust layout to prevent text cutoff
    fig.update_layout(
        margin=dict(t=100, b=80, l=60, r=60),  # Even more top margin
        height=500,  # Increase height further
        showlegend=False,  # Hide legend to save space
        yaxis=dict(
            range=[0, y_range_max],  # Set explicit y-axis range
            automargin=True  # Enable automatic margin adjustment
        )
    )
    
    return fig


def create_full_dataset_chart(metrics_dict, metric_name, title):
    """Create charts for full dataset performance"""
    models = list(metrics_dict.keys())
    values = [metrics_dict[model].get(metric_name, 0) for model in models]
    
    fig = px.bar(
        x=models,
        y=values,
        title=title,
        labels={'x': 'Model', 'y': metric_name.upper()},
        color=values,
        color_continuous_scale='Viridis',
        text=[f'{val:.4f}' for val in values]  # Add text labels directly
    )
    
    # Update text position and styling
    fig.update_traces(
        textposition='outside',
        textfont_size=12,
        textfont_color='black'
    )
    
    # Calculate max value and set y-axis range to accommodate text labels
    max_value = max(values) if values else 1
    y_range_max = max_value * 1.15  # Add 15% more space above the highest bar
    
    # Adjust layout to prevent text cutoff
    fig.update_layout(
        margin=dict(t=100, b=80, l=60, r=60),  # Even more top margin
        height=500,  # Increase height further
        showlegend=False,  # Hide legend to save space
        yaxis=dict(
            range=[0, y_range_max],  # Set explicit y-axis range
            automargin=True  # Enable automatic margin adjustment
        )
    )
    
    return fig


def add_prediction_to_history(model_name, inputs, prediction, probability):
    """Add prediction to history"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_record = {
        'timestamp': timestamp,
        'model': model_name,
        'inputs': inputs.copy(),
        'prediction': 'Depressed' if prediction == 1 else 'Not Depressed',
        'probability': probability
    }
    st.session_state.prediction_history.append(prediction_record)
    
    # Keep only last 50 predictions
    if len(st.session_state.prediction_history) > 50:
        st.session_state.prediction_history = st.session_state.prediction_history[-50:]


def convert_history_to_csv():
    """Convert prediction history to CSV format"""
    if not st.session_state.prediction_history:
        return None
    
    # Flatten the prediction history for CSV export with prioritized order
    csv_data = []
    for record in st.session_state.prediction_history:
        row = {
            'Timestamp': record['timestamp'],
            'Model': record['model'],
            'Prediction': record['prediction'],
            'Probability': record['probability']
        }
        # Add input features with capitalized names
        for feature, value in record['inputs'].items():
            # Convert feature names to proper case
            formatted_feature = feature.replace('_', ' ').title()
            row[formatted_feature] = value
        csv_data.append(row)
    
    # Convert to DataFrame and reorder columns to prioritize important information
    df = pd.DataFrame(csv_data)
    
    # Define column order - most important first
    priority_columns = ['Timestamp', 'Model', 'Prediction', 'Probability']
    feature_columns = [col for col in df.columns if col not in priority_columns]
    ordered_columns = priority_columns + feature_columns
    
    # Reorder DataFrame columns
    df = df[ordered_columns]
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


def train_models_interactive(test_size, random_state=42):
    """Train models with custom test size and return both train and test metrics"""
    try:
        from src.data.preprocess import load_dataset, train_test_split_dataset
        from src.models.pipelines import build_models, evaluate, save_model
        
        # Load dataset
        X, y = load_dataset("Depression_Student_Dataset.csv")
        
        # Custom train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state, stratify=y
        )
        
        # Build models
        feature_names = list(X.columns)
        models = build_models(feature_names)
        
        train_metrics = {}
        test_metrics = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate on training set
            train_metrics[name] = evaluate(model, X_train, y_train)
            
            # Evaluate on test set
            test_metrics[name] = evaluate(model, X_test, y_test)
            
            # Save model
            save_model(model, MODELS_DIR / f"{name}.joblib")
        
        # Save test metrics to file
        save_json(MODELS_DIR / "metrics.json", test_metrics)
        
        return train_metrics, test_metrics, len(X_train), len(X_test)
        
    except Exception as e:
        st.error(f"Error during training: {e}")
        return {}, {}, 0, 0


def get_feature_descriptions():
    """Get feature descriptions for the dataset"""
    return {
        "Gender": "Student's gender (0=Male, 1=Female)",
        "Academic_Pressure": "Level of academic pressure experienced (1-5 scale, 1=Very Low, 5=Very High)",
        "Study_Satisfaction": "Satisfaction level with current studies (1-5 scale, 1=Very Dissatisfied, 5=Very Satisfied)",
        "Sleep_Duration": "Average hours of sleep per night (4.5=<5hrs, 5.5=5-6hrs, 7.5=7-8hrs, 9.0=>8hrs)",
        "Dietary_Habit": "Quality of dietary habits (0=Healthy, 1=Moderate, 2=Unhealthy)",
        "Financial_Stress": "Level of financial stress (0-5 scale, 0=No Stress, 5=Extreme Stress)",
        "Family_History": "Family history of mental illness (0=No, 1=Yes)",
        "Depression": "Depression status (0=Not Depressed, 1=Depressed)"
    }


def main():
    # Set page config for wider layout
    st.set_page_config(
        page_title="Student Mental Health Prediction Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Student Mental Health Prediction Dashboard")
    st.write("Comprehensive analysis and prediction of student depression using machine learning models.")
    
    # Load data and models
    feature_info = load_feature_info()
    features = feature_info.get("features", [])
    models = load_models()
    X_full, y_full = load_full_dataset()
    
    if not models:
        st.warning("No trained models found. Please run training first: python -m src.app.train_eval --csv 'Depression_Student_Dataset.csv'")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Prediction", 
        "📊 Data Overview", 
        "🤖 Train Models", 
        "📈 Data Visualization", 
        "🎯 Full Dataset Performance"
    ])
    
    # Tab 1: Prediction with History
    with tab1:
        st.header("🔮 Make Predictions")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("📝 Input Features")
            X_input = get_input_features(features)
            
            st.subheader("🤖 Model Selection & Prediction")
            selected_model = st.selectbox("Select Model", list(models.keys()))
            
            if st.button("🔮 Predict", type="primary"):
                if selected_model in models:
                    model = models[selected_model]
                    proba = 0.5
                    if hasattr(model, "predict_proba"):
                        proba = float(model.predict_proba(X_input)[:, 1][0])
                    pred = int(model.predict(X_input)[0])
                    
                    # Display results
                    st.success(f"**Prediction:** {'Depressed' if pred == 1 else 'Not Depressed'}")
                    st.info(f"**Confidence:** {proba:.3f}")
                    
                    # Add to history
                    inputs_dict = X_input.iloc[0].to_dict()
                    add_prediction_to_history(selected_model, inputs_dict, pred, proba)
                else:
                    st.error("Selected model not available")
        
        with col2:
            st.subheader("📋 Prediction History")
            if st.session_state.prediction_history:
                # Add download button
                csv_data = convert_history_to_csv()
                if csv_data:
                    st.download_button(
                        label="📥 Download History as CSV",
                        data=csv_data,
                        file_name=f"Prediction History {datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download all prediction history as a CSV file"
                    )
                
                # Show last 10 predictions in expandable format
                recent_predictions = st.session_state.prediction_history[-10:]
                for pred in reversed(recent_predictions):
                    with st.expander(f"{pred['timestamp']} - {pred['model']}"):
                        st.write(f"**Result:** {pred['prediction']}")
                        st.write(f"**Probability:** {pred['probability']:.3f}")
                        st.write("**Inputs:**")
                        for key, value in pred['inputs'].items():
                            st.write(f"- {key}: {value}")
            else:
                st.info("No predictions made yet")
    
    # Tab 2: Data Overview
    with tab2:
        st.header("📊 Dataset Overview")
        
        if X_full is not None and y_full is not None:
            # Basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(X_full))
            with col2:
                st.metric("Features", len(X_full.columns))
            with col3:
                depression_rate = (y_full == 1).mean() * 100
                st.metric("Depression Rate", f"{depression_rate:.1f}%")
            
            # Feature descriptions
            st.subheader("📋 Feature Descriptions")
            feature_descriptions = get_feature_descriptions()
            
            # Create a nice table for feature descriptions
            desc_data = []
            for feature in X_full.columns:
                if feature in feature_descriptions:
                    desc_data.append({
                        'Feature': feature,
                        'Description': feature_descriptions[feature]
                    })
            
            if desc_data:
                desc_df = pd.DataFrame(desc_data)
                st.dataframe(desc_df, use_container_width=True)
            
            # Visualizations
            st.subheader("📈 Data Visualizations")
            fig1, fig2, fig3, fig4 = create_data_overview_visualizations(X_full, y_full)
            
            st.plotly_chart(fig1, use_container_width=True, key="data_overview_pie")
            st.plotly_chart(fig2, use_container_width=True, key="data_overview_box")
            st.plotly_chart(fig3, use_container_width=True, key="data_overview_corr")
            st.plotly_chart(fig4, use_container_width=True, key="data_overview_scatter")
            
            # Data table
            st.subheader("📊 Sample Data")
            df_sample = X_full.copy()
            df_sample['Depression'] = y_full
            st.dataframe(df_sample.head(10))
        else:
            st.error("Could not load dataset for visualization")
    
    # Tab 3: Interactive Model Training
    with tab3:
        st.header("🤖 Interactive Model Training")
        st.write("Train models with custom train-test split and compare performance on both training and testing sets.")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Training Configuration")
            test_size = st.slider(
                "Test Size (%)", 
                min_value=10, 
                max_value=50, 
                value=30, 
                step=5,
                help="Percentage of data to use for testing"
            )
            train_size = 100 - test_size
            
            st.metric("Training Size", f"{train_size}%")
            st.metric("Testing Size", f"{test_size}%")
            
            random_state = st.number_input(
                "Random State", 
                min_value=0, 
                max_value=1000, 
                value=42,
                help="Random seed for reproducible results"
            )
            
            if st.button("🚀 Train Models", type="primary"):
                with st.spinner("Training models..."):
                    train_metrics, test_metrics, train_samples, test_samples = train_models_interactive(test_size, random_state)
                    
                    if train_metrics and test_metrics:
                        st.success("✅ Models trained successfully!")
                        
                        # Store metrics in session state for display
                        st.session_state.train_metrics = train_metrics
                        st.session_state.test_metrics = test_metrics
                        st.session_state.train_samples = train_samples
                        st.session_state.test_samples = test_samples
        
        with col2:
            if 'train_metrics' in st.session_state and 'test_metrics' in st.session_state:
                st.subheader("📊 Training Results")
                
                # Show sample counts
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Training Samples", st.session_state.train_samples)
                with col_b:
                    st.metric("Testing Samples", st.session_state.test_samples)
                
                # Training metrics table
                st.subheader("📈 Training Set Performance")
                train_df = pd.DataFrame(st.session_state.train_metrics).T
                st.dataframe(train_df, use_container_width=True)
                
                # Test metrics table
                st.subheader("📉 Testing Set Performance")
                test_df = pd.DataFrame(st.session_state.test_metrics).T
                st.dataframe(test_df, use_container_width=True)
                
                # Performance comparison charts
                st.subheader("📊 Performance Comparison")
                
                # Create comparison charts
                metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
                
                for metric in metrics_to_compare:
                    if metric in train_df.columns and metric in test_df.columns:
                        fig = go.Figure()
                        
                        models = list(train_df.index)
                        train_values = [train_df.loc[model, metric] for model in models]
                        test_values = [test_df.loc[model, metric] for model in models]
                        
                        fig.add_trace(go.Bar(
                            name='Training',
                            x=models,
                            y=train_values,
                            marker_color='lightblue',
                            text=[f'{val:.4f}' for val in train_values],
                            textposition='outside'
                        ))
                        
                        fig.add_trace(go.Bar(
                            name='Testing',
                            x=models,
                            y=test_values,
                            marker_color='lightcoral',
                            text=[f'{val:.4f}' for val in test_values],
                            textposition='outside'
                        ))
                        
                        fig.update_layout(
                            title=f'{metric.upper()} Comparison (Training vs Testing)',
                            xaxis_title='Model',
                            yaxis_title=metric.upper(),
                            barmode='group',
                            height=500,
                            margin=dict(t=100, b=80, l=60, r=60)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key=f"train_tab_{metric}_v2")
            else:
                st.info("👆 Configure training parameters and click 'Train Models' to see results")
    
    # Tab 4: Data Visualization
    with tab4:
        st.header("📈 Data Visualization & Model Performance")
        
        # Check if we have training results from the interactive training tab
        if 'train_metrics' in st.session_state and 'test_metrics' in st.session_state:
            st.subheader("📊 Training Results (Interactive Training)")
            
            # Show sample counts
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Training Samples", st.session_state.train_samples)
            with col_b:
                st.metric("Testing Samples", st.session_state.test_samples)
            
            # Training metrics table
            st.subheader("📈 Training Set Performance")
            train_df = pd.DataFrame(st.session_state.train_metrics).T
            st.dataframe(train_df, use_container_width=True)
            
            # Test metrics table
            st.subheader("📉 Testing Set Performance")
            test_df = pd.DataFrame(st.session_state.test_metrics).T
            st.dataframe(test_df, use_container_width=True)
            
            # Performance comparison charts
            st.subheader("📊 Performance Comparison (Training vs Testing)")
            
            # Create comparison charts for all metrics
            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
            
            for metric in metrics_to_compare:
                if metric in train_df.columns and metric in test_df.columns:
                    fig = go.Figure()
                    
                    models = list(train_df.index)
                    train_values = [train_df.loc[model, metric] for model in models]
                    test_values = [test_df.loc[model, metric] for model in models]
                    
                    fig.add_trace(go.Bar(
                        name='Training',
                        x=models,
                        y=train_values,
                        marker_color='lightblue',
                        text=[f'{val:.4f}' for val in train_values],
                        textposition='outside'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Testing',
                        x=models,
                        y=test_values,
                        marker_color='lightcoral',
                        text=[f'{val:.4f}' for val in test_values],
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title=f'{metric.upper()} Comparison (Training vs Testing)',
                        xaxis_title='Model',
                        yaxis_title=metric.upper(),
                        barmode='group',
                        height=500,
                        margin=dict(t=100, b=80, l=60, r=60)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key=f"data_viz_train_test_{metric}_v2")
        
        # Also show the standard metrics from file if available
        metrics_path = MODELS_DIR / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                test_metrics = json.load(f)
            
            if 'train_metrics' not in st.session_state:
                st.subheader("📈 Standard Model Performance (70% Training, 30% Testing)")
                
                # Display metrics table
                st.subheader("Performance Metrics")
                metrics_df = pd.DataFrame(test_metrics).T
                st.dataframe(metrics_df, use_container_width=True)
                
                # Create comparison charts
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    fig_acc = create_metrics_comparison_chart(test_metrics, 'accuracy', 'Accuracy Comparison')
                    st.plotly_chart(fig_acc, use_container_width=True, key="data_viz_accuracy_v2")
                    
                    fig_f1 = create_metrics_comparison_chart(test_metrics, 'f1', 'F1-Score Comparison')
                    st.plotly_chart(fig_f1, use_container_width=True, key="data_viz_f1_v2")
                
                with col2:
                    fig_prec = create_metrics_comparison_chart(test_metrics, 'precision', 'Precision Comparison')
                    st.plotly_chart(fig_prec, use_container_width=True, key="data_viz_precision_v2")
                    
                    fig_rec = create_metrics_comparison_chart(test_metrics, 'recall', 'Recall Comparison')
                    st.plotly_chart(fig_rec, use_container_width=True, key="data_viz_recall_v2")
            
        else:
            st.warning("No metrics found. Please train models first using the 'Train Models' tab.")
    
    # Tab 5: Full Dataset Performance
    with tab5:
        st.header("🎯 Full Dataset Performance (100%)")
        
        if st.button("🔄 Evaluate on Full Dataset", type="primary"):
            with st.spinner("Evaluating models on full dataset..."):
                full_metrics = evaluate_full_dataset()
                
                if full_metrics:
                    st.subheader("📊 Full Dataset Performance Metrics")
                    full_metrics_df = pd.DataFrame(full_metrics).T
                    st.dataframe(full_metrics_df, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("📈 Performance Visualization")
                    
                    # Comparison charts
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        fig_full_acc = create_full_dataset_chart(full_metrics, 'accuracy', 'Full Dataset Accuracy')
                        st.plotly_chart(fig_full_acc, use_container_width=True, key="full_dataset_accuracy_v2")
                        
                    
                    with col2:
                        fig_full_f1 = create_full_dataset_chart(full_metrics, 'f1', 'Full Dataset F1-Score')
                        st.plotly_chart(fig_full_f1, use_container_width=True, key="full_dataset_f1_v2")
                        
                else:
                    st.error("Failed to evaluate models on full dataset")
        else:
            st.info("Click the button above to evaluate models on the full dataset (100%)")


if __name__ == "__main__":
    main()


