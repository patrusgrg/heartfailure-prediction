import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import logging

logger = logging.getLogger(__name__)

def evaluate_model(df, models_dir=None):
    """Evaluate model on a dataset and print detailed metrics.
    
    Args:
        df: pandas DataFrame containing features and target column.
        models_dir: optional path to directory where model artifacts are stored.
    """
    if models_dir is None:
        models_dir = os.path.join(os.getcwd(), 'models')

    model_path = os.path.join(models_dir, 'heart_failure_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    metrics_path = os.path.join(models_dir, 'metrics.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Detect target column
    possible_targets = ['DEATH_EVENT', 'HeartDisease', 'target', 'Outcome', 'DEATH']
    target_col = None
    for t in possible_targets:
        if t in df.columns:
            target_col = t
            break
    if target_col is None:
        raise KeyError(f"No target column found. Tried: {possible_targets}")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Align features to training columns
    feature_path = os.path.join(models_dir, 'feature_columns.pkl')
    if os.path.exists(feature_path):
        feature_columns = joblib.load(feature_path)
        X = pd.get_dummies(X, drop_first=True)
        X = X.reindex(columns=feature_columns, fill_value=0)
    else:
        X = pd.get_dummies(X, drop_first=True)

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]

    # Compute metrics
    accuracy = model.score(X_scaled, y)
    roc_auc = roc_auc_score(y, y_pred_proba)
    
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    print(f"\nDataset: {len(y)} samples")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(y, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    # Print training metrics if available
    if os.path.exists(metrics_path):
        print("\n" + "-"*70)
        print("TRAINING METRICS (from training set):")
        print("-"*70)
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            print(f"Training Set Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"Training Set ROC-AUC: {metrics['test_roc_auc']:.4f}")
            print(f"CV Accuracy: {metrics['cv_accuracy_mean']:.4f} (+/- {metrics['cv_accuracy_std']:.4f})")
            print(f"CV Precision: {metrics['cv_precision_mean']:.4f}")
            print(f"CV Recall: {metrics['cv_recall_mean']:.4f}")
            print(f"CV F1: {metrics['cv_f1_mean']:.4f}")
            
            print("\nTop 10 Most Important Features:")
            feature_importance = metrics['feature_importance']
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:10]:
                print(f"  {feature}: {importance:.4f}")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Locate repository root and data directory
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(repo_root, 'data')
    
    # Find CSV file
    candidates = [
        os.path.join(data_dir, 'heart_failure_clinical_records_dataset.csv'),
        os.path.join(data_dir, 'heart.csv'),
    ]
    data_file = None
    for c in candidates:
        if os.path.exists(c):
            data_file = c
            break
    
    if data_file is None:
        import glob
        files = glob.glob(os.path.join(data_dir, '*.csv'))
        if files:
            data_file = files[0]
    
    if data_file is None:
        logger.error(f'No CSV data file found in {data_dir}')
        sys.exit(1)

    logger.info(f'Loading data from {data_file}')
    df = pd.read_csv(data_file)
    evaluate_model(df)
