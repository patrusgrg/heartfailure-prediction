import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import logging

logger = logging.getLogger(__name__)

def train_model(df, models_dir=None, test_size=0.2, random_state=42):
    """Train a RandomForest on the provided dataframe and save model with metrics.

    Args:
        df: pandas DataFrame containing features and target column.
        models_dir: optional path to directory where artifacts will be saved.
        test_size: proportion of data for test set (default 0.2).
        random_state: random seed for reproducibility.
        
    Returns:
        Tuple of (model_path, scaler_path, feature_path, metrics_path)
    """
    if models_dir is None:
        models_dir = os.path.join(os.getcwd(), 'models')

    os.makedirs(models_dir, exist_ok=True)

    # Detect target column from a list of common names
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
    
    logger.info(f"Dataset shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RandomForest
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    test_accuracy = model.score(X_test_scaled, y_test)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation scores
    cv_scores = cross_validate(
        model, X_train_scaled, y_train, 
        cv=5, 
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
    
    logger.info(f"Test Accuracy: {test_accuracy:.4f}, Test ROC-AUC: {test_roc_auc:.4f}")
    logger.info(f"CV Mean Accuracy: {cv_scores['test_accuracy'].mean():.4f} (+/- {cv_scores['test_accuracy'].std():.4f})")

    # Save artifacts
    model_path = os.path.join(models_dir, 'heart_failure_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    feature_path = os.path.join(models_dir, 'feature_columns.pkl')
    metrics_path = os.path.join(models_dir, 'metrics.json')

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(X.columns.tolist(), feature_path)

    # Save metrics
    metrics = {
        'test_accuracy': float(test_accuracy),
        'test_roc_auc': float(test_roc_auc),
        'test_set_size': len(y_test),
        'cv_accuracy_mean': float(cv_scores['test_accuracy'].mean()),
        'cv_accuracy_std': float(cv_scores['test_accuracy'].std()),
        'cv_precision_mean': float(cv_scores['test_precision'].mean()),
        'cv_recall_mean': float(cv_scores['test_recall'].mean()),
        'cv_f1_mean': float(cv_scores['test_f1'].mean()),
        'cv_roc_auc_mean': float(cv_scores['test_roc_auc'].mean()),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'feature_importance': dict(zip(X.columns, model.feature_importances_.tolist()))
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Model artifacts saved to {models_dir}")

    return model_path, scaler_path, feature_path, metrics_path


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
    
    logger.info(f'Training model on {df.shape[0]} samples with {df.shape[1]} features')
    model_path, scaler_path, feature_path, metrics_path = train_model(df)
    
    logger.info(f'✓ Model saved to {model_path}')
    logger.info(f'✓ Scaler saved to {scaler_path}')
    logger.info(f'✓ Features saved to {feature_path}')
    logger.info(f'✓ Metrics saved to {metrics_path}')
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
        print(f"CV Accuracy: {metrics['cv_accuracy_mean']:.4f} (+/- {metrics['cv_accuracy_std']:.4f})")
        print(f"CV Precision: {metrics['cv_precision_mean']:.4f}")
        print(f"CV Recall: {metrics['cv_recall_mean']:.4f}")
        print(f"CV F1: {metrics['cv_f1_mean']:.4f}")
    print("="*60 + "\n")
