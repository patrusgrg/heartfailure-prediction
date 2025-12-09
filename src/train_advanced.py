"""
Advanced training with SHAP explanations and visualization.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import logging

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Run: pip install shap")


def train_with_explanations(df, models_dir=None, output_dir=None):
    """Train model and generate SHAP explanations.
    
    Args:
        df: pandas DataFrame with features and target
        models_dir: directory to save model artifacts
        output_dir: directory to save visualizations
        
    Returns:
        Tuple of (model_path, metrics_path, plots_dict)
    """
    if models_dir is None:
        models_dir = os.path.join(os.getcwd(), 'models')
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'reports', 'figures')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect target
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
    X = pd.get_dummies(X, drop_first=True)
    
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    test_accuracy = model.score(X_test_scaled, y_test)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation
    cv_scores = cross_validate(
        model, X_train_scaled, y_train, cv=StratifiedKFold(n_splits=5),
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], n_jobs=-1
    )
    
    # Save model artifacts
    model_path = os.path.join(models_dir, 'heart_failure_model_advanced.pkl')
    scaler_path = os.path.join(models_dir, 'scaler_advanced.pkl')
    feature_path = os.path.join(models_dir, 'feature_columns_advanced.pkl')
    metrics_path = os.path.join(models_dir, 'metrics_advanced.json')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(X.columns.tolist(), feature_path)
    
    # Save metrics
    metrics = {
        'test_accuracy': float(test_accuracy),
        'test_roc_auc': float(test_roc_auc),
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
    
    logger.info(f"Test Accuracy: {test_accuracy:.4f}, Test ROC-AUC: {test_roc_auc:.4f}")
    
    plots = {}
    
    # Feature Importance Plot
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([X.columns[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title('Top 15 Feature Importances')
        plt.tight_layout()
        importance_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['feature_importance'] = importance_path
        logger.info(f"✓ Saved feature importance plot to {importance_path}")
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {e}")
    
    # ROC Curve
    try:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC-AUC = {test_roc_auc:.4f}', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['roc_curve'] = roc_path
        logger.info(f"✓ Saved ROC curve to {roc_path}")
    except Exception as e:
        logger.error(f"Error creating ROC curve: {e}")
    
    # Confusion Matrix
    try:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['confusion_matrix'] = cm_path
        logger.info(f"✓ Saved confusion matrix to {cm_path}")
    except Exception as e:
        logger.error(f"Error creating confusion matrix: {e}")
    
    # SHAP Explanations (if available)
    if SHAP_AVAILABLE:
        try:
            logger.info("Generating SHAP explanations...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_scaled)
            
            # SHAP summary plot (for class 1 - heart failure)
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values[1], X_test_scaled, feature_names=X.columns, 
                            show=False, ax=ax)
            plt.tight_layout()
            shap_path = os.path.join(output_dir, 'shap_summary.png')
            plt.savefig(shap_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['shap_summary'] = shap_path
            logger.info(f"✓ Saved SHAP summary plot to {shap_path}")
        except Exception as e:
            logger.error(f"Error creating SHAP plots: {e}")
    else:
        logger.info("SHAP not available. Install with: pip install shap")
    
    logger.info(f"✓ Model artifacts saved to {models_dir}")
    logger.info(f"✓ Visualizations saved to {output_dir}")
    
    return model_path, metrics_path, plots


if __name__ == '__main__':
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(repo_root, 'data')
    
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
    
    train_with_explanations(df)
    print("\n✓ Training complete!")
