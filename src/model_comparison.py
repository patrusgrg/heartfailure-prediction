"""
Compare multiple classification models for heart failure prediction.
Tests Logistic Regression, Random Forest, Gradient Boosting, and SVM.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import logging
import joblib

logger = logging.getLogger(__name__)

def compare_models(df, models_dir=None, cv_folds=5):
    """Compare multiple models and return results.
    
    Args:
        df: pandas DataFrame with features and target
        models_dir: directory to save results
        cv_folds: number of cross-validation folds
        
    Returns:
        Dictionary with comparison results
    """
    if models_dir is None:
        models_dir = os.path.join(os.getcwd(), 'models', 'comparisons')
    
    os.makedirs(models_dir, exist_ok=True)
    
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
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_split=5, 
            min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, 
            random_state=42, subsample=0.8
        ),
        'Support Vector Machine': SVC(
            kernel='rbf', C=1.0, probability=True, random_state=42, class_weight='balanced'
        )
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Test set metrics
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        test_acc = model.score(X_test_scaled, y_test)
        test_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_validate(
            model, X_train_scaled, y_train, cv=StratifiedKFold(n_splits=cv_folds),
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            n_jobs=-1
        )
        
        results[name] = {
            'test_accuracy': float(test_acc),
            'test_roc_auc': float(test_roc),
            'cv_accuracy': {
                'mean': float(cv_scores['test_accuracy'].mean()),
                'std': float(cv_scores['test_accuracy'].std())
            },
            'cv_precision': {
                'mean': float(cv_scores['test_precision'].mean()),
                'std': float(cv_scores['test_precision'].std())
            },
            'cv_recall': {
                'mean': float(cv_scores['test_recall'].mean()),
                'std': float(cv_scores['test_recall'].std())
            },
            'cv_f1': {
                'mean': float(cv_scores['test_f1'].mean()),
                'std': float(cv_scores['test_f1'].std())
            },
            'cv_roc_auc': {
                'mean': float(cv_scores['test_roc_auc'].mean()),
                'std': float(cv_scores['test_roc_auc'].std())
            },
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"  Test Accuracy: {test_acc:.4f}, Test ROC-AUC: {test_roc:.4f}")
        logger.info(f"  CV Accuracy: {results[name]['cv_accuracy']['mean']:.4f} (+/- {results[name]['cv_accuracy']['std']:.4f})")
    
    # Save results
    results_path = os.path.join(models_dir, 'model_comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
    
    return results


def print_comparison_table(results):
    """Print a nice comparison table of model results."""
    print("\n" + "="*100)
    print("MODEL COMPARISON RESULTS")
    print("="*100)
    
    print(f"{'Model':<30} {'Test Acc':<12} {'Test ROC-AUC':<15} {'CV Acc (mean)':<18} {'CV F1 (mean)':<15}")
    print("-"*100)
    
    for name, metrics in results.items():
        print(f"{name:<30} {metrics['test_accuracy']:<12.4f} {metrics['test_roc_auc']:<15.4f} "
              f"{metrics['cv_accuracy']['mean']:<18.4f} {metrics['cv_f1']['mean']:<15.4f}")
    
    print("="*100)
    
    # Find best models
    best_acc = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    best_roc = max(results.items(), key=lambda x: x[1]['test_roc_auc'])
    
    print(f"\n✓ Best Test Accuracy: {best_acc[0]} ({best_acc[1]['test_accuracy']:.4f})")
    print(f"✓ Best Test ROC-AUC: {best_roc[0]} ({best_roc[1]['test_roc_auc']:.4f})")
    print("="*100 + "\n")


if __name__ == '__main__':
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Locate data
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
    
    results = compare_models(df)
    print_comparison_table(results)
