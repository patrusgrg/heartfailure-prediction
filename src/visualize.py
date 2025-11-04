import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)


def find_data_file(repo_root):
    data_dir = os.path.join(repo_root, 'data')
    candidates = [
        os.path.join(data_dir, 'heart_failure_clinical_records_dataset.csv'),
        os.path.join(data_dir, 'heart.csv'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    import glob
    files = glob.glob(os.path.join(data_dir, '*.csv'))
    return files[0] if files else None


def load_artifacts(models_dir):
    model_path = os.path.join(models_dir, 'heart_failure_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    feature_path = os.path.join(models_dir, 'feature_columns.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(feature_path)
    return model, scaler, features


def make_dirs(path):
    os.makedirs(path, exist_ok=True)


def plot_target_distribution(y, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=y, ax=ax)
    ax.set_title('Target distribution')
    ax.set_xlabel('Target')
    ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_corr_heatmap(df, out_path):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature correlation heatmap')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importances(model, feature_columns, out_path, top_n=20):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        sel_feats = [feature_columns[i] for i in idx]
        vals = importances[idx]
        fig, ax = plt.subplots(figsize=(8, max(4, len(sel_feats) * 0.3)))
        sns.barplot(x=vals, y=sel_feats, ax=ax)
        ax.set_title('Top feature importances')
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion matrix')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc_pr(y_true, y_score, out_roc, out_pr):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_roc)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label='Precision-Recall')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall curve')
    fig.tight_layout()
    fig.savefig(out_pr)
    plt.close(fig)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_file = find_data_file(repo_root)
    if data_file is None:
        raise FileNotFoundError('No CSV found in data/ folder')

    models_dir = os.path.join(repo_root, 'models')
    model, scaler, feature_columns = load_artifacts(models_dir)

    df = pd.read_csv(data_file)
    # detect target
    possible_targets = ['DEATH_EVENT', 'HeartDisease', 'target', 'Outcome', 'DEATH']
    target_col = None
    for t in possible_targets:
        if t in df.columns:
            target_col = t
            break
    if target_col is None:
        raise KeyError(f'No target column found. Tried: {possible_targets}')

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X = pd.get_dummies(X)
    X = X.reindex(columns=feature_columns, fill_value=0)

    # predicted probabilities for ROC/PR and predicted labels
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(scaler.transform(X))[:, 1]
    else:
        # fallback to decision_function or predictions
        try:
            y_score = model.decision_function(scaler.transform(X))
        except Exception:
            y_score = model.predict(scaler.transform(X))

    y_pred = model.predict(scaler.transform(X))

    out_dir = os.path.join(repo_root, 'reports', 'figures')
    make_dirs(out_dir)

    # Target distribution
    td_path = os.path.join(out_dir, 'target_distribution.png')
    plot_target_distribution(y, td_path)

    # Correlation heatmap on numeric features (after dummy encoding)
    corr_path = os.path.join(out_dir, 'correlation_heatmap.png')
    plot_corr_heatmap(pd.concat([X, y.rename('target')], axis=1), corr_path)

    # Feature importances
    fi_path = os.path.join(out_dir, 'feature_importances.png')
    plot_feature_importances(model, feature_columns, fi_path)

    # Confusion matrix
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y, y_pred, cm_path)

    # ROC and PR
    roc_path = os.path.join(out_dir, 'roc_curve.png')
    pr_path = os.path.join(out_dir, 'pr_curve.png')
    plot_roc_pr(y, y_score, roc_path, pr_path)

    print('Saved figures to', out_dir)


if __name__ == '__main__':
    main()
