import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report

def evaluate_model(df, models_dir=None):
    if models_dir is None:
        models_dir = os.path.join(os.getcwd(), 'models')

    model_path = os.path.join(models_dir, 'heart_failure_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # detect target column
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

    # align features to training columns
    feature_path = os.path.join(models_dir, 'feature_columns.pkl')
    if os.path.exists(feature_path):
        feature_columns = joblib.load(feature_path)
        X = pd.get_dummies(X)
        X = X.reindex(columns=feature_columns, fill_value=0)

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    print(classification_report(y, y_pred))


if __name__ == '__main__':
    # CLI: load dataset and evaluate
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
        raise FileNotFoundError(f'No CSV data file found in {data_dir}')

    print('Loading data from', data_file)
    df = pd.read_csv(data_file)
    evaluate_model(df)
