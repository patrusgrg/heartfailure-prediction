import os
import joblib
import pandas as pd

def predict(input_data, models_dir=None):
    """Predict a single sample.

    input_data: dict or pandas Series/DataFrame row
    models_dir: optional folder where model/scaler are stored
    """
    if models_dir is None:
        models_dir = os.path.join(os.getcwd(), 'models')

    model_path = os.path.join(models_dir, 'heart_failure_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    feature_path = os.path.join(models_dir, 'feature_columns.pkl')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feature_path)

    input_df = pd.DataFrame([input_data])
    # one-hot encode any categorical values in the single-row input
    input_df = pd.get_dummies(input_df)
    # align to training feature columns, fill missing with 0
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    return prediction[0]


if __name__ == '__main__':
    # simple CLI: take first row from dataset and predict
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

    print('Loading sample from', data_file)
    df = pd.read_csv(data_file)
    # detect target column name to drop from features
    possible_targets = ['DEATH_EVENT', 'HeartDisease', 'target', 'Outcome', 'DEATH']
    target_col = None
    for t in possible_targets:
        if t in df.columns:
            target_col = t
            break
    if target_col is not None:
        sample = df.drop(target_col, axis=1).iloc[0].to_dict()
    else:
        # no target column found, just take first row
        sample = df.iloc[0].to_dict()

    pred = predict(sample)
    print('Sample prediction:', pred)
