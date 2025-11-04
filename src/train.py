import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(df, models_dir=None):
    """Train a RandomForest on the provided dataframe and save model+scaler.

    Args:
        df: pandas DataFrame containing features and `DEATH_EVENT` target.
        models_dir: optional path to directory where artifacts will be saved.
    """
    if models_dir is None:
        models_dir = os.path.join(os.getcwd(), 'models')

    os.makedirs(models_dir, exist_ok=True)

    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    model_path = os.path.join(models_dir, 'heart_failure_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return model_path, scaler_path


if __name__ == '__main__':
    # simple CLI to run training: locate repository root relative to this file and find data/
    from preprocess import preprocess_data
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(repo_root, 'data')
    # prefer original filename but fall back to common alternatives
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
        raise FileNotFoundError(f'No CSV data file found in {data_dir} (expected one of: {candidates})')

    print('Loading data from', data_file)
    df = preprocess_data(data_file)
    print('Training model on', df.shape)
    mp, sp = train_model(df)
    print('Saved model to', mp)
