import os
import pandas as pd

def preprocess_data(file_path):
    """Load CSV and perform minimal cleaning. Raises FileNotFoundError if path is invalid."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)
    # basic cleaning - drop missing rows
    df.dropna(inplace=True)
    # Additional preprocessing steps can be added here
    return df


if __name__ == '__main__':
    # quick CLI to check data - compute repo root relative to this file
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(repo_root, 'data')
    candidates = [
        os.path.join(data_dir, 'heart_failure_clinical_records_dataset.csv'),
        os.path.join(data_dir, 'heart.csv'),
    ]
    test_path = None
    for c in candidates:
        if os.path.exists(c):
            test_path = c
            break
    if test_path is None:
        import glob
        files = glob.glob(os.path.join(data_dir, '*.csv'))
        if files:
            test_path = files[0]
    if test_path is None:
        raise FileNotFoundError(f'No CSV data file found in {data_dir}')

    print('Checking data at', test_path)
    df = preprocess_data(test_path)
    print('Loaded rows,cols:', df.shape)
