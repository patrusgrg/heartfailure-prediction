import pandas as pd

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    # Additional preprocessing steps
    return df
