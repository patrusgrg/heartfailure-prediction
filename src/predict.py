import os
import joblib
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def predict(input_data, models_dir=None):
    """Predict a single sample or batch.

    Args:
        input_data: dict, pandas Series, or DataFrame row(s)
        models_dir: optional folder where model/scaler are stored
        
    Returns:
        Prediction (int for single sample, array for batch)
    """
    if models_dir is None:
        models_dir = os.path.join(os.getcwd(), 'models')

    model_path = os.path.join(models_dir, 'heart_failure_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    feature_path = os.path.join(models_dir, 'feature_columns.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_columns = joblib.load(feature_path)
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
        raise

    # Convert input to DataFrame
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()

    # One-hot encode categorical values
    input_df = pd.get_dummies(input_df, drop_first=True)
    
    # Align to training feature columns
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    if input_df.isnull().any().any():
        logger.warning("NaN values detected after alignment. Filling with 0.")
        input_df = input_df.fillna(0)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)
    
    # Return predictions and probabilities
    return {
        'prediction': prediction[0] if len(prediction) == 1 else prediction,
        'probability': proba[0, 1] if len(proba) == 1 else proba[:, 1]
    }


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

    logger.info(f'Loading sample from {data_file}')
    df = pd.read_csv(data_file)
    
    # Detect target column
    possible_targets = ['DEATH_EVENT', 'HeartDisease', 'target', 'Outcome', 'DEATH']
    target_col = None
    for t in possible_targets:
        if t in df.columns:
            target_col = t
            break
    
    # Test multiple samples including both positive and negative cases
    print("\n" + "="*70)
    print("TESTING MULTIPLE PREDICTIONS")
    print("="*70)
    
    # Test first 5 samples
    num_samples = min(5, len(df))
    
    # Find positive case if available
    positive_indices = []
    if target_col is not None:
        positive_indices = df[df[target_col] == 1].index.tolist()[:3]
    
    # Indices to test
    test_indices = [0, 1, 2] + positive_indices
    test_indices = sorted(list(set(test_indices)))[:5]
    
    for idx in test_indices:
        if target_col is not None:
            sample = df.drop(target_col, axis=1).iloc[idx].to_dict()
            true_label = df[target_col].iloc[idx]
        else:
            sample = df.iloc[idx].to_dict()
            true_label = None
        
        logger.info(f'Making prediction on sample {idx}...')
        result = predict(sample)
        
        prediction_text = 'Heart Failure' if result['prediction'] == 1 else 'No Heart Failure'
        print(f"\nSample {idx}:")
        print(f"  Prediction: {result['prediction']} ({prediction_text})")
        print(f"  Confidence: {result['probability']:.4f}")
        if true_label is not None:
            actual_text = 'Heart Failure' if true_label == 1 else 'No Heart Failure'
            match = "✓ Correct" if result['prediction'] == true_label else "✗ Wrong"
            print(f"  Actual: {true_label} ({actual_text}) {match}")
    
    print("\n" + "="*70 + "\n")
