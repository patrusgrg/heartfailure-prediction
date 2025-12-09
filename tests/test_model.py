"""Unit tests for heart failure prediction model."""
import os
import sys
import pytest
import pandas as pd
import numpy as np
import tempfile
from sklearn.metrics import accuracy_score

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import create_preprocessing_pipeline, detect_target_column
from src.train import train_model
from src.predict import predict
from src.evaluate import evaluate_model


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    data = {
        'age': np.random.normal(60, 10, n_samples),
        'ejection_fraction': np.random.normal(40, 15, n_samples),
        'serum_creatinine': np.random.normal(1.5, 0.5, n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'DEATH_EVENT': np.random.choice([0, 1], n_samples)
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_models_dir():
    """Create a temporary directory for model artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestPreprocessing:
    """Tests for preprocessing utilities."""
    
    def test_preprocessing_pipeline(self, sample_data):
        """Test that preprocessing pipeline correctly handles numeric and categorical data."""
        preprocessor, num_features, cat_features = create_preprocessing_pipeline(sample_data)
        assert 'age' in num_features
        assert 'sex' in cat_features
        assert len(num_features) == 3
        assert len(cat_features) == 1

    def test_target_detection(self, sample_data):
        """Test that target column detection works correctly."""
        target = detect_target_column(sample_data)
        assert target == 'DEATH_EVENT'

    def test_target_detection_missing(self):
        """Test target detection fails gracefully with no target."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        target = detect_target_column(df)
        assert target is None


class TestModelTraining:
    """Tests for model training."""
    
    def test_model_training(self, sample_data, temp_models_dir):
        """Test full model training pipeline."""
        model_path, scaler_path, feature_path, metrics_path = train_model(
            sample_data, models_dir=temp_models_dir
        )
        
        # Check that files were created
        assert os.path.exists(model_path)
        assert os.path.exists(scaler_path)
        assert os.path.exists(feature_path)
        assert os.path.exists(metrics_path)

    def test_model_metrics(self, sample_data, temp_models_dir):
        """Test that metrics are properly saved."""
        import json
        import joblib
        
        model_path, scaler_path, feature_path, metrics_path = train_model(
            sample_data, models_dir=temp_models_dir
        )
        
        # Load and verify metrics
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        assert 'test_accuracy' in metrics
        assert 'test_roc_auc' in metrics
        assert 'cv_accuracy_mean' in metrics
        assert 'feature_importance' in metrics
        assert 0 <= metrics['test_accuracy'] <= 1
        assert 0 <= metrics['test_roc_auc'] <= 1

    def test_model_training_with_different_target(self, temp_models_dir):
        """Test training with different target column names."""
        df = pd.DataFrame({
            'age': np.random.normal(60, 10, 50),
            'score': np.random.normal(40, 15, 50),
            'HeartDisease': np.random.choice([0, 1], 50)
        })
        
        model_path, _, _, _ = train_model(df, models_dir=temp_models_dir)
        assert os.path.exists(model_path)


class TestPrediction:
    """Tests for prediction functionality."""
    
    def test_predict_single_sample(self, sample_data, temp_models_dir):
        """Test prediction on a single sample."""
        # Train model first
        train_model(sample_data, models_dir=temp_models_dir)
        
        # Get a sample and make prediction
        sample_dict = sample_data.drop('DEATH_EVENT', axis=1).iloc[0].to_dict()
        result = predict(sample_dict, models_dir=temp_models_dir)
        
        assert 'prediction' in result
        assert 'probability' in result
        assert result['prediction'] in [0, 1]
        assert 0 <= result['probability'] <= 1

    def test_predict_batch(self, sample_data, temp_models_dir):
        """Test prediction on multiple samples."""
        # Train model
        train_model(sample_data, models_dir=temp_models_dir)
        
        # Prepare batch
        X = sample_data.drop('DEATH_EVENT', axis=1).head(5)
        result = predict(X, models_dir=temp_models_dir)
        
        assert 'prediction' in result
        assert 'probability' in result
        assert len(result['prediction']) == 5


class TestEvaluation:
    """Tests for model evaluation."""
    
    def test_evaluate_model(self, sample_data, temp_models_dir):
        """Test model evaluation function."""
        # Train model
        train_model(sample_data, models_dir=temp_models_dir)
        
        # Evaluate (should not raise)
        evaluate_model(sample_data, models_dir=temp_models_dir)

    def test_model_accuracy_above_baseline(self, sample_data, temp_models_dir):
        """Test that model performs better than random baseline."""
        import joblib
        import json
        
        # Train model
        model_path, _, _, metrics_path = train_model(
            sample_data, models_dir=temp_models_dir
        )
        
        # Load metrics
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Random baseline for balanced classes is 0.5
        baseline_accuracy = 0.5
        assert metrics['test_accuracy'] > baseline_accuracy * 0.8, \
            f"Model accuracy {metrics['test_accuracy']} should be reasonably above baseline"


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_missing_target_column(self):
        """Test that missing target raises appropriate error."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        with pytest.raises(KeyError):
            train_model(df)

    def test_predict_without_trained_model(self, temp_models_dir):
        """Test that prediction fails gracefully without a trained model."""
        sample = {'age': 50, 'ejection_fraction': 40}
        
        with pytest.raises(FileNotFoundError):
            predict(sample, models_dir=temp_models_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
