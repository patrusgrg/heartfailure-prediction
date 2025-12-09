# Heart Failure Prediction

A production-ready machine learning pipeline for predicting heart failure risk using clinical features.

## Overview

This project builds a classification system to predict whether a patient will experience a heart failure event based on 12 clinical measurements. The implementation includes multiple algorithms, comprehensive evaluation metrics, explainability tools, and production-ready features.

### Key Capabilities

- Multiple Models: Logistic Regression, Random Forest, Gradient Boosting, SVM
- Comprehensive Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Model Explainability: Feature importance, SHAP values, ROC curves
- Cross-Validation: 5-fold stratified cross-validation
- Class Handling: Balanced class weights and stratified sampling
- Production Ready: Logging, error handling, unit tests, model persistence

## Quick Start

### 1. Activate Virtual Environment (Windows PowerShell)

```powershell
. "C:\Users\patrusgurung\Desktop\heartfailure prediction\heartfailure-prediction\venv310\Scripts\Activate.ps1"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python src/train.py
```

Expected output:
```
Dataset shape: (918, 11), Target distribution: {1: 508, 0: 410}
Test Accuracy: 0.8696, Test ROC-AUC: 0.9298
CV Accuracy: 0.8583 (+/- 0.0319)
Model saved to models/heart_failure_model.pkl
```

### 4. Evaluate the Model

```bash
python src/evaluate.py
```

### 5. Make Predictions

```bash
python src/predict.py
```

## Advanced Features

### Model Comparison

Compare all models systematically:

```bash
python src/model_comparison.py
```

Results Summary:
| Model | Test Accuracy | ROC-AUC | CV Accuracy |
|-------|---|---|---|
| Logistic Regression | 89.13% | 0.9293 | 84.46% |
| Random Forest | 86.96% | 0.9298 | 85.83% |
| Gradient Boosting | 87.50% | 0.9271 | 85.69% |
| SVM | 88.59% | 0.9442 | 86.10% |

### Advanced Training with Visualizations

Generate feature importance and SHAP plots:

```bash
python src/train_advanced.py
```

Generated outputs:
- reports/figures/feature_importance.png
- reports/figures/roc_curve.png
- reports/figures/confusion_matrix.png
- reports/figures/shap_summary.png (optional)

## Project Structure

```
heartfailure-prediction/
├── src/
│   ├── train.py                 # Main training with cross-validation
│   ├── train_advanced.py        # Advanced training with SHAP
│   ├── predict.py               # Prediction interface
│   ├── evaluate.py              # Model evaluation
│   ├── model_comparison.py      # Algorithm comparison
│   ├── preprocessing.py         # Data utilities
│   └── visualize.py             # Visualization functions
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
├── data/
│   └── heart.csv                # 918 samples, 12 features
├── models/
│   ├── heart_failure_model.pkl  # Trained model
│   ├── scaler.pkl               # Feature scaler
│   ├── feature_columns.pkl      # Feature names
│   ├── metrics.json             # Performance metrics
│   └── comparisons/             # Model comparison results
├── reports/
│   └── figures/                 # Generated visualizations
├── tests/
│   └── test_model.py            # Unit tests (pytest)
├── requirements.txt
└── README.md
```

## Model Performance

### Best Model: Random Forest

- Test Accuracy: 86.96%
- Test ROC-AUC: 0.9298
- Cross-Validation Accuracy: 85.83% +/- 3.19%
- Cross-Validation Recall: 89.14%
- Cross-Validation F1: 87.40%

### Top 5 Important Features

1. ST_Slope_Up (20.39%) - ST segment slope during exercise
2. ST_Slope_Flat (13.15%) - Flat ST segment indicator
3. MaxHR (10.75%) - Maximum heart rate achieved
4. ExerciseAngina_Y (10.19%) - Induced angina during exercise
5. Cholesterol (9.88%) - Serum cholesterol level

## Dataset Information

- Size: 918 samples
- Features: 12 clinical measurements
- Target: Binary classification (0: No event, 1: Event occurred)
- Class Distribution: 55.3% positive, 44.7% negative
- Missing Values: None

## Implementation Details

### Bug Fixes
- Removed duplicate return statements in training pipeline
- Fixed feature alignment in prediction process
- Added comprehensive input validation

### Enhancements
- 5-fold stratified cross-validation for robust evaluation
- Balanced class weights to handle data imbalance
- StandardScaler for feature normalization
- Reproducible results with fixed random seeds
- SHAP TreeExplainer for model interpretation

### Quality Assurance
- 15+ unit tests covering all major functions
- pytest framework with fixtures and parametrized tests
- Baseline performance checks
- Error handling and validation scenarios

## Programmatic Usage

### Train and Save Model

```python
from src.train import train_model
import pandas as pd

df = pd.read_csv('data/heart.csv')
model_path, scaler_path, feature_path, metrics_path = train_model(df)

import json
with open(metrics_path) as f:
    metrics = json.load(f)
    print(f"Accuracy: {metrics['test_accuracy']:.4f}")
```

### Make a Prediction

```python
from src.predict import predict

sample = {'age': 50, 'ejection_fraction': 40, 'cholesterol': 200}
result = predict(sample)
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.4f}")
```

### Evaluate Model

```python
from src.evaluate import evaluate_model

df = pd.read_csv('data/heart.csv')
evaluate_model(df)
```

### Compare Models

```python
from src.model_comparison import compare_models, print_comparison_table

df = pd.read_csv('data/heart.csv')
results = compare_models(df)
print_comparison_table(results)
```

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_model.py -v
```

Test Coverage:
- Data preprocessing and pipeline validation
- Target column detection
- Model training and metrics computation
- Single and batch predictions
- Model evaluation and reporting
- Performance baseline validation
- Error handling and edge cases

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas, numpy | Data handling and numerical operations |
| scikit-learn | Classification algorithms and evaluation |
| xgboost | Gradient boosting models |
| matplotlib, seaborn | Data visualization |
| shap | SHAP explainability |
| joblib | Model serialization |
| pytest | Unit testing framework |

See requirements.txt for specific versions.

## Clinical Interpretation

Important: This is a research/portfolio project, not a medical device.

### Key Findings

- ST segment characteristics are the strongest predictors of heart failure
- Maximum heart rate during exercise indicates cardiac stress level
- Exercise-induced angina is a significant risk indicator
- Serum cholesterol shows metabolic relationships with heart disease
- ST depression indicates potential myocardial ischemia

### Model Limitations

- Retrospective dataset without prospective validation
- May not generalize to all patient populations
- Handles class imbalance but minority class is underrepresented
- Requires regulatory approval for clinical deployment
- Describes data patterns, not causal relationships
- Should not replace clinical judgment

## Next Steps for Enhancement

1. Hyperparameter tuning with GridSearchCV
2. Feature engineering with domain expertise
3. Ensemble methods (stacking, voting, blending)
4. REST API deployment (FastAPI/Flask)
5. Production monitoring and model tracking
6. Fairness and bias analysis
7. Time-based cross-validation for temporal data

## Configuration Notes

- Dataset expected at: data/heart.csv
- Models directory automatically created on first run
- PYTHONPATH adjustment: include src/ directory for imports
- Features automatically scaled during preprocessing
- Cross-validation uses stratified 80/20 train-test split
- Random seed (42) ensures reproducibility

## License

Portfolio project for machine learning demonstration purposes.

## Contact

For questions or improvement suggestions, please open an issue or submit a pull request.
