# Heart Failure Prediction

**A complete machine learning pipeline for predicting heart failure risk using clinical features.**

## Overview

This project predicts whether a patient will experience a heart failure event based on clinical measurements using multiple classification models. The system includes:

- ✅ **Baseline & Advanced Models**: Logistic Regression, Random Forest, Gradient Boosting, SVM
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- ✅ **Explainability**: Feature importance plots, SHAP values, ROC curves
- ✅ **Cross-Validation**: 5-fold stratified cross-validation for robust evaluation
- ✅ **Class Imbalance Handling**: Balanced class weights & stratified sampling
- ✅ **Production-Ready**: Logging, error handling, reproducibility

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

**Output:**
```
Dataset shape: (918, 11), Target distribution: {1: 508, 0: 410}
Test Accuracy: 0.8696, Test ROC-AUC: 0.9298
CV Accuracy: 0.8583 (+/- 0.0319)
✓ Model saved to models/heart_failure_model.pkl
```

### 4. Evaluate on Full Dataset

```bash
python src/evaluate.py
```

### 5. Make Predictions

```bash
python src/predict.py
```

## Advanced Features

### Model Comparison

Compare multiple models side-by-side:

```bash
python src/model_comparison.py
```

Results:
| Model | Test Acc | Test ROC-AUC | CV Acc | CV F1 |
|-------|----------|--------------|--------|-------|
| Logistic Regression | 0.8913 | 0.9293 | 0.8446 | 0.8585 |
| Random Forest | 0.8696 | 0.9298 | 0.8583 | 0.8740 |
| Gradient Boosting | 0.8750 | 0.9271 | 0.8569 | 0.8721 |
| **SVM** | **0.8859** | **0.9442** | **0.8610** | **0.8760** |

### Advanced Training with SHAP

Generate feature importance plots and SHAP explanations:

```bash
python src/train_advanced.py
```

Generates:
- `reports/figures/feature_importance.png` - Top 15 feature importances
- `reports/figures/roc_curve.png` - ROC curve with AUC score
- `reports/figures/confusion_matrix.png` - Heatmap of confusion matrix
- `reports/figures/shap_summary.png` - SHAP summary plot (optional)

## Project Structure

```
heartfailure-prediction/
├── src/
│   ├── train.py              # Main training script with cross-validation
│   ├── train_advanced.py     # Advanced training with SHAP & visualizations
│   ├── predict.py            # Make predictions on new data
│   ├── evaluate.py           # Comprehensive model evaluation
│   ├── model_comparison.py   # Compare multiple algorithms
│   ├── preprocessing.py      # Data preprocessing utilities
│   └── visualize.py          # Visualization functions
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
├── data/
│   └── heart.csv            # Dataset (918 samples, 12 features)
├── models/
│   ├── heart_failure_model.pkl      # Trained model
│   ├── scaler.pkl                   # Feature scaler
│   ├── feature_columns.pkl          # Feature names
│   ├── metrics.json                 # Performance metrics
│   └── comparisons/
│       └── model_comparison_results.json
├── reports/
│   └── figures/
│       ├── feature_importance.png
│       ├── roc_curve.png
│       ├── confusion_matrix.png
│       └── shap_summary.png
├── tests/
│   └── test_model.py        # Comprehensive unit tests
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Model Performance

### Best Model: Random Forest

- **Test Accuracy**: 86.96%
- **Test ROC-AUC**: 0.9298
- **CV Accuracy**: 85.83% ± 3.19%
- **CV Precision**: 85.84%
- **CV Recall**: 89.14%
- **CV F1-Score**: 87.40%

### Top 5 Important Features

1. **ST_Slope_Up** (20.39%) - ST segment slope during exercise
2. **ST_Slope_Flat** (13.15%) - Flat ST segment indicator
3. **MaxHR** (10.75%) - Maximum heart rate achieved
4. **ExerciseAngina_Y** (10.19%) - Induced angina during exercise
5. **Cholesterol** (9.88%) - Serum cholesterol level

## Dataset

- **Size**: 918 samples
- **Features**: 12 clinical measurements
- **Target**: Heart failure event (binary: 0 or 1)
- **Class Distribution**: 55.3% positive, 44.7% negative
- **Missing Values**: None

## Key Improvements

### ✅ Fixed Critical Bugs
- Removed duplicate return statements in `train.py`
- Fixed feature alignment in prediction pipeline
- Added proper error handling and validation

### ✅ Added Comprehensive Metrics
- Confusion matrix with heatmap
- Classification report (precision, recall, F1)
- ROC-AUC score and ROC curves
- Feature importance ranking
- Cross-validation scores (accuracy, precision, recall, F1, ROC-AUC)

### ✅ Implemented Best Practices
- **Stratified Sampling**: Maintains class balance
- **Balanced Class Weights**: Handles class imbalance
- **Cross-Validation**: 5-fold stratified CV
- **Feature Scaling**: StandardScaler for normalization
- **Reproducibility**: Fixed random seeds

### ✅ Enhanced Explainability
- Feature importance plots
- SHAP value explanations (TreeExplainer)
- Confusion matrix visualization
- ROC curve with AUC annotation
- Comprehensive classification reports

### ✅ Production-Ready Features
- Logging throughout pipeline
- Graceful error handling
- Unit tests (pytest)
- Model persistence with joblib
- Metrics saved as JSON

### ✅ Model Comparison Framework
- Logistic Regression (interpretable baseline)
- Random Forest (ensemble, feature importance)
- Gradient Boosting (state-of-the-art)
- SVM (best ROC-AUC: 0.9442)

## Programmatic Usage

### Train a Model

```python
from src.train import train_model
import pandas as pd

df = pd.read_csv('data/heart.csv')
model_path, scaler_path, feature_path, metrics_path = train_model(df)

# Load metrics
import json
with open(metrics_path) as f:
    metrics = json.load(f)
    print(f"Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['test_roc_auc']:.4f}")
```

### Make Predictions

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
evaluate_model(df)  # Prints detailed evaluation report
```

### Compare Multiple Models

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

**Test Coverage:**
- Preprocessing pipeline validation
- Target column detection
- Model training with different target names
- Single & batch predictions
- Model evaluation
- Baseline performance checks
- Error handling scenarios

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas, numpy | Data handling |
| scikit-learn | ML algorithms |
| xgboost | Gradient boosting |
| matplotlib, seaborn | Visualization |
| shap | Model explainability |
| joblib | Model persistence |
| pytest | Unit testing |

See `requirements.txt` for versions.

## Clinical Interpretation

⚠️ **Important**: This is a research/portfolio project, not a medical device.

### Key Findings

1. **ST Segment Characteristics** (33.54%): ST segment properties are strongest predictors
2. **Heart Rate Response** (10.75%): Maximum heart rate during exercise is critical
3. **Symptom Presentation** (10.19%): Exercise-induced angina indicates risk
4. **Metabolic Factors** (9.88%): Cholesterol correlates with heart failure
5. **Cardiac Markers** (9.65%): ST depression indicates ischemia

### Model Limitations

- Retrospective dataset - not prospective validation
- May not generalize to different populations
- Real deployment requires regulatory approval
- Model explains data patterns, not causal relationships
- Class imbalance handled but minority class underrepresented

## Next Steps

1. **Hyperparameter Tuning**: GridSearchCV with extensive parameter grids
2. **Feature Engineering**: Domain-expert features, interaction terms
3. **Ensemble Methods**: Stacking, voting, or blending
4. **Deployment**: FastAPI/Flask REST API
5. **Monitoring**: Track model performance in production
6. **Fairness Analysis**: Ensure equitable performance across demographics
7. **Cross-Validation Strategy**: Time-based CV for temporal data

## Resume Highlights

- ✅ Achieved 87% ROC-AUC on heart failure prediction task
- ✅ Implemented 4 classification algorithms with systematic comparison
- ✅ Added SHAP explainability to model decisions
- ✅ 5-fold cross-validation with stratified sampling
- ✅ Handled class imbalance using balanced class weights
- ✅ Comprehensive unit tests with pytest
- ✅ Production-ready logging and error handling
- ✅ Automated metric tracking and visualization

## License

Portfolio project for machine learning demonstration.

## Contact

For questions or improvements, please open an issue or submit a pull request.
