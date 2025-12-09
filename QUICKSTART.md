# Quick Reference Guide

## All Commands at a Glance

### 1. Initial Setup
```powershell
# Activate virtual environment
. "C:\Users\patrusgurung\Desktop\heartfailure prediction\heartfailure-prediction\venv310\Scripts\Activate.ps1"

# Install/update dependencies
pip install -r requirements.txt
```

### 2. Core Pipeline

#### Train Model (with cross-validation & metrics)
```bash
python src/train.py
```
**Output**: Model, scaler, features, and metrics.json

#### Evaluate Model (on full dataset)
```bash
python src/evaluate.py
```
**Output**: Accuracy, ROC-AUC, classification report, confusion matrix, top features

#### Make Predictions
```bash
python src/predict.py
```
**Output**: Prediction and probability for a sample

### 3. Advanced Analysis

#### Compare Multiple Models
```bash
python src/model_comparison.py
```
**Output**: Side-by-side comparison of 4 algorithms with saved results

#### Advanced Training (with visualizations)
```bash
python src/train_advanced.py
```
**Output**: Model + 4 visualization files (feature importance, ROC, confusion matrix, SHAP)

### 4. Testing
```bash
pytest tests/test_model.py -v
```
**Output**: 15+ test results

---

## Key Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 86.96% |
| **Test ROC-AUC** | 0.9298 |
| **CV Accuracy** | 85.83% ± 3.19% |
| **CV Precision** | 85.84% |
| **CV Recall** | 89.14% |
| **CV F1-Score** | 87.40% |

---

## Generated Files

### Models & Artifacts
- `models/heart_failure_model.pkl` - Trained model
- `models/scaler.pkl` - Feature scaler
- `models/feature_columns.pkl` - Feature names
- `models/metrics.json` - Performance metrics

### Visualizations
- `reports/figures/feature_importance.png`
- `reports/figures/roc_curve.png`
- `reports/figures/confusion_matrix.png`
- `reports/figures/shap_summary.png`

### Comparisons
- `models/comparisons/model_comparison_results.json`

---

## Top 5 Important Features

1. ST_Slope_Up (20.39%)
2. ST_Slope_Flat (13.15%)
3. MaxHR (10.75%)
4. ExerciseAngina_Y (10.19%)
5. Cholesterol (9.88%)

---

## Python API

### Import and Train
```python
from src.train import train_model
import pandas as pd

df = pd.read_csv('data/heart.csv')
model_path, scaler_path, feature_path, metrics_path = train_model(df)
```

### Make Predictions
```python
from src.predict import predict

sample = {'age': 50, 'ejection_fraction': 40}
result = predict(sample)
print(f"Prediction: {result['prediction']}, Prob: {result['probability']:.2%}")
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

### Advanced Training
```python
from src.train_advanced import train_with_explanations

df = pd.read_csv('data/heart.csv')
model_path, metrics_path, plots = train_with_explanations(df)
print(f"Generated plots: {plots.keys()}")
```

---

## Model Comparison Results

| Model | Test Acc | ROC-AUC | CV Acc | CV F1 |
|-------|----------|---------|--------|-------|
| Logistic Regression | 89.13% | 0.9293 | 84.46% | 85.85% |
| Random Forest | 86.96% | 0.9298 | 85.83% | 87.40% |
| Gradient Boosting | 87.50% | 0.9271 | 85.69% | 87.21% |
| **SVM** | **88.59%** | **0.9442** | **86.10%** | **87.60%** |

---

## Project Improvements Summary

✅ Fixed all critical bugs (duplicate returns, feature alignment)
✅ Added comprehensive evaluation metrics (ROC-AUC, confusion matrix, etc.)
✅ Implemented 5-fold cross-validation
✅ Created model comparison framework (4 algorithms)
✅ Added SHAP explainability
✅ Generated visualizations (feature importance, ROC curves)
✅ Added production-ready logging
✅ Comprehensive error handling
✅ 15+ unit tests
✅ Professional documentation

---

## Clinical Notes

⚠️ This is a research/portfolio project, NOT a medical device.

- Uses retrospective data (not prospective validation)
- Requires clinical validation before real-world use
- Model explains data patterns, not causality
- Use only with domain expert supervision

---

## Next Steps

1. Hyperparameter tuning with GridSearchCV
2. Feature engineering with domain experts
3. Ensemble methods (stacking, voting)
4. REST API deployment (FastAPI)
5. Production monitoring pipeline
6. Fairness analysis across demographics
7. Time-series cross-validation (if temporal data)

---

## Troubleshooting

### Virtual Environment Not Found
```powershell
python -m venv venv310
.\venv310\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Model Not Found
- Run `python src/train.py` first to train the model
- Check that `models/heart_failure_model.pkl` exists

### SHAP Visualization Missing
- SHAP is optional and may not install on all systems
- Run `pip install shap` to enable
- Advanced training will work without it (just skip SHAP plots)

### Test Failures
- Ensure pytest is installed: `pip install pytest`
- Run from project root: `cd heartfailure-prediction`
- Run: `pytest tests/test_model.py -v`

---

## File Locations

| File | Purpose |
|------|---------|
| `data/heart.csv` | Input dataset |
| `src/train.py` | Main training script |
| `src/evaluate.py` | Model evaluation |
| `src/predict.py` | Make predictions |
| `src/model_comparison.py` | Compare models |
| `src/train_advanced.py` | Advanced training |
| `tests/test_model.py` | Unit tests |
| `models/` | Trained models |
| `reports/figures/` | Visualizations |
| `README_NEW.md` | Full documentation |
| `IMPROVEMENTS.md` | Change log |

---

## Performance Targets

✅ **Test Accuracy**: >85% ✓ (achieved 86.96%)
✅ **ROC-AUC**: >0.90 ✓ (achieved 0.9298)
✅ **Recall**: >85% ✓ (achieved 89.14%)
✅ **Precision**: >80% ✓ (achieved 85.84%)
✅ **F1-Score**: >85% ✓ (achieved 87.40%)

---

## Timeline

1. **Data Loading** (~0.5s)
2. **Training** (~1-2s per model)
3. **Evaluation** (~0.1s)
4. **Visualization** (~0.5-1s)
5. **Full Pipeline** (~5 minutes for all models)

---

## Resume Bullets

✓ Developed ML pipeline predicting heart failure with 87% ROC-AUC  
✓ Implemented 4 classification algorithms with systematic comparison  
✓ Added SHAP explainability and feature importance visualization  
✓ Achieved 85.83% 5-fold cross-validation accuracy with stratified sampling  
✓ Handled class imbalance using balanced class weights and stratified sampling  
✓ Created comprehensive unit tests covering 15+ test cases  
✓ Production-ready code with logging, error handling, and reproducibility  
✓ Generated ROC curves, confusion matrices, and feature importance plots  

---

## Contact & Support

For questions or improvements:
1. Check `README_NEW.md` for detailed documentation
2. Review `IMPROVEMENTS.md` for what was changed
3. Check test cases in `tests/test_model.py` for usage examples
4. Open an issue or submit a pull request

---

**Last Updated**: December 9, 2025
**Status**: Production Ready ✅
**Version**: 2.0 (Fully Refactored)
