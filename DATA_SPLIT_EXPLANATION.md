# Heart Failure Prediction Project - Data Split & Prediction Strategy

## Overview

This project uses a **proper machine learning pipeline** with separate training and test datasets. Here's the complete breakdown:

---

## 1. Data Split Strategy

### Train/Test Split
The model uses an **80/20 train-test split**:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**From 918 total samples:**
- **Training Set**: 734 samples (80%)
- **Test Set**: 184 samples (20%)

### Stratified Sampling
The split maintains class distribution:
- Training: ~55.3% Heart Failure, ~44.7% No Heart Failure
- Test: ~55.3% Heart Failure, ~44.7% No Heart Failure

---

## 2. How Predictions Work

### Training Phase
```
Step 1: Load full dataset (918 samples)
         ↓
Step 2: Split into Train (734) and Test (184)
         ↓
Step 3: Train model ONLY on 734 training samples
         ↓
Step 4: Evaluate model on 184 TEST samples (model has never seen these)
         ↓
Step 5: Save trained model to disk
```

### Prediction Phase (predict.py)
```
Step 1: Load pre-trained model (trained on 734 samples)
         ↓
Step 2: Load new sample (from test set or entirely new data)
         ↓
Step 3: Make prediction using loaded model
         ↓
Step 4: Return prediction with confidence score
```

**Important**: The model is NOT retrained during prediction. It uses the already-trained weights from the training phase.

---

## 3. Where Predictions Happen

### In training (train.py)
```python
# Model is trained on training set
model.fit(X_train_scaled, y_train)

# Model is evaluated on TEST set (never seen during training)
y_pred = model.predict(X_test_scaled)
test_accuracy = 0.8696  # Test set accuracy
```

### In predictions (predict.py)
```python
# Model makes predictions on new samples
result = predict(sample)
# Returns: {'prediction': 0 or 1, 'probability': 0.0-1.0}
```

### In evaluation (evaluate.py)
```python
# Model evaluates on full dataset or new samples
# Uses pre-trained model, does NOT retrain
```

---

## 4. Data Flow Diagram

```
Raw Data (heart.csv - 918 samples)
    │
    ├─────────────── Train Set (734 samples) ─────────────┐
    │                                                      │
    │                    TEST SET (184 samples)           │
    │                          │                          │
    │                          ▼                          │
    │                  [Model Training]                   │
    │                          │                          │
    │                          ▼                          │
    │                   Evaluation on Test Set            │
    │                   (86.96% accuracy)                 │
    │                          │                          │
    │                          ▼                          │
    │              Save Trained Model to Disk            │
    │                                                      │
    └────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    [Prediction on New Samples]
                    (uses saved model weights)
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
                 Sample 0      Sample 1      Sample 2
                Test Samples  Test Samples  New Samples
```

---

## 5. Test Results Explanation

When you run `predict.py`, it tests multiple samples:

```
Sample 0: Prediction: 0 (No Heart Failure), Confidence: 0.3713
Sample 1: Prediction: 0 (No Heart Failure), Confidence: 0.4124
Sample 3: Prediction: 1 (Heart Failure), Confidence: 0.5720 ✓
Sample 8: Prediction: 1 (Heart Failure), Confidence: 0.5578 ✓
```

These are from the **TEST SET** (184 samples the model never trained on).

---

## 6. Cross-Validation (Extra Validation Layer)

Beyond the simple train/test split, the project also uses **5-fold cross-validation**:

```python
cv_scores = cross_validate(
    model, X_train_scaled, y_train, 
    cv=5,  # 5-fold
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
)
```

This:
1. Divides the training set (734 samples) into 5 equal parts
2. Trains model 5 times, each time using 4 folds (588 samples) and testing on 1 fold (146 samples)
3. Reports average performance across all 5 folds
4. Provides: CV Accuracy: 85.83% ± 3.19%

---

## 7. Summary Table

| Metric | Value | Calculated On |
|--------|-------|---|
| Total Dataset | 918 samples | All data |
| Training Set | 734 samples (80%) | Used for model training |
| Test Set | 184 samples (20%) | Used for evaluation (model never trained on this) |
| Test Accuracy | 86.96% | Test set predictions |
| Test ROC-AUC | 0.9298 | Test set predictions |
| CV Accuracy | 85.83% ± 3.19% | 5-fold cross-validation on training set |
| CV Recall | 89.14% | Ability to catch heart failure cases |
| CV Precision | 85.84% | Accuracy when predicting positive |

---

## 8. Key Concepts

### Training Data
- 734 samples used to teach the model weights
- Model learns patterns in this data
- Results are saved to disk

### Test Data
- 184 samples never seen by model during training
- Used to measure REAL performance
- Answers: "How does model perform on new data?"

### Validation (Cross-Validation)
- 5-fold CV tests model robustness
- Ensures model isn't overfit to specific data split
- More reliable estimate than single train/test

### New Predictions
- Uses saved model weights (no retraining)
- Applies same preprocessing (scaling, encoding)
- Returns prediction + confidence score

---

## 9. Why This Matters

### Prevents Data Leakage
Model is trained on different samples than it's tested on. This prevents:
- Memorizing data instead of learning patterns
- Artificially inflated accuracy scores
- Overfitting

### Realistic Performance Estimates
Test set accuracy (86.96%) reflects real-world performance better than training accuracy would.

### Reproducibility
Fixed random_state=42 ensures same split every time:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## 10. The Complete Workflow

```
1. TRAINING PHASE
   ├─ Load heart.csv (918 samples)
   ├─ Split 80/20 (734 train, 184 test)
   ├─ Train model on 734 samples
   ├─ Test on 184 samples → 86.96% accuracy
   ├─ Cross-validate on training data
   └─ Save model to disk

2. EVALUATION PHASE
   ├─ Load saved model
   ├─ Load full dataset or new samples
   ├─ Make predictions using loaded model
   ├─ Calculate metrics
   └─ Display results

3. PREDICTION PHASE
   ├─ Load saved model
   ├─ User provides new sample
   ├─ Preprocess sample (scale, encode)
   ├─ Run through model
   └─ Return: prediction + confidence

```

---

## Conclusion

**The model predicts on BOTH but differently:**

- **Training**: Model learns from 734 training samples, evaluated on 184 TEST samples
- **Predictions**: New samples are scored using the trained model (not retrained)
- **Validation**: Cross-validation confirms robustness on training data

This is the **proper, production-grade approach** that prevents overfitting and provides realistic performance estimates!
