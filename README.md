Heart Failure Prediction
========================

Quick start
-----------

1. Activate the project's virtual environment (Windows PowerShell):

	"C:/Users/patrusgurung/Desktop/heartfailure prediction/heartfailure-prediction/venv310/Scripts/Activate.ps1"

2. Install dependencies if needed:

	pip install -r requirements.txt

3. Train a model (this will save artifacts into `models/`):

	"C:/Users/patrusgurung/Desktop/heartfailure prediction/heartfailure-prediction/venv310/Scripts/python.exe" src/train.py

4. Run a sample prediction:

	"C:/Users/patrusgurung/Desktop/heartfailure prediction/heartfailure-prediction/venv310/Scripts/python.exe" src/predict.py

5. Evaluate on the full dataset:

	"C:/Users/patrusgurung/Desktop/heartfailure prediction/heartfailure-prediction/venv310/Scripts/python.exe" src/evaluate.py

Notes
-----
- The dataset is expected at `data/heart_failure_clinical_records_dataset.csv`.
- Scripts already create a `models/` dir when needed.
- For programmatic usage import `train.train_model`, `predict.predict` and `evaluate.evaluate_model` from `src` (adjust PYTHONPATH to include `src/`).
