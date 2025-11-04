import sys, os

# make src importable
sys.path.append(os.path.join(os.getcwd(), 'src'))
print('PYTHONPATH includes', sys.path[-1])

try:
    from preprocess import preprocess_data
    from train import train_model
    print('imported preprocess and train ok')
except Exception as e:
    print('import failed:', e)
    raise

# find data file (accept common alternative names)
data_dir = os.path.join(os.getcwd(), 'data')
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

print('Data path resolved to:', data_file)
if data_file is None:
    print('No CSV data file found in', data_dir)
else:
    try:
        df = preprocess_data(data_file)
        print('Loaded df shape:', df.shape)
    except Exception as e:
        print('loading data failed:', e)

    # Ensure models directory exists
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    print('models dir exists before training:', os.path.exists(models_dir))

    # run a short training to ensure save/load works
    try:
        mp, sp, fp = train_model(df, models_dir=models_dir)
        print('Saved model files exist:', os.path.exists(mp), os.path.exists(sp))
        print('Saved feature file exists:', os.path.exists(fp))
    except Exception as e:
        print('training failed:', e)
