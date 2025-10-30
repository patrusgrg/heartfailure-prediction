import sys, os
sys.path.append(os.path.join(os.getcwd(), 'src'))
print('PYTHONPATH includes', sys.path[-1])
try:
    from preprocess import preprocess_data
    print('imported preprocess ok')
except Exception as e:
    print('import preprocess failed:', e)

data_path = os.path.join(os.getcwd(), 'data', 'heart_failure_clinical_records_dataset.csv')
print('Data path:', data_path)

try:
    df = preprocess_data(data_path)
    print('Loaded df shape:', df.shape)
except Exception as e:
    print('loading data failed:', e)

print('models dir exists before:', os.path.exists('models'))
