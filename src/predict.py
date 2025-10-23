import joblib
import pandas as pd

def predict(input_data):
    model = joblib.load('models/heart_failure_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)
    return prediction[0]
