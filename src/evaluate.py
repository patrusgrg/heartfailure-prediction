import joblib
import pandas as pd
from sklearn.metrics import classification_report

def evaluate_model(df):
    model = joblib.load('models/heart_failure_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']
    
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    print(classification_report(y, y_pred))
