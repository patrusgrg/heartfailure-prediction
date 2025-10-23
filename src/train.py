from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(df):
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    joblib.dump(model, 'models/heart_failure_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
