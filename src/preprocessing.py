# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(csv_path):
    # Load CSV
    data = pd.read_csv(csv_path)
    
    # Separate features and label
    X = data.drop(["formula", "scan_id", "hydrolysis_grade"], axis=1)
    y = data["hydrolysis_grade"]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler
