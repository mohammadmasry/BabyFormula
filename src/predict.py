# src/predict.py
import joblib
import pandas as pd

# Load trained objects
model = joblib.load("../models/hydrolysis_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
pca = joblib.load("../models/pca.pkl")
le = joblib.load("../models/label_encoder.pkl")

# Load new scan(s)
new_data = pd.read_csv("../data/new_scan.csv")
X_new = new_data.drop(["formula", "scan_id"], axis=1)

# Scale and reduce dimensions
X_scaled = scaler.transform(X_new)
X_reduced = pca.transform(X_scaled)

# Predict
y_pred = model.predict(X_reduced)
print("Predicted hydrolysis grades:", le.inverse_transform(y_pred))
