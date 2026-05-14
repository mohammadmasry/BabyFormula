# src/train_model.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import joblib
from preprocessing import load_and_preprocess

# Load and preprocess data
X, y, scaler = load_and_preprocess("../data/espectros_unificados.xlsx - Espectros.csv")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Optional: reduce dimensions
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model, scaler, PCA, label encoder
joblib.dump(model, "../models/hydrolysis_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(pca, "../models/pca.pkl")
joblib.dump(le, "../models/label_encoder.pkl")

print("Model training complete!")
