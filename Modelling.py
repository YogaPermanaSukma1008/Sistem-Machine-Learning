import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# === 1. Set MLflow Tracking URI ===
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Aktifkan autolog
mlflow.sklearn.autolog(log_models=True)

# === 2. Load Data ===
X_train = pd.read_csv("loandata_preprocessing/X_train_processed.csv")
X_test = pd.read_csv("loandata_preprocessing/X_test_processed.csv")
y_train = pd.read_csv("loandata_preprocessing/y_train.csv")
y_test = pd.read_csv("loandata_preprocessing/y_test.csv")

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# === 3. MLflow Run dengan Autologging ===
with mlflow.start_run(run_name="RandomForest_Local"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    model.predict(X_test)  # Memastikan autolog juga mencatat predict() jika relevan

print("✅ Model berhasil dilatih dan semua informasi dicatat otomatis oleh MLflow.")
