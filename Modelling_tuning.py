import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import pydantic
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, log_loss, matthews_corrcoef
)
from sklearn.model_selection import GridSearchCV

# 1. Setup MLflow + DagsHub
os.environ['MLFLOW_TRACKING_USERNAME'] = "YogaPermanaSukma1008"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "40dea981be77ab45c5501b0810cd96844a71a99a"
mlflow.set_tracking_uri("https://dagshub.com/YogaPermanaSukma1008/membangun-model.mlflow")

# 2. Load data
X_train = pd.read_csv("loandata_preprocessing/X_train_processed.csv")
X_test = pd.read_csv("loandata_preprocessing/X_test_processed.csv")
y_train = pd.read_csv("loandata_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("loandata_preprocessing/y_test.csv").values.ravel()

# 3. Visualisasi artefak
def log_confusion_matrix(cm):
    with tempfile.TemporaryDirectory() as tmp_dir:
        fig_path = os.path.join(tmp_dir, "confusion_matrix.png")
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(fig_path, artifact_path="confusion_matrix")

def log_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    with tempfile.TemporaryDirectory() as tmp_dir:
        fig_path = os.path.join(tmp_dir, "roc_curve.png")
        plt.figure()
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(fig_path, artifact_path="roc_curve")

# 4. MLflow run
with mlflow.start_run(run_name="RandomForest_Tuned_DagsHub") as run:
    print(f"Run ID: {run.info.run_id}")

    # 5. Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }

    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        verbose=2,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    # 6. Ambil model terbaik
    best_model = grid.best_estimator_

    # Logging best params
    mlflow.log_params(grid.best_params_)

    # 7. Prediksi dan evaluasi
    preds = best_model.predict(X_test)
    probas = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, probas)
    logloss = log_loss(y_test, probas)
    mcc = matthews_corrcoef(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    # 8. Logging metrik manual
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("log_loss", logloss)
    mlflow.log_metric("matthews_corrcoef", mcc)

    # 9. Logging artefak visual
    log_confusion_matrix(cm)
    log_roc_curve(y_test, probas)

    # 10. Logging model
    mlflow.sklearn.log_model(best_model, "model")

print("✅ Model tuning, logging metrik dan artefak ke DagsHub selesai.")
