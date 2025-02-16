import joblib
import os
import json
import pandas as pd
import numpy as np
import optuna
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from imblearn.over_sampling import SMOTE
from data_processing import preprocess  # Ensure this function is implemented

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Setup logging
log_file = "logs/training_logistic.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

# Load & Preprocess Data
train_data = pd.read_excel("data/raw/train_file.xlsx")
processed_data = preprocess(train_data)

# Define Features (X) and Target (y)
target_col = "y"
X = processed_data.drop(columns=[target_col])
y = processed_data[target_col]

# One-Hot Encoding
X = pd.get_dummies(X, drop_first=True)

# Apply SMOTE to Balance Data
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Standardize Features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Save Scaler for later use
joblib.dump(scaler, "models/scaler.pkl")

# Save Processed Data for Evaluation
pd.DataFrame(X_resampled, columns=X.columns).to_csv("data/processed/X_train.csv", index=False)
pd.DataFrame(y_resampled, columns=["y"]).to_csv("data/processed/y_train.csv", index=False)

logging.info("Processed data saved successfully.")

# Compute Class Weights
neg, pos = np.bincount(y_resampled)
class_weight = {0: 1, 1: neg / pos}

# Hyperparameter Optimization Using Optuna
def objective(trial):
    params = {
        "C": trial.suggest_loguniform("C", 0.01, 10),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "solver": "liblinear" if trial.suggest_categorical("penalty", ["l1", "l2"]) == "l1" else "lbfgs",
        "class_weight": class_weight,
    }
    model = LogisticRegression(**params, max_iter=1000)
    return cross_val_score(model, X_resampled, y_resampled, cv=3, scoring="average_precision").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
best_params = study.best_params
logging.info(f"Best Parameters: {best_params}")

# Train Best Model Using Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_auc = 0

for fold, (train_idx, val_idx) in enumerate(skf.split(X_resampled, y_resampled)):
    logging.info(f"Training Fold {fold + 1}...")

    X_train_fold, X_val_fold = X_resampled[train_idx], X_resampled[val_idx]
    y_train_fold, y_val_fold = y_resampled[train_idx], y_resampled[val_idx]

    model = LogisticRegression(**best_params, max_iter=1000)
    model.fit(X_train_fold, y_train_fold)

    y_val_pred_proba = model.predict_proba(X_val_fold)[:, 1]
    auc_score = roc_auc_score(y_val_fold, y_val_pred_proba)

    logging.info(f"Fold {fold + 1} AUC: {auc_score:.4f}")

    if auc_score > best_auc:
        best_auc = auc_score
        best_model = model

logging.info(f"Best Model AUC: {best_auc:.4f}")

# Evaluate on Training Set
y_pred_proba = best_model.predict_proba(X_resampled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_resampled, y_pred_proba)
pr_auc = auc(recall, precision)
best_threshold = thresholds[np.argmax((2 * precision * recall) / (precision + recall + 1e-9))]

logging.info(f"Final PR-AUC: {pr_auc:.4f}, Optimal Threshold: {best_threshold:.4f}")

# Save Model & Performance Metrics
model_data = {
    "model": best_model,
    "best_threshold": float(best_threshold),
    "feature_columns": list(X.columns),
    "roc_auc": float(best_auc),
    "pr_auc": float(pr_auc),
}

joblib.dump(model_data, "models/best_logistic_regression_model.pkl")

with open("results/model_performance.json", "w") as f:
    json.dump({
        "roc_auc": float(best_auc),
        "pr_auc": float(pr_auc),
        "best_threshold": float(best_threshold)
    }, f)

logging.info("Model training completed successfully!")
