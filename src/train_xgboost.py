import joblib
import json
import pandas as pd
import xgboost as xgb
import os
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score, roc_curve, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import logging
from data_processing import preprocess  # Ensure the function is correctly implemented


# Ensure 'logs/' directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  

# Setup logging
logging.basicConfig(filename="logs/training_xgboost.log", level=logging.INFO, format="%(asctime)s - %(message)s")
os
# **Load & Preprocess Data**
train_data = pd.read_excel("data/raw/train_file.xlsx")  
processed_data = preprocess(train_data)

# **Define X (features) and y (target)**
target_col = 'y'
X = processed_data.drop(columns=[target_col])
y = processed_data[target_col]

# **One-Hot Encoding**
X = pd.get_dummies(X, drop_first=True)

# **Train-Test Split**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# **Apply SMOTE to Balance Data**
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

processed_data_dir = "data/processed"
os.makedirs(processed_data_dir, exist_ok=True)

X_train.to_csv(f"{processed_data_dir}/X_train.csv", index=False)
X_test.to_csv(f"{processed_data_dir}/X_test.csv", index=False)
y_train.to_csv(f"{processed_data_dir}/y_train.csv", index=False)
y_test.to_csv(f"{processed_data_dir}/y_test.csv", index=False)

logging.info("Processed data saved successfully for later use.")


# **Compute Class Weight**
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# **Hyperparameter Optimization Using Optuna**
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'gamma': trial.suggest_loguniform('gamma', 0.01, 0.5),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': scale_pos_weight
    }
    model = xgb.XGBClassifier(**params, eval_metric="aucpr", use_label_encoder=False)
    return cross_val_score(model, X_train, y_train, cv=3, scoring="average_precision").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
best_params = study.best_params
logging.info(f"Best Parameters: {best_params}")

# **Train Best Model Using Stratified K-Fold**
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_auc = 0

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    logging.info(f"Training Fold {fold + 1}...")
    
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = xgb.XGBClassifier(**best_params, eval_metric="aucpr", use_label_encoder=False)
    model.fit(X_train_fold, y_train_fold)

    y_val_pred_proba = model.predict_proba(X_val_fold)[:, 1]
    auc_score = roc_auc_score(y_val_fold, y_val_pred_proba)
    logging.info(f"Fold {fold + 1} AUC: {auc_score:.4f}")

    if auc_score > best_auc:
        best_auc = auc_score
        best_model = model

logging.info(f"Best Model AUC: {best_auc:.4f}")

# **Evaluate Model**
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
best_threshold = thresholds[np.argmax((2 * precision * recall) / (precision + recall + 1e-9))]

logging.info(f"Final PR-AUC: {pr_auc:.4f}, Optimal Threshold: {best_threshold:.4f}")

# Convert NumPy float32 values to Python float
model_data = {
    "model": best_model,
    "best_threshold": float(best_threshold),
    "feature_columns": list(X_train.columns),
    "roc_auc": float(best_auc),
    "pr_auc": float(pr_auc)
}

# Save the model
joblib.dump(model_data, "models/best_xgboost_model.pkl")

# Save performance metrics in a JSON file
with open("results/model_performance_xgboost.json", "w") as f:
    json.dump(
        {
            "roc_auc": float(best_auc),
            "pr_auc": float(pr_auc),
            "best_threshold": float(best_threshold)
        },
        f
    )

# Log success message
logging.info("Model training completed and saved successfully!")