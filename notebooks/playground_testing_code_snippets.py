
# load libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.utils import shuffle
import joblib
import os
import scipy.stats as stats
from itertools import combinations
from src.data_processing import preprocess

# Load data
train_data = pd.read_excel("../data/raw/train_file.xlsx")

# First look at the dataset
print(train_data.head())   # Shows the first 5 rows
print(train_data.info())   # Shows data types & missing values
print(train_data.describe(include='all'))  


# Identify categorical and numerical columns
categorical_features = train_data.select_dtypes(include=['object', 'category']).columns
numerical_features = train_data.select_dtypes(include=['number']).columns


# Plot histograms for numerical features
train_data[numerical_features].hist(figsize=(12, 10), bins=30)
plt.tight_layout()
plt.show()

# Boxplot for 'duration' (Call Duration)
duration_minutes= round(train_data['duration']/60) # convert to minutes for ease of interpretation
sns.boxplot(x=duration_minutes)
plt.xlabel("Call Duration (Minutes)")
plt.title("Boxplot of Call Duration")
plt.show()



# Boxplot for '# campaign' 
sns.boxplot(x=train_data['campaign'])
plt.title("Boxplot of # campaign")
plt.show()



# Plot bar charts for categorical features
for col in categorical_features:
    plt.figure(figsize=(8, 4))
    train_data[col].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()







## ============ Statistical Feature Assocciations =================================

#Detect categorical columns (including ordinal categories with <15 unique values)
categorical_columns = [col for col in train_data.columns if train_data[col].dtype == 'object' or train_data[col].nunique() < 15]

# Dictionary to store p-values
p_values = {}

# Generate unique feature pairs (no duplicates like (A, B) and (B, A))
unique_pairs = list(combinations(categorical_columns, 2))

# Loop through unique categorical feature pairs and apply Chi-Square test
for col1, col2 in unique_pairs:
    contingency_table = pd.crosstab(train_data[col1], train_data[col2])  # Create contingency table
    _, p_value, _, _ = stats.chi2_contingency(contingency_table)  # Apply Chi-Square test
    p_values[(col1, col2)] = p_value  # Store p-value

# Convert results to DataFrame and rank by p-value
p_value_df = pd.DataFrame.from_dict(p_values, orient='index', columns=['p-value']).sort_values(by='p-value')

# Display results
print(p_value_df)


# Detect numerical columns (excluding target variable 'y')
numerical_columns = [col for col in train_data.columns if train_data[col].dtype in ['int64', 'float64'] and col != 'y']

# Dictionary to store correlation results
correlation_results = {}

# Generate unique feature pairs (no duplicates like (A, B) and (B, A))
unique_pairs = list(combinations(numerical_columns, 2))

# Loop through unique numerical feature pairs and apply Pearson & Spearman correlation tests
for col1, col2 in unique_pairs:
    # Pearson correlation (linear relationships)
    pearson_corr, pearson_p = stats.pearsonr(train_data[col1], train_data[col2])
    
    # Spearman correlation (monotonic relationships, less sensitive to outliers)
    spearman_corr, spearman_p = stats.spearmanr(train_data[col1], train_data[col2])

    # Store correlation values and p-values
    correlation_results[(col1, col2)] = {
        "Pearson Corr": round(pearson_corr, 5),
        "Pearson p-value": round(pearson_p, 5),
        "Spearman Corr": round(spearman_corr, 5),
        "Spearman p-value": round(spearman_p, 5),
    }

# Convert results to DataFrame and sort by Spearman p-value
correlation_df = pd.DataFrame.from_dict(correlation_results, orient='index').sort_values(by='Spearman p-value')

# Display results
print(correlation_df)


### Feature processing 
processed_data = preprocess_data(train_data)


train_data['y'] = train_data['y'].map({'yes': 1, 'no': 0})

train_data = shuffle(train_data, random_state=42)
# Define age bins and labels
bins = [18, 25, 35, 45, 55, 65, 100]  # Age ranges
labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']

# Apply binning
train_data['age_group'] = pd.cut(train_data['age'], bins=bins, labels=labels)


# Define a mapping dictionary
season_mapping = {
    'dec': 'Winter', 'jan': 'Winter', 'feb': 'Winter',
    'mar': 'Spring', 'apr': 'Spring', 'may': 'Spring',
    'jun': 'Summer', 'jul': 'Summer', 'aug': 'Summer',
    'sep': 'Fall', 'oct': 'Fall', 'nov': 'Fall'
}

# Apply mapping to create a new 'season' column
train_data['season'] = train_data['month'].map(season_mapping)


# Campaign
# Define bins and labels
bins = [0, 1, 2, 3, 7, float('inf')]  # Bins: (0-1], (1-2], (2-3], (3-7], (7+)
labels = ['1', '2', '3', '4-7', '>7']  # Ordinal scale

# Apply binning
train_data['campaign_binned'] = pd.cut(train_data['campaign'], bins=bins, labels=labels, right=True)



# previous
train_data['previous_bin'] = pd.cut(train_data['previous'], bins=[0, 0.5, 2, 10, 100], labels=['None', 'Few', 'Moderate', 'Many'])
train_data = pd.get_dummies(train_data, columns=['previous_bin'], drop_first=True)


# Define bins based on boxplot observation
bins = [0, 5, 15, 30, train_data['duration'].max()]  # Set max duration dynamically
labels = ['Short Call', 'Medium Call', 'Long Call', 'Extreme Long Call']

# Apply binning
train_data['duration_category'] = pd.cut(train_data['duration'], bins=bins, labels=labels, right=False)




#


### -----------Model Training-------------------------- 
   """
Machine Learning Pipeline for Bank Marketing Campaign Prediction
================================================================

This module provides a structured pipeline for training, optimizing, and evaluating machine learning models
to predict customer subscription to a bank's term deposit based on marketing campaign data.

### Implemented Models:
1. **XGBoost**: A powerful gradient boosting model optimized for performance and feature importance analysis.
2. **Logistic Regression**: A linear model serving as a strong baseline.
3. **Optuna Hyperparameter Optimization**: Automated tuning to improve model performance.

### Features:
1. **Data Preprocessing & Feature Engineering**
   - Reads training and test datasets.
   - Identifies categorical and numerical features.
   - Handles missing values (if applicable).
   - Applies **One-Hot Encoding** for categorical variables.
   - **Standardizes numerical features** for Logistic Regression.
   - Uses **SMOTE (Synthetic Minority Oversampling Technique)** to handle class imbalance.

2. **Model Training & Evaluation**
   - Implements **XGBoost** and **Logistic Regression** models.
   - Uses **Optuna** for **hyperparameter tuning**.
   - **Cross-validation** ensures robustness of performance.
   - Evaluates models using:
     - **AUC-ROC (Receiver Operating Characteristic Area Under Curve)**
     - **PR-AUC (Precision-Recall Area Under Curve) for imbalanced data**
     - **F1 Score for classification performance**
     - **Confusion Matrix & Classification Report for detailed error analysis**

3. **Feature Importance & Explainability**
   - Uses **SHAP (SHapley Additive Explanations)** for interpretability.
   - Plots feature importance for understanding key predictive factors.

4. **Visualization & Performance Analysis**
   - Generates **Precision-Recall and ROC Curves** for model evaluation.
   - Displays **Normalized Confusion Matrix** to analyze prediction errors.
   - Identifies **optimal decision thresholds** based on **F1 score maximization**.
"""

import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score, roc_curve, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

# **Preprocess Data with Clustering**
processed_data = preprocess(train_data)

# **Define X (features) and y (target)**
target_col = 'y'
X = processed_data.drop(columns=[target_col])
y = processed_data[target_col]

# **Convert categorical columns for XGBoost**
X = pd.get_dummies(X, drop_first=True)  # One-Hot Encoding for better handling

# **Split Data (Stratify to Preserve Imbalance)**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# **Apply SMOTE to Balance Data**
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

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
        'scale_pos_weight': scale_pos_weight  # Adjust for imbalance
    }
    model = xgb.XGBClassifier(**params, eval_metric="aucpr", use_label_encoder=False)
    return cross_val_score(model, X_train, y_train, cv=3, scoring="average_precision").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
best_params = study.best_params
print("Best Parameters:", best_params)

# **Stratified K-Fold Cross-Validation for Final Model Training**
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_auc = 0

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"Training Fold {fold + 1}...")

    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = xgb.XGBClassifier(**best_params, eval_metric="aucpr", use_label_encoder=False)
    model.fit(X_train_fold, y_train_fold)

    # Evaluate model on validation set
    y_val_pred_proba = model.predict_proba(X_val_fold)[:, 1]
    auc_score = roc_auc_score(y_val_fold, y_val_pred_proba)

    print(f"Fold {fold + 1} AUC: {auc_score:.4f}")

    # Store the best model
    if auc_score > best_auc:
        best_auc = auc_score
        best_model = model

print(f"Best Model AUC: {best_auc:.4f}")

# **Predict Probabilities for PR-AUC Calculation**
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# **Calculate PR-AUC (Better for Imbalanced Data)**
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
print(f"Test Set PR-AUC: {pr_auc:.4f}")

# **Optimal Decision Threshold Selection**
f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal Decision Threshold: {best_threshold:.4f}")

# **Convert Probabilities to Binary Predictions Using Best Threshold**
y_pred = (y_pred_proba >= best_threshold).astype(int)

# **Calculate AUC on the Test Set**
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"Test Set AUC: {auc_score:.4f}")

# **Calculate F1 Score**
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")

# **Plot Precision-Recall Curve**
plt.figure(figsize=(8,6))
plt.plot(recall, precision, marker='.', label=f'PR AUC = {pr_auc:.2f}')
plt.axvline(x=recall[np.argmax(f1_scores)], color='r', linestyle='--', label="Best F1 Threshold")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# **Plot ROC Curve**
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC AUC = {auc_score:.2f}')
plt.plot([0,1], [0,1], linestyle="--", color='gray')  # Random chance line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# **Plot Normalized Confusion Matrix**
conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')  # Normalize by row
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt=".2%", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Normalized Confusion Matrix")
plt.show()

# **Print Classification Report**
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# **Feature Importance Using SHAP**
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test)

# Store model, best threshold, and feature columns in a dictionary
model_data = {
    "model": best_model,
    "best_threshold": best_threshold,
    "feature_columns": X_train.columns.tolist()  # Save feature names
}

# Save the model, threshold, and feature columns
joblib.dump(model_data, "models/best_xgboost_model.pkl")

print(f"Best model, threshold, and feature columns saved successfully! Threshold: {best_threshold:.4f}")



#### ========== LOGISTIC REGRESSION =============================================

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score, roc_curve, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

# **Preprocess Data**
processed_data = preprocess(train_data)  # Custom preprocessing function

# **Define Features (X) and Target (y)**
target_col = 'y'
X = processed_data.drop(columns=[target_col])
y = processed_data[target_col]

# **One-Hot Encode Categorical Features**
X = pd.get_dummies(X, drop_first=True)

# **Apply SMOTE to Balance Training Data**
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# **Standardize Features**
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# **Compute Class Weights**
neg, pos = np.bincount(y_resampled)
class_weight = {0: 1, 1: neg / pos}

# **Hyperparameter Optimization Using Optuna**
def objective(trial):
    params = {
        'C': trial.suggest_loguniform('C', 0.01, 10),  # Regularization strength
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),  # Regularization type
        'solver': 'liblinear' if 'l1' in trial.params.get('penalty', 'l2') else 'lbfgs',
        'class_weight': class_weight
    }
    model = LogisticRegression(**params, max_iter=1000)
    return cross_val_score(model, X_resampled, y_resampled, cv=3, scoring="average_precision").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
best_params = study.best_params
print("Best Parameters:", best_params)

# **Stratified K-Fold Cross-Validation Training**
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_auc = 0

for fold, (train_idx, val_idx) in enumerate(skf.split(X_resampled, y_resampled)):
    print(f"Training Fold {fold + 1}...")

    X_train_fold, X_val_fold = X_resampled[train_idx], X_resampled[val_idx]
    y_train_fold, y_val_fold = y_resampled[train_idx], y_resampled[val_idx]

    model = LogisticRegression(**best_params, max_iter=1000)
    model.fit(X_train_fold, y_train_fold)

    # Evaluate model on validation set
    y_val_pred_proba = model.predict_proba(X_val_fold)[:, 1]
    auc_score = roc_auc_score(y_val_fold, y_val_pred_proba)

    print(f"Fold {fold + 1} AUC: {auc_score:.4f}")

    # Store the best model
    if auc_score > best_auc:
        best_auc = auc_score
        best_model = model

print(f"Best Model AUC: {best_auc:.4f}")

# **Predict Probabilities for PR-AUC Calculation**
y_pred_proba = best_model.predict_proba(X_resampled)[:, 1]

# **Calculate PR-AUC**
precision, recall, thresholds = precision_recall_curve(y_resampled, y_pred_proba)
pr_auc = auc(recall, precision)
print(f"Final PR-AUC: {pr_auc:.4f}")

# **Optimal Decision Threshold Selection**
f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal Decision Threshold: {best_threshold:.4f}")

# **Convert Probabilities to Binary Predictions**
y_pred = (y_pred_proba >= best_threshold).astype(int)

# **Calculate Metrics**
auc_score = roc_auc_score(y_resampled, y_pred_proba)
f1 = f1_score(y_resampled, y_pred)
print(f"Final Test AUC: {auc_score:.4f}")
print(f"Final Test F1 Score: {f1:.4f}")

# **Plot Precision-Recall Curve**
plt.figure(figsize=(8,6))
plt.plot(recall, precision, marker='.', label=f'PR AUC = {pr_auc:.2f}')
plt.axvline(x=recall[np.argmax(f1_scores)], color='r', linestyle='--', label="Best F1 Threshold")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# **Plot ROC Curve**
fpr, tpr, _ = roc_curve(y_resampled, y_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC AUC = {auc_score:.2f}')
plt.plot([0,1], [0,1], linestyle="--", color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# **Plot Normalized Confusion Matrix**
conf_matrix = confusion_matrix(y_resampled, y_pred, normalize='true')
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt=".2%", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Normalized Confusion Matrix")
plt.show()

# **Print Classification Report**
print("\nClassification Report:")
print(classification_report(y_resampled, y_pred))

# **Save Model, Threshold, and Feature Names**
model_data = {
    "model": best_model,
    "best_threshold": best_threshold,
    "feature_columns": list(X.columns)  # Save feature names
}

joblib.dump(model_data, "models/best_logistic_regression_model.pkl")
print(f"Best Logistic Regression model, threshold, and feature columns saved successfully!")







##### MAKE PREDICTIONS ON NEW DATA ======
import pandas as pd
import joblib
from predict_subscription import predict_subscription


predictions_test = predict_subscription('models/best_logistic_regression_model.pkl', 'data/raw/test_file.xlsx', standard_scaler=True)


predictions_test.to_csv('results/test_data_predictions.csv')