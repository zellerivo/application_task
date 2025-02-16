from src.data_processing import preprocess
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def predict_subscription(model_path, test_data_path, standard_scaler=False):
    """
    Loads a trained model, preprocesses input data, and predicts subscription outcomes.

    Parameters:
    - model_path (str): Path to the saved model file (e.g., 'models/best_xgboost_model.pkl').
    - test_data_path (str): Path to the new data file (e.g., 'data/raw/test_file.xlsx').
    - standard_scaler (bool): Whether to standardize numerical features (default: False).

    Returns:
    - pd.DataFrame: Original data with an added 'y' column containing predictions (0 or 1).
    """
   

    # **Load the trained model and metadata**
    model_data = joblib.load(model_path)
    feature_columns = model_data["feature_columns"]
    model = model_data["model"]
    best_threshold = model_data["best_threshold"]



    # **Load new data**
    test_data = pd.read_excel(test_data_path)

    # **Preprocess new data**
    new_data_processed = preprocess(test_data)  # Apply same preprocessing as training

    # **One-Hot Encode Categorical Features**
    new_data_processed = pd.get_dummies(new_data_processed, drop_first=True)

    # **Ensure test data has the same feature columns as training**
    new_data_processed = new_data_processed.reindex(columns=feature_columns, fill_value=0)

    # **Optionally standardize numerical features**
    if standard_scaler:
        scaler = StandardScaler()
        new_data_processed = scaler.fit_transform(new_data_processed)

    # **Make predictions (probabilities)**
    predictions = model.predict_proba(new_data_processed)[:, 1]  # Probability of "Yes"

    # **Apply best threshold to convert probabilities to binary outcomes**
    predictions = (predictions >= best_threshold).astype(int)

    # **Attach predictions to original data**
    test_data['y'] = predictions

    return test_data
