# src/model_eval.py

import os
import logging
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def smape(y_true, y_pred):
    """Symmetric mean absolute percentage error."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff  = np.abs(y_pred - y_true)
    # avoid division by zero
    mask = denom != 0
    return np.mean(diff[mask] / denom[mask]) * 100

def evaluate_model():
    # 1. Load validation features + actuals
    df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "features.csv"))
    cutoff = int(len(df) * 0.8)
    X_val = df.drop(columns=["Weekly_Sales"]).iloc[cutoff:]
    y_true = df["Weekly_Sales"].iloc[cutoff:]
    logging.info(f"Loaded validation set: {len(y_true)} rows")

    # 2. Load the best model
    model_path = os.path.join(config.MODEL_DIR, "xgb_model_best_grid.pkl")
    model = joblib.load(model_path)
    logging.info(f"Loaded model from {model_path}")

    # 3. Make predictions
    y_pred = model.predict(X_val)

    # 4. Compute RMSE
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    logging.info(f"Validation RMSE : {rmse:,.2f}")

    # 5. Compute MAPE on non-zero sales only
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        mape = (np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])).mean() * 100
        logging.info(f"Validation MAPE (y>0 only): {mape:.2f}%")
    else:
        logging.info("No non-zero y_true values for MAPE calculation.")

    # 6. Compute sMAPE on all data
    smape_val = smape(y_true.values, y_pred)
    logging.info(f"Validation sMAPE (all data): {smape_val:.2f}%")

    # 7. Residual analysis
    resid = y_true.values - y_pred
    logging.info(f"Residuals → mean: {resid.mean():.2f}, std: {resid.std():.2f}")

if __name__ == "__main__":
    evaluate_model()
