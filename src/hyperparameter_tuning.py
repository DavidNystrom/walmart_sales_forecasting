# src/hyperparameter_grid.py

import os
import logging
import pandas as pd
import joblib
import itertools
from math import sqrt
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from src import config

# ─── Setup Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def tune_hyperparams_grid():
    # 1. Load features
    df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "features.csv"))
    X = df.drop(columns=["Weekly_Sales"])
    y = df["Weekly_Sales"]

    # 2. Time-ordered split
    cutoff = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_val = y.iloc[:cutoff], y.iloc[cutoff:]
    logging.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # 3. Define a small uniform grid around the previous best
    param_grid = {
        "n_estimators": [175, 200, 225],
        "max_depth":    [5, 6, 7],
        "learning_rate":[0.075, 0.1, 0.125],
        "subsample":    [0.8, 0.9],
        "colsample_bytree":[0.8, 0.9],
    }

    # Prepare for manual grid search
    best_rmse = float("inf")
    best_params = None
    best_model = None

    # 4. Iterate over all combinations
    for ne, md, lr, ss, cs in itertools.product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["learning_rate"],
        param_grid["subsample"],
        param_grid["colsample_bytree"],
    ):
        params = {
            "n_estimators": ne,
            "max_depth": md,
            "learning_rate": lr,
            "subsample": ss,
            "colsample_bytree": cs,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "random_state": 42,
        }
        logging.info(f"Training with params: {params}")
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)  # no extra fit_params needed

        preds = model.predict(X_val)
        current_rmse = rmse(y_val, preds)
        logging.info(f" → Validation RMSE: {current_rmse:.2f}")

        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_params = params
            best_model = model

    # 5. Report & save best
    logging.info(f"Best RMSE: {best_rmse:.2f} with params: {best_params}")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    out_path = os.path.join(config.MODEL_DIR, "xgb_model_best_grid.pkl")
    joblib.dump(best_model, out_path)
    logging.info(f"Saved best model to {out_path}")

if __name__ == "__main__":
    tune_hyperparams_grid()
