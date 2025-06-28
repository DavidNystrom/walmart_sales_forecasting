# src/model_train.py

import os
import logging
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib
from src import config

# ─── Setup Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def train_and_save_model():
    # 1. Load features
    path = os.path.join(config.PROCESSED_DATA_DIR, "features.csv")
    logging.info(f"Loading features from {path}")
    df = pd.read_csv(path)
    logging.info(f"  → features.csv shape: {df.shape}")

    # 2. Split into X and y
    X = df.drop(columns=["Weekly_Sales"])
    y = df["Weekly_Sales"]

    # 3. Time-ordered train/val split (last 20% as validation)
    cutoff = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_val = y.iloc[:cutoff], y.iloc[cutoff:]
    logging.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

    # 4. Initialize & train model (no early stopping)
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42
    )
    logging.info("Training XGBoost model (no early stopping)...")
    model.fit(X_train, y_train)

    # 5. Evaluate on validation set
    logging.info("Evaluating model on validation set...")
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    rmse = mse ** 0.5
    logging.info(f"Validation RMSE: {rmse:.2f}")

    # 6. Save the trained model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_DIR, "xgb_model.pkl")
    joblib.dump(model, model_path)
    logging.info(f"Saved trained model to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
