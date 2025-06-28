# src/baseline.py

import os
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def compute_baseline():
    # 1. Load the merged CSV (with Date & Weekly_Sales)
    path = os.path.join(config.PROCESSED_DATA_DIR, "combined.csv")
    logging.info(f"Loading combined data from {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    
    # 2. Sort and shift to get last week’s sales
    df = df.sort_values(["Store", "Dept", "Date"])
    df["Last_Week_Sales"] = df.groupby(["Store","Dept"])["Weekly_Sales"].shift(1)
    
    # 3. Drop rows where we have no baseline
    df = df.dropna(subset=["Last_Week_Sales"])
    
    # 4. Compute RMSE
    y_true = df["Weekly_Sales"]
    y_pred = df["Last_Week_Sales"]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    logging.info(f"Baseline RMSE (last-week sales): {rmse:.2f}")

if __name__ == "__main__":
    compute_baseline()
