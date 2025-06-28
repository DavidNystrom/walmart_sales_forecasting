# src/feature_engineering.py

import os
import logging
import pandas as pd
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def build_features():
    path = os.path.join(config.PROCESSED_DATA_DIR, "combined.csv")
    logging.info(f"Loading combined data from {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    logging.info(f"  → combined.csv shape: {df.shape}")

    # Sort for time-based group ops
    df = df.sort_values(["Store","Dept","Date"])

    # Date parts
    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["DayOfWeek"]  = df["Date"].dt.dayofweek

    # Lag features
    logging.info("Creating lag & rolling features")
    df["lag_1"] = df.groupby(["Store","Dept"])["Weekly_Sales"].shift(1)
    df["rolling_mean_4"] = df.groupby(["Store","Dept"])["Weekly_Sales"] \
                              .shift(1).rolling(window=4).mean()
    df["rolling_std_4"]  = df.groupby(["Store","Dept"])["Weekly_Sales"] \
                              .shift(1).rolling(window=4).std()

    # Fill NaNs from these new features
    df[["lag_1","rolling_mean_4","rolling_std_4"]] = df[["lag_1","rolling_mean_4","rolling_std_4"]].fillna(0)

    # One-hot encode Store Type
    df = pd.get_dummies(df, columns=["Type"], prefix="StoreType", drop_first=True)

    # Convert boolean
    df["IsHoliday"] = df["IsHoliday"].astype(int)

    # Drop irrelevants
    df = df.drop(columns=["Date"])

    out = os.path.join(config.PROCESSED_DATA_DIR, "features.csv")
    df.to_csv(out, index=False)
    logging.info(f"Saved enriched features to {out} (shape: {df.shape})")

if __name__ == "__main__":
    build_features()
