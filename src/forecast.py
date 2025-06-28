# src/forecast.py

import os
import logging
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def generate_forecast():
    # 1. Load combined data (with Date, Store, Dept, Weekly_Sales, etc.)
    combined = pd.read_csv(
        os.path.join(config.PROCESSED_DATA_DIR, "combined.csv"),
        parse_dates=["Date"]
    )
    logging.info(f"Loaded combined.csv ({combined.shape})")

    # 2. Re-create exactly the same features as in training
    df = combined.sort_values(["Store", "Dept", "Date"]).copy()

    # Date-based
    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["DayOfWeek"]  = df["Date"].dt.dayofweek

    # Lag & rolling
    df["lag_1"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1).fillna(0)
    df["rolling_mean_4"] = (
        df.groupby(["Store", "Dept"])["Weekly_Sales"]
          .shift(1)
          .rolling(4)
          .mean()
          .fillna(0)
    )
    df["rolling_std_4"] = (
        df.groupby(["Store", "Dept"])["Weekly_Sales"]
          .shift(1)
          .rolling(4)
          .std()
          .fillna(0)
    )

    # One-hot encode store type
    df = pd.get_dummies(df, columns=["Type"], prefix="StoreType", drop_first=True)

    # Boolean to int
    df["IsHoliday"] = df["IsHoliday"].astype(int)

    # 3. Load the trained model
    model = joblib.load(os.path.join(config.MODEL_DIR, "xgb_model_best_grid.pkl"))
    logging.info("Loaded model for forecasting")

    # 4. Assemble feature matrix **including** Store & Dept
    #    Must match the exact cols you used for training:
    feature_cols = [
        "Store", "Dept", "IsHoliday", "Size", "Temperature", "Fuel_Price",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
        "CPI", "Unemployment", "Year", "Month", "WeekOfYear", "DayOfWeek",
        "lag_1", "rolling_mean_4", "rolling_std_4",
        "StoreType_B", "StoreType_C"
    ]
    X_all = df[feature_cols]
    logging.info(f"Prepared feature matrix with shape {X_all.shape}")

    # 5. Predict
    df["Prediction"] = model.predict(X_all)
    logging.info("Forecasts generated")

    # 6. Save full forecast
    out_csv = os.path.join(config.PROCESSED_DATA_DIR, "forecast.csv")
    df[["Store","Dept","Date","Weekly_Sales","Prediction"]].to_csv(out_csv, index=False)
    logging.info(f"Saved forecast CSV to {out_csv}")

    # 7. Plot some examples
    vis_dir = os.path.join("visualizations", "forecasts")
    os.makedirs(vis_dir, exist_ok=True)
    for store, dept in [(1,1), (2,10), (3,20)]:
        sub = df[(df.Store==store)&(df.Dept==dept)].set_index("Date")
        plt.figure(figsize=(10,4))
        plt.plot(sub.index, sub.Weekly_Sales, label="Actual")
        plt.plot(sub.index, sub.Prediction, label="Predicted")
        plt.title(f"Store {store} Dept {dept}")
        plt.legend()
        fname = os.path.join(vis_dir, f"forecast_s{store}_d{dept}.png")
        plt.savefig(fname)
        plt.close()
        logging.info(f"Saved plot {fname}")

if __name__ == "__main__":
    generate_forecast()
