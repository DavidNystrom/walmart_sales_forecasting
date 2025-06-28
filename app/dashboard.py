# app/dashboard.py
import os, sys

# ─── Add project root so src/ is on PYTHONPATH ──────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from src import config
import numpy as np
from sklearn.metrics import mean_squared_error

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "forecast.csv"), parse_dates=["Date"])
    return df

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(config.MODEL_DIR, "xgb_model_best_grid.pkl"))

def compute_metrics(sub):
    y_true = sub["Weekly_Sales"].values
    y_pred = sub["Prediction"].values
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # sMAPE
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.abs(y_pred - y_true)[denom != 0] / denom[denom != 0]) * 100
    return rmse, smape

def main():
    st.set_page_config(page_title="Walmart Sales Forecast", layout="wide")
    st.title("🛒 Walmart Sales Forecast Dashboard")

    df = load_data()
    stores = sorted(df["Store"].unique())
    store = st.sidebar.selectbox("Select Store", stores)

    depts = sorted(df[df.Store == store]["Dept"].unique())
    dept = st.sidebar.selectbox("Select Department", depts)

    sub = df[(df.Store == store) & (df.Dept == dept)].set_index("Date")
    if sub.empty:
        st.warning("No data for this selection.")
        return

    # Metrics
    rmse, smape = compute_metrics(sub)
    col1, col2 = st.columns(2)
    col1.metric("Validation RMSE", f"${rmse:,.0f}")
    col2.metric("Validation sMAPE", f"{smape:.1f}%")

    # Time series plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sub.index, sub["Weekly_Sales"], label="Actual", linewidth=2)
    ax.plot(sub.index, sub["Prediction"], label="Predicted", linestyle="--")
    ax.set_title(f"Store {store} · Dept {dept}")
    ax.set_ylabel("Weekly Sales")
    ax.legend()
    st.pyplot(fig)

    # Data table
    st.markdown("#### Data Sample")
    st.dataframe(sub[["Weekly_Sales", "Prediction"]].reset_index().tail(10))

if __name__ == "__main__":
    main()
