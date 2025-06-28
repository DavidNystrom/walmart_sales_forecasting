# Walmart Sales Forecasting & Inventory Optimization


## 📋 Table of Contents
1. [🚀 Overview](#-overview)  
2. [📦 Data Source](#-data-source)  
3. [⚡ Quickstart](#-quickstart)  
4. [🛠️ Installation](#️-installation)  
5. [📊 Results](#-results)  
6. [🖼️ Screenshots](#️-screenshots)  
7. [☁️ Live Demo](#️-live-demo)  
8. [🔧 Tech Stack](#-tech-stack)  
9. [📈 Future Work](#-future-work)  
10. [🤝 Contributing](#-contributing)  
11. [⚖️ License](#️-license)  

---

## 🚀 Overview
A streamlined analytics pipeline that forecasts Walmart’s weekly sales and guides inventory optimization.  
**Highlights:**  
- **Data:** 2010–2012, 45 stores, 143 weeks  
- **Model:** XGBoost regressor vs. naïve last-week baseline  
- **Performance:** RMSE ≈ \$3,379 (sMAPE ≈ 40.4%) vs. baseline RMSE ≈ \$9,986  
- **Deployment:** Interactive Streamlit dashboard packaged in Docker  

---

## 📦 Data Source
This project uses the [Walmart Store Sales Forecasting][kaggle-link] dataset from Kaggle:
- **train.csv** — Weekly sales by store & department  
- **features.csv** — External factors (weather, CPI, fuel price, promotions)  
- **stores.csv** — Store metadata (type, size)  

> **Attribution:** Provided by Walmart via Kaggle for educational use.

---

## ⚡ Quickstart
```bash
git clone https://github.com/youruser/walmart-sales-forecasting.git
cd walmart-sales-forecasting

# 1. Create & activate virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
python -m src.preprocess
python -m src.feature_engineering
python -m src.hyperparameter_grid
python -m src.model_eval
python -m src.forecast

# 4. Launch dashboard
streamlit run app/dashboard.py
```

## ☁️ Live Demo
Try it live on Streamlit Cloud:  
👉 [Walmart Forecast Dashboard](https://walmartsalesforecasting-8qgin3zjyeghyancrfffux.streamlit.app)  
[![Streamlit][st-badge]][st]

...

[st-badge]: https://static.streamlit.io/badges/streamlit_badge_black_white.svg  
[st]: https://walmartsalesforecasting-8qgin3zjyeghyancrfffux.streamlit.app

