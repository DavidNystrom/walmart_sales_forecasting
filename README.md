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
A retail analytics pipeline that predicts Walmart’s weekly sales.  
Key highlights:
- **Data span:** 2010–2012, 45 stores, 143 weeks  
- **Models:** XGBoost regressor (with naïve last-week sales baseline for comparison)  
- **Performance:** RMSE ~\$3,379 (40.4% sMAPE) vs.\ naive baseline RMSE ~\$9,986  
- **Deployment:** Streamlit dashboard & Docker container  

---

## 📦 Data Source
This project uses the [Walmart Store Sales Forecasting][kaggle-link] dataset from Kaggle, containing:  
- `train.csv`: Weekly sales per store & department  
- `features.csv`: External factors (weather, CPI, fuel price, promotions)  
- `stores.csv`: Store metadata (type, size)  

> **Attribution:** Dataset provided by Walmart via Kaggle for educational use.

---

## ⚡ Quickstart
Get the full pipeline up and running:

```bash
git clone https://github.com/youruser/walmart-sales-forecasting.git
cd walmart-sales-forecasting

# Create & activate virtual env
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run entire pipeline
python -m src.preprocess
python -m src.feature_engineering
python -m src.hyperparameter_grid
python -m src.model_eval
python -m src.forecast

# Launch dashboard
streamlit run app/dashboard.py
