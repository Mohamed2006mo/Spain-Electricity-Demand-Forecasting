# ⚡ Spain Electricity Demand Forecasting

A machine learning pipeline that forecasts Spain's national hourly electricity load (MW), trained on 4 years (2015–2018) of energy market and weather data across 5 major Spanish cities. Deployed as an interactive Streamlit app.

**Live App:** _[https://spain-electricity-demand-forecasting-djtzjjrzgobspk5cs76nua.streamlit.app/]_

---

## 📌 Project Overview

This project predicts Spain's national hourly electricity demand using historical energy market and weather data. It follows a full ML pipeline — from data cleaning and feature engineering to model comparison and deployment — with a strong focus on avoiding data leakage so the model reflects a genuine forecasting task.

**Key principle:** Only information realistically available *in advance* (weather forecasts, day-ahead market forecasts, time features) is used as input — no same-hour actual generation values or the incumbent day-ahead load forecast are included, ensuring the model isn't just mirroring an existing prediction.

## 🗂️ Data Sources

- **Energy data:** Hourly generation, load, and price data for Spain (2015–2018), sourced from ENTSOE and Red Eléctrica de España (REE).
- **Weather data:** Hourly weather data across 5 Spanish cities (Madrid, Barcelona, Valencia, Seville, Bilbao), aggregated nationally.

## 🧠 Modeling Approach

- Four ML algorithms compared: **Random Forest**, **XGBoost**, **Gradient Boosting**, and **K-Nearest Neighbors**.
- **Chronological train/validation/test split (70/15/15)**, respecting the time-series nature of the data (no random shuffling).
- Feature engineering across weather aggregates (mean/std across cities), time-based features (hour, month, day of week, weekend flag), and day-ahead market forecasts.

### Validation Set Results

| Model | RMSE | MAE | R² | MAPE (%) |
|---|---|---|---|---|
| **Random Forest** | 2320.06 | 1716.80 | 0.7600 | 6.07 |
| Gradient Boosting | 2338.75 | 1794.48 | 0.7530 | 6.36 |
| XGBoost | 2392.11 | 1789.76 | 0.7416 | 6.30 |
| KNN | 3330.26 | 2570.65 | 0.4993 | 8.94 |

**Random Forest** was selected as the final deployed model.

### Final Test Set Performance (Random Forest)

| Metric | Value |
|---|---|
| RMSE | 2861.61 MW |
| MAE | 2128.42 MW |
| R² | 0.6115 |
| MAPE | 7.63% |

## 🚀 Deployment

The final trained Random Forest model is deployed as an interactive **Streamlit** app with three pages:

- **🔮 Predict** — Input time, weather, and market conditions to get a live electricity demand prediction, visualized with a gauge chart.
- **📈 Model Insights** — Feature importance chart and full model comparison table.
- **ℹ️ About** — Project background, data sources, and methodology.

## 🛠️ Tech Stack

- **Python**, **Pandas**, **NumPy** — data processing
- **Scikit-learn**, **XGBoost** — model training and evaluation
- **Streamlit** — web app deployment
- **Plotly** — interactive visualizations
- **Joblib** — model serialization

## 📁 Repository Structure

```
Spain-Electricity-Demand-Forecasting/
├── App/
│   └── app.py                      # Streamlit application
├── Notebooks/
│   ├── Spain Electricity Demand Forecasting.ipynb   # Full analysis & modeling notebook
│   ├── random_forest_model.pkl     # Trained model (compressed)
│   └── feature_columns.pkl         # Feature column order for inference
├── requirements.txt
└── README.md
```

## ▶️ Running Locally

```bash
git clone https://github.com/Mohamed2006mo/Spain-Electricity-Demand-Forecasting.git
cd Spain-Electricity-Demand-Forecasting
pip install -r requirements.txt
streamlit run App/app.py
```
