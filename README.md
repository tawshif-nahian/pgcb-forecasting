# PGCB Hourly Power Generation Forecasting

An end-to-end time-series forecasting pipeline using XGBoost and LSTM deep learning to predict national grid generation (MW). Features 72-hour recursive forecasting, residual-based 95% confidence intervals, and an interactive Streamlit analytical dashboard.
## Overview

- **Target:** hourly power generation (MW)
- **Horizon:** 1-hour-ahead backtested forecast on a held-out test period, plus a genuine 72-hour recursive forecast into the future
- **Models:** XGBoost (classical ML) and an LSTM (deep learning)
- **Dashboard:** Streamlit + Plotly, reading from precomputed CSV exports

## Results

| Model   | MAE (MW) | RMSE (MW) | MAPE  |
| ------- | -------- | --------- | ----- |
| XGBoost | 263.61   | 389.97    | 2.44% |
| LSTM    | 298.80   | 414.85    | 2.72% |

XGBoost slightly outperforms the LSTM, likely because the hand-engineered lag/rolling features already give it strong signal that the LSTM has to learn from raw sequences on its own.

## Project Structure

```
pgcb-forecasting/
├── notebook.ipynb              # Full pipeline: cleaning, both models, plots, CSV export
├── data/
│   └── PGCB_date_power_demand.xlsx   # Raw dataset
├── dashboard/
│   ├── app.py                  # Streamlit dashboard
│   ├── dashboard_data.csv      # Exported predictions + confidence bands (from notebook)
│   └── dashboard_metrics.csv   # Exported MAE/RMSE/MAPE (from notebook)
├── .gitignore
└── README.md
```

## What the Notebook Does

1. **Cleaning** — collapses duplicate timestamps, removes a corrupted outlier reading, and resamples onto a strict hourly grid (critical: without this, lag features silently misalign since the raw data has mixed 30-minute/hourly sampling and gaps).
2. **Feature engineering** — hour of day, day of week, month, plus lag features at 1 hour, 24 hours, and 168 hours (one week), and a rolling 24-hour mean.
3. **XGBoost model** — trained on the engineered features with a chronological (no-shuffle) 80/20 train-test split.
4. **LSTM model** — trained on raw 48-hour rolling windows instead of hand-picked features, using PyTorch.
5. **Evaluation** — MAE, RMSE, MAPE for both models, plus actual-vs-predicted plots.
6. **Export** — recursively forecasts 72 hours into the future, then writes `dashboard_data.csv` and `dashboard_metrics.csv` for the dashboard to consume.

## Running the Notebook

```bash
pip install pandas numpy xgboost torch scikit-learn matplotlib openpyxl
```

Open `notebook.ipynb` in Jupyter or VS Code and run all cells top to bottom. This regenerates the two CSV files the dashboard depends on.

## Running the Dashboard

```bash
pip install streamlit plotly pandas numpy
cd dashboard
streamlit run app.py
```

Opens automatically at `http://localhost:8501`.

### Dashboard Features

- **Summary panel:** latest reading, historical peak, historical average, and MAPE for both models
- **Model selector:** switch between XGBoost and LSTM forecasts
- **View toggle:** full 10-year history, or a zoomed "recent + future forecast" window
- **Date range filter:** inspect any custom time window
- **Confidence band:** shaded 95% interval (± 1.96 × test-set residual standard deviation) around the forecast
- **Forecast boundary marker:** a red dashed line marks exactly where real data ends and the forward-looking forecast begins

## Limitations

- The confidence interval is a residual-based approximation, not a fully calibrated predictive distribution (quantile regression or Monte Carlo dropout would be more rigorous).
- The 72-hour future forecast is generated recursively (each prediction feeds into the next), so error compounds the further out it goes.
- Neither model currently uses exogenous data (temperature, fuel prices, holidays), which likely influence generation.

## Data Source

[PGCB Hourly Generation Dataset (Bangladesh) — UCI Machine Learning Repository](<https://archive.ics.uci.edu/dataset/1175/pgcb+hourly+generation+dataset+(bangladesh)>)

## Author

MD Tawshif Islam Nahian
Email: ntawshif@gmail.com
