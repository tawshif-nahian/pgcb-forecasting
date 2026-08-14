import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="PGCB Power Demand Forecasting Dashboard", page_icon="⚡", layout="wide")

st.title("⚡ PGCB Power Demand Forecasting Dashboard")
st.markdown("Web-Based Remote Monitoring & AI Forecasting System for Bangladesh's Power Grid.")

# ---- Load exported data ----
try:
    data = pd.read_csv("dashboard_data.csv", parse_dates=["datetime"])
    metrics = pd.read_csv("dashboard_metrics.csv")
except FileNotFoundError:
    st.error("⚠️ CSV files not found! Make sure 'dashboard_data.csv' and 'dashboard_metrics.csv' "
              "exist in the same folder as app.py.")
    st.stop()

last_actual_time = data.loc[data["actual"].notna(), "datetime"].max()

# ---- Summary panel ----
st.subheader("📌 Key Grid Metrics & Model Performance")
col1, col2, col3, col4, col5 = st.columns(5)

latest_val = data["actual"].dropna().iloc[-1]
peak_val = data["actual"].max()
avg_val = data["actual"].mean()
xgb_mape = metrics.loc[metrics["model"] == "XGBoost", "MAPE"].values[0]
lstm_mape = metrics.loc[metrics["model"] == "LSTM", "MAPE"].values[0]

col1.metric("Latest Actual Demand", f"{latest_val:,.1f} MW")
col2.metric("Historical Peak Demand", f"{peak_val:,.1f} MW")
col3.metric("Historical Average", f"{avg_val:,.1f} MW")
col4.metric("XGBoost Error (MAPE)", f"{xgb_mape:.2f}%")
col5.metric("LSTM Error (MAPE)", f"{lstm_mape:.2f}%")

st.write("---")

# ---- Interactive controls ----
st.subheader("🎛️ Interactive Forecast Controls")
control_col1, control_col2, control_col3 = st.columns([1, 1, 1])

with control_col1:
    selected_model = st.selectbox(
        "Select Forecasting Model", ["XGBoost", "LSTM"],
        help="Choose which model's forecast and confidence interval to view."
    )
model_key = "xgb" if selected_model == "XGBoost" else "lstm"

with control_col2:
    view_mode = st.radio(
        "View", ["Full history", "Recent + future forecast"], horizontal=True,
        help="'Recent + future forecast' zooms into the last 30 days plus the "
             "72-hour forward forecast, since it's a small sliver against 10 years of history."
    )

with control_col3:
    date_range = st.date_input(
        "Filter Date Range",
        value=(data["datetime"].min().date(), data["datetime"].max().date()),
        min_value=data["datetime"].min().date(),
        max_value=data["datetime"].max().date(),
    )

if view_mode == "Recent + future forecast":
    window_start = last_actual_time - pd.Timedelta(days=30)
    view = data[data["datetime"] >= window_start]
else:
    if len(date_range) == 2:
        start_date, end_date = date_range
        view = data[(data["datetime"].dt.date >= start_date) & (data["datetime"].dt.date <= end_date)]
    else:
        view = data

# ---- Forecast chart ----
st.subheader(f"📈 Hourly Generation Forecast: {selected_model}")

color = "orange" if model_key == "xgb" else "royalblue"
# Neutral gray band: reads as an uncertainty "bracket" around the line at any zoom,
# instead of blending into a same-hue line and disappearing at full-history zoom.
fill_color = "rgba(150,150,150,0.35)"
band_line_color = "rgba(120,120,120,0.5)"

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=view["datetime"], y=view["actual"], name="Actual Historical Generation",
    line=dict(color="black", width=1)
))
fig.add_trace(go.Scatter(
    x=view["datetime"], y=view[f"{model_key}_upper"], line=dict(width=0),
    showlegend=False, hoverinfo="skip"
))
fig.add_trace(go.Scatter(
    x=view["datetime"], y=view[f"{model_key}_lower"], line=dict(width=0.5, color=band_line_color),
    fill="tonexty", fillcolor=fill_color,
    name=f"{selected_model} 95% Confidence Interval", hoverinfo="skip"
))
fig.add_trace(go.Scatter(
    x=view["datetime"], y=view[f"{model_key}_pred"], name=f"{selected_model} Forecast",
    line=dict(color=color, width=1.5)
))

# Mark where real data ends and genuine future forecast begins
if view["datetime"].max() >= last_actual_time:
    fig.add_vline(x=last_actual_time, line_dash="dash", line_color="red")
    fig.add_annotation(
        x=last_actual_time, y=1.05, yref="paper", showarrow=False,
        text="Last actual reading → forecast begins", font=dict(color="red", size=11)
    )

fig.update_layout(
    height=550, xaxis_title="Datetime", yaxis_title="Power Generation (MW)",
    legend=dict(orientation="h", y=1.12), margin=dict(t=80)
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Note: within the historical range, the shown forecast/CI is a **backtested** prediction on "
    "the held-out 20% test period (never seen during training). Beyond the red dashed line is a "
    "genuine 72-hour-ahead recursive forecast into the unobserved future. Tip: at full 10-year "
    "zoom, the forecast line's daily oscillation can visually mask the confidence band — switch "
    "to 'Recent + future forecast' or narrow the date range to see the band clearly on both "
    "sides of the line."
)

with st.expander("🔍 View Detailed Evaluation Metrics Table"):
    st.table(metrics)