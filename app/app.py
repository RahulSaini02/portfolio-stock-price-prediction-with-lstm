import streamlit as st
import pandas as pd
import os

from src.utils import download_dataset
from src.model import LSTMModel
from src.modeling.utils import load_model
from src.modeling.predict import predict_price
from src.modeling.train import train_lstm_model
from utils import (
    validate_weights,
    plot_multi_historical_trend,
    plot_grouped_bar_chart,
    plot_pie_chart,
)

import json


# --- PAGE CONFIG --- #
st.set_page_config(
    page_title="📈 Portfolio Predictor", layout="wide", initial_sidebar_state="expanded"
)

# --- THEME COLORS --- #
BG_COLOR = "#0D1821"
PRIMARY = "#344966"
ACCENT = "#B4CDED"
TEXT = "#F0F4EF"
SUCCESS = "#BFCC94"

# --- DEFAULTS --- #
DEFAULT_TICKERS = ["AAPL", "TSLA", "MSFT", "GOOGL", "NVDA"]

# --- SESSION STATE --- #
if "predictions" not in st.session_state:
    if os.path.exists("reports/predictions.json"):
        with open("reports/predictions.json", "r") as f:
            raw = json.load(f)
            # Parse dates back from string if needed
            for ticker in raw:
                raw[ticker]["trained_on"] = pd.to_datetime(
                    raw[ticker]["trained_on"]
                ).strftime("%Y-%m-%d")
                try:
                    raw[ticker]["df"] = pd.read_csv(f"data/raw/{ticker}/{ticker}.csv")
                except:
                    raw[ticker]["df"] = pd.DataFrame()
            st.session_state.predictions = raw
    else:
        st.session_state.predictions = {}

# --- SIDEBAR --- #
st.sidebar.title("⚙️ Portfolio Setup")
st.sidebar.markdown("Choose up to 4 stocks and set weights")
# selected_tickers = st.sidebar.multiselect(
#     "Select stocks:", DEFAULT_TICKERS, default=DEFAULT_TICKERS, max_selections=4
# )

# 1. Input new ticker
custom_ticker = st.sidebar.text_input("Add a custom ticker (e.g., NFLX):").upper()
if custom_ticker and custom_ticker not in DEFAULT_TICKERS:
    DEFAULT_TICKERS.append(custom_ticker)

# 2. Multi-select with up to 4 stocks
selected_tickers = st.sidebar.multiselect(
    "Select stocks:", DEFAULT_TICKERS, default=DEFAULT_TICKERS[:4], max_selections=4
)

weights = {}
total_weight = 0
for ticker in selected_tickers:
    weights[ticker] = st.sidebar.slider(f"Weight for {ticker}", 0.0, 1.0, 0.25, 0.05)
    total_weight += weights[ticker]

if not validate_weights(weights):
    st.sidebar.warning("Weights must sum to 1. Adjust sliders.")

st.sidebar.markdown("---")

# --- MAIN PANEL --- #
st.title("📊 Multi-Stock Portfolio Predictor")

if st.sidebar.button("🚀 Train & Predict", disabled=not validate_weights(weights)):
    with st.spinner("Loading models and predicting..."):
        for ticker in selected_tickers:
            model_path = f"models/{ticker}_lstm.pt"
            scaler_path = f"models/{ticker}_scaler.save"
            data_path = f"data/raw/{ticker}/{ticker}.csv"

            if (
                os.path.exists(model_path)
                and os.path.exists(scaler_path)
                and os.path.exists(data_path)
            ):
                df = pd.read_csv(data_path)
                df["Date"] = pd.to_datetime(df["Date"])
                model, scaler = load_model(LSTMModel, model_path, scaler_path)
                predicted_price = float(predict_price(model, scaler, df))
                last_price = float(df["Close"].iloc[-1])
                trained_date = df["Date"].iloc[-1]

                st.session_state.predictions[ticker] = {
                    "df": df,
                    "last": last_price,
                    "pred": predicted_price,
                    "change": round(
                        (predicted_price - last_price) / last_price * 100, 2
                    ),
                    "trained_on": trained_date.strftime("%Y-%m-%d"),
                }
                os.makedirs("reports", exist_ok=True)
                with open("reports/predictions.json", "w") as f:
                    json.dump(st.session_state.predictions, f, indent=2, default=str)
            else:
                df = download_dataset(ticker)
                df.reset_index(inplace=True)

                os.makedirs(f"data/raw/{ticker}", exist_ok=True)
                data_path = f"data/raw/{ticker}/{ticker}.csv"
                df.to_csv(data_path, index=False)

                # --- Artifacts Paths --- #
                model_path = f"models/{ticker}_lstm.pt"
                scaler_path = f"models/{ticker}_scaler.save"
                os.makedirs("models", exist_ok=True)

                train_lstm_model(data_path, model_path, scaler_path)

                model, scaler = load_model(LSTMModel, model_path, scaler_path)
                predicted_price = float(predict_price(model, scaler, df))
                last_price = float(df["Close"].iloc[-1])
                trained_date = df["Date"].iloc[-1]

                st.session_state.predictions[ticker] = {
                    "df": df,
                    "last": last_price,
                    "pred": predicted_price,
                    "change": round(
                        (predicted_price - last_price) / last_price * 100, 2
                    ),
                    "trained_on": trained_date.strftime("%Y-%m-%d"),
                }

                os.makedirs("reports", exist_ok=True)
                with open("reports/predictions.json", "w") as f:
                    json.dump(st.session_state.predictions, f, indent=2, default=str)
if not st.session_state.predictions:
    st.info("Use the sidebar to train and forecast your portfolio.")
    st.stop()

# Historical Trend Chart
st.plotly_chart(
    plot_multi_historical_trend(selected_tickers, st.session_state.predictions),
    use_container_width=True,
)

# Today Vs Tomorrow Predicted Value
st.plotly_chart(
    plot_grouped_bar_chart(selected_tickers, st.session_state.predictions),
    use_container_width=True,
)

for ticker in selected_tickers:
    result = st.session_state.predictions[ticker]
    st.markdown(f"**Model last trained on:** {result['trained_on']}")
    signal = "📈 Rise" if float(result["change"]) > 0 else "📉 Drop"
    st.markdown(f"**{ticker} → {signal}** ({result['change']}%)")

# --- SUGGESTED WEIGHTS --- #
st.markdown("---")
st.subheader("📊 Suggested Portfolio Allocation")

suggested_weights = {}
total_positive_change = sum(
    max(0, r["change"]) for r in st.session_state.predictions.values()
)
for ticker, result in st.session_state.predictions.items():
    gain = max(0, result["change"])
    suggested_weights[ticker] = (
        round(gain / total_positive_change, 2) if total_positive_change else 0
    )

st.plotly_chart(plot_pie_chart(suggested_weights), use_container_width=True)

st.dataframe(
    pd.DataFrame(st.session_state.predictions).T[["last", "pred", "change"]],
    use_container_width=True,
)
