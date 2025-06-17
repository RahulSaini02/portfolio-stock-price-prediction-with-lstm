import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd



# --- Validate Weights Sum --- #
def validate_weights(weights: dict) -> bool:
    total = sum(weights.values())
    return abs(total - 1.0) < 0.01


# --- Historical Line Plot --- #
def plot_historical_trend(df, ticker):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#344966"),
        )
    )
    fig.update_layout(
        title=f"{ticker} - Closing Price Trend",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        plot_bgcolor="#F0F4EF",
        paper_bgcolor="#F0F4EF",
    )
    return fig


# --- Bar Plot for Actual vs Predicted --- #
def plot_price_bar(actual, predicted):
    fig = go.Figure(
        data=[
            go.Bar(name="Actual", x=["Price"], y=[actual], marker_color="#344966"),
            go.Bar(
                name="Predicted", x=["Price"], y=[predicted], marker_color="#BFCC94"
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Actual vs Predicted Price",
        yaxis_title="Price ($)",
        plot_bgcolor="#F0F4EF",
        paper_bgcolor="#F0F4EF",
    )
    return fig


# --- Pie Chart for Suggested Portfolio --- #
def plot_pie_chart(suggested_weights):
    labels = list(suggested_weights.keys())
    values = list(suggested_weights.values())
    fig = px.pie(
        names=labels,
        values=values,
        title="Suggested Portfolio Allocation",
        color_discrete_sequence=px.colors.sequential.Tealgrn,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(plot_bgcolor="#F0F4EF", paper_bgcolor="#F0F4EF")
    return fig


def plot_multi_historical_trend(selected_tickers, predictions):
    import plotly.graph_objects as go

    fig = go.Figure()

    for ticker in selected_tickers:
        df = predictions[ticker]["df"].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)

        fig.add_trace(
            go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name=ticker)
        )

    fig.update_layout(
        title="📈 Historical Close Prices (All Selected Stocks)",
        xaxis_title="Date",
        yaxis_title="Close Price ($)",
        plot_bgcolor="#F0F4EF",
        paper_bgcolor="#F0F4EF",
        legend_title="Ticker",
    )
    return fig


def plot_grouped_bar_chart(selected_tickers, predictions):
    import plotly.graph_objects as go

    actual = [predictions[t]["last"] for t in selected_tickers]
    pred = [predictions[t]["pred"] for t in selected_tickers]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=selected_tickers, y=actual, name="Today's Price", marker_color="#344966"
        )
    )
    fig.add_trace(
        go.Bar(
            x=selected_tickers,
            y=pred,
            name="Next Day Predicted Price",
            marker_color="#BFCC94",
        )
    )

    fig.update_layout(
        barmode="group",
        title="📊 Actual vs Predicted Prices (All Stocks)",
        xaxis_title="Ticker",
        yaxis_title="Price ($)",
        plot_bgcolor="#F0F4EF",
        paper_bgcolor="#F0F4EF",
    )
    return fig
