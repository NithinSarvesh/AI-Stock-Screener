import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_fetcher import StockFetcher
from indicators import IndicatorEngine
from charts import ChartBuilder

from features.pivot_levels import PivotCalculator
from features.technical_summary import TechnicalSummaryEngine
from features.technical_events import TechnicalEventDetector


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Technicals",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 Technicals")
st.caption(
    "Kite/Streak-style technical summary — moving average votes, "
    "oscillator votes, pivot support/resistance and recent events."
)
st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧭 Technicals")

query_ticker = st.query_params.get("ticker", "")
default_ticker = query_ticker.upper().strip() if query_ticker else "RELIANCE"

ticker = st.sidebar.text_input("Enter Stock Symbol", value=default_ticker).strip().upper()

timeframe = st.sidebar.radio(
    "Timeframe",
    ["Daily", "Weekly"],
    help=(
        "Yahoo Finance only reliably provides end-of-day history for "
        "NSE/BSE symbols, so intraday timeframes (1min/5min/1hour like "
        "Kite offers) aren't available here without a broker data feed. "
        "Weekly resamples the daily candles."
    ),
)

st.sidebar.divider()
st.sidebar.info(
    """
    **How to read the gauge**

    Each moving average and oscillator casts one vote:
    Buy, Sell or Neutral. The gauge shows the overall lean
    across all of them — it is a vote count, not a guarantee.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_technicals_data(ticker: str, timeframe: str):

    stock = StockFetcher(ticker)
    raw = stock.fetch_all()

    history = raw["history"]

    if timeframe == "Weekly":
        history = history.resample("W").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna()

    history = IndicatorEngine(history).calculate_all()
    history = history.dropna(subset=["Close"])

    return {
        "symbol": stock.symbol,
        "history": history,
        "info": raw["info"],
    }


try:
    with st.spinner(f"Loading {ticker}..."):
        data = load_technicals_data(ticker, timeframe)
except Exception as e:
    st.error(f"Unable to load **{ticker}**.\n\n`{e}`")
    st.stop()

history = data["history"]
resolved_symbol = data["symbol"]
info = data["info"]

if len(history) < 30:
    st.error("Not enough historical data for a reliable technical read.")
    st.stop()

price = float(history["Close"].iloc[-1])
currency = "₹" if resolved_symbol.endswith((".NS", ".BO")) else "$"

company = info.get("longName", ticker)
st.subheader(f"{company} ({resolved_symbol})")
st.metric("Price", f"{currency} {price:,.2f}")
st.divider()


# ============================================================
# TECHNICAL SUMMARY (GAUGE)
# ============================================================

summary_engine = TechnicalSummaryEngine(history)
summary = summary_engine.summary()

st.subheader("📟 Summary")

g1, g2, g3 = st.columns(3)
with g1:
    st.metric("🔴 Bearish", summary["bearish_count"])
with g2:
    st.metric("🟡 Neutral", summary["neutral_count"])
with g3:
    st.metric("🟢 Bullish", summary["bullish_count"])


def render_gauge(gauge_value: float, verdict: str) -> go.Figure:
    """
    Horizontal red -> yellow -> green gradient bar with a triangle
    marker, similar in spirit to Kite's Technicals summary meter.
    gauge_value is expected in [-1, +1].
    """

    n_segments = 120
    xs = np.linspace(-1, 1, n_segments)

    colors = []
    for x in xs:
        if x < 0:
            t = 1 + x  # 0 (bearish) -> 1 (neutral)
            r, g, b = 220, int(60 + t * 150), 60
        else:
            t = x  # 0 (neutral) -> 1 (bullish)
            r, g, b = int(220 - t * 160), int(210 - t * 10), 60
        colors.append(f"rgb({r},{g},{b})")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[2 / n_segments] * n_segments,
        y=["gauge"] * n_segments,
        base=xs - (1 / n_segments),
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=[gauge_value],
        y=["gauge"],
        mode="markers",
        marker=dict(symbol="triangle-down", size=22, color="white",
                    line=dict(width=2, color="black")),
        showlegend=False,
        hovertemplate=f"{verdict}<extra></extra>",
    ))

    fig.update_layout(
        height=110,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-1.05, 1.05], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        template="plotly_dark",
        showlegend=False,
    )

    return fig


st.plotly_chart(
    render_gauge(summary["gauge_value"], summary["verdict"]),
    use_container_width=True,
)

st.markdown(f"**Overall read: {summary['verdict']}**  ·  {summary['bullish_count']} Bullish / "
            f"{summary['neutral_count']} Neutral / {summary['bearish_count']} Bearish "
            f"out of {summary['total']} indicators")

st.divider()


# ============================================================
# SUPPORT / RESISTANCE (CLASSIC PIVOTS)
# ============================================================

st.subheader("🎯 Support & Resistance (Classic Pivots)")

try:
    pivots = PivotCalculator(history).calculate()

    cols = st.columns(7)
    labels_values = [
        ("S3", pivots["s3"]), ("S2", pivots["s2"]), ("S1", pivots["s1"]),
        ("Pivot", pivots["pivot"]),
        ("R1", pivots["r1"]), ("R2", pivots["r2"]), ("R3", pivots["r3"]),
    ]

    for col, (label, value) in zip(cols, labels_values):
        with col:
            st.metric(label, f"{currency} {value:,.2f}")

    zone = PivotCalculator(history).locate_price(price, pivots)
    st.caption(
        f"Basis: {pivots['basis_date'].strftime('%d %b %Y')} candle · "
        f"Current price is **{zone}**"
    )

except ValueError as e:
    st.info(str(e))

st.divider()


# ============================================================
# CROSSOVER SIGNALS
# ============================================================

st.subheader("↔️ Moving Average Crossovers")

crossovers = summary["crossovers"]

c1, c2 = st.columns(2)

with c1:
    label = crossovers["short_term"]
    st.metric("Short Term (5 & 20 SMA CrossOver)", label)

with c2:
    label = crossovers["long_term"]
    st.metric("Long Term (50 & 200 SMA CrossOver)", label)

st.divider()


# ============================================================
# MOVING AVERAGES TABLE
# ============================================================

st.subheader("📈 Moving Averages")

ma_df = pd.DataFrame(summary["moving_averages"])
ma_df["value"] = ma_df["value"].apply(lambda v: f"{currency} {v:,.2f}" if v is not None else "N/A")
ma_df.columns = ["Indicator", "Value", "Signal"]

st.dataframe(ma_df, use_container_width=True, hide_index=True)

st.divider()


# ============================================================
# OSCILLATORS TABLE
# ============================================================

st.subheader("🌊 Oscillators")

osc_df = pd.DataFrame(summary["oscillators"])
osc_df["value"] = osc_df["value"].apply(lambda v: f"{v:,.2f}" if v is not None else "N/A")
osc_df.columns = ["Indicator", "Value", "Signal"]

rsi_val = next((r["value"] for r in summary["oscillators"] if r["name"] == "RSI (14)"), None)
if rsi_val is not None:
    if rsi_val >= 70:
        st.warning(f"RSI (14): {rsi_val:.1f} — Overbought")
    elif rsi_val <= 30:
        st.info(f"RSI (14): {rsi_val:.1f} — Oversold")

st.dataframe(osc_df, use_container_width=True, hide_index=True)

st.divider()


# ============================================================
# TECHNICAL EVENTS
# ============================================================

st.subheader("📰 Technical Events")

events = TechnicalEventDetector(history, lookback_bars=60).detect(max_events=15)

if not events:
    st.info("No notable technical events detected in the recent lookback window.")
else:
    for e in events:
        icon = "🟢" if e["bias"] == "bullish" else ("🔴" if e["bias"] == "bearish" else "🟡")
        st.write(f"{icon} **{e['event']}** — {e['timestamp'].strftime('%d %b %Y')}")

st.divider()


# ============================================================
# CHART
# ============================================================

st.subheader("📊 Chart")

try:
    fig = ChartBuilder(history).create_dashboard()
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Chart could not be rendered: {e}")


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()
st.warning(
    """
    ⚠️ **Research / Educational Use Only**

    This summary is a transparent, open implementation of common
    technical-analysis conventions. It is not identical to any
    broker's proprietary indicator engine and is not financial
    advice.
    """
)
