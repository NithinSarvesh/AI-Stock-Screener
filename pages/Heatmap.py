import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from data.universe import NIFTY_100, yahoo_symbol
from data.sector_map import get_sector


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sector Heatmap",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ Sector Heatmap")
st.caption(
    "NIFTY-style sector heatmap — colored by today's % change, "
    "sized by today's traded value (price × volume)."
)
st.caption(
    "Note: this sizes boxes by traded value, not free-float market "
    "cap like Kite's heatmap does — market cap for the full universe "
    "would need a slow per-stock lookup. Color (the day's move) is "
    "the part that matters most and is accurate."
)
st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🗺️ Heatmap Settings")

universe_choice = st.sidebar.multiselect(
    "Stocks to include",
    NIFTY_100,
    default=NIFTY_100[:50],
)

refresh = st.sidebar.button("🔄 Refresh", use_container_width=True)

st.sidebar.divider()
st.sidebar.info(
    """
    **Reading the map**

    🟢 Green = up on the day
    🔴 Red = down on the day

    Box size = today's traded value
    (price × volume), grouped by sector.
    """
)


# ============================================================
# DATA FETCH
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_heatmap_data(symbols: tuple) -> pd.DataFrame:

    yahoo_symbols = [yahoo_symbol(s) for s in symbols]

    raw = yf.download(
        yahoo_symbols,
        period="5d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    rows = []

    for symbol, yf_symbol in zip(symbols, yahoo_symbols):

        try:
            frame = raw[yf_symbol] if isinstance(raw.columns, pd.MultiIndex) else raw
            frame = frame.dropna(subset=["Close"])

            if len(frame) < 2:
                continue

            last_close = float(frame["Close"].iloc[-1])
            prev_close = float(frame["Close"].iloc[-2])
            last_volume = float(frame["Volume"].iloc[-1])

            pct_change = ((last_close - prev_close) / prev_close) * 100
            traded_value = last_close * last_volume

            rows.append({
                "Symbol": symbol,
                "Sector": get_sector(symbol),
                "Price": last_close,
                "Change %": round(pct_change, 2),
                "Traded Value": max(traded_value, 1.0),
            })

        except Exception:
            continue

    return pd.DataFrame(rows)


if not universe_choice:
    st.warning("Select at least one stock from the sidebar.")
    st.stop()

if refresh:
    fetch_heatmap_data.clear()

with st.spinner(f"Loading {len(universe_choice)} stocks..."):
    df = fetch_heatmap_data(tuple(universe_choice))

if df.empty:
    st.error("Could not load market data for the selected stocks.")
    st.stop()


# ============================================================
# TREEMAP
# ============================================================

max_abs_change = max(df["Change %"].abs().max(), 1.0)

fig = px.treemap(
    df,
    path=[px.Constant("NIFTY"), "Sector", "Symbol"],
    values="Traded Value",
    color="Change %",
    color_continuous_scale="RdYlGn",
    range_color=[-max_abs_change, max_abs_change],
    color_continuous_midpoint=0,
    custom_data=["Price", "Change %"],
)

fig.update_traces(
    texttemplate="<b>%{label}</b><br>%{customdata[1]:+.2f}%",
    textposition="middle center",
    hovertemplate="<b>%{label}</b><br>Price: %{customdata[0]:.2f}<br>Change: %{customdata[1]:+.2f}%<extra></extra>",
)

fig.update_layout(
    height=650,
    margin=dict(l=10, r=10, t=10, b=10),
    template="plotly_dark",
)

st.plotly_chart(fig, use_container_width=True)

st.divider()


# ============================================================
# GAINERS / LOSERS
# ============================================================

g1, g2 = st.columns(2)

with g1:
    st.subheader("🟢 Top Gainers")
    gainers = df.sort_values("Change %", ascending=False).head(10)
    st.dataframe(
        gainers[["Symbol", "Sector", "Price", "Change %"]],
        use_container_width=True,
        hide_index=True,
    )

with g2:
    st.subheader("🔴 Top Losers")
    losers = df.sort_values("Change %", ascending=True).head(10)
    st.dataframe(
        losers[["Symbol", "Sector", "Price", "Change %"]],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.subheader("🏭 Sector Averages")

sector_avg = (
    df.groupby("Sector")["Change %"]
    .mean()
    .round(2)
    .sort_values(ascending=False)
    .reset_index()
)

st.dataframe(sector_avg, use_container_width=True, hide_index=True)

st.caption("Data: Yahoo Finance, delayed. Not a real-time exchange feed.")
