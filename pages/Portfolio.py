import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import yfinance as yf
except ImportError:
    yf = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Portfolio",
    page_icon="💼",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("💼 Portfolio")

st.caption(
    "Track holdings, P&L, allocation and portfolio risk."
)

st.divider()


# ============================================================
# SESSION STATE
# ============================================================

if "portfolio_holdings" not in st.session_state:

    st.session_state.portfolio_holdings = [
        {
            "ticker": "LT",
            "quantity": 5.0,
            "avg_price": 3900.0,
        },
        {
            "ticker": "ICICIBANK",
            "quantity": 10.0,
            "avg_price": 1380.0,
        },
    ]


# ============================================================
# STOCK HELPERS
# ============================================================

SECTOR_MAP = {

    "RELIANCE": "Energy",
    "TCS": "Information Technology",
    "INFY": "Information Technology",
    "HCLTECH": "Information Technology",
    "WIPRO": "Information Technology",
    "ICICIBANK": "Financial Services",
    "HDFCBANK": "Financial Services",
    "SBIN": "Financial Services",
    "AXISBANK": "Financial Services",
    "LT": "Industrials",
    "ITC": "Consumer Defensive",
    "MARUTI": "Consumer Cyclical",
    "TATAMOTORS": "Consumer Cyclical",
    "SUNPHARMA": "Healthcare",
}


def resolve_symbol(ticker):

    ticker = str(ticker).strip().upper()

    if ticker.endswith(".NS"):

        return ticker

    return ticker + ".NS"


@st.cache_data(ttl=300)
def get_stock_data(ticker):

    if yf is None:

        return None

    symbol = resolve_symbol(ticker)

    try:

        stock = yf.Ticker(symbol)

        history = stock.history(
            period="6mo",
            auto_adjust=False,
        )

        if history is None or history.empty:

            return None

        current_price = float(
            history["Close"].dropna().iloc[-1]
        )

        previous_price = (
            float(history["Close"].dropna().iloc[-2])
            if len(history) >= 2
            else current_price
        )

        return {

            "price": current_price,

            "previous_price": previous_price,

            "daily_change":
                current_price - previous_price,

            "daily_change_pct":
                (
                    (current_price / previous_price) - 1
                ) * 100
                if previous_price != 0
                else 0.0,

            "history": history,

        }

    except Exception:

        return None


# ============================================================
# ADD HOLDING
# ============================================================

st.subheader("➕ Add Holding")

c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 1])


with c1:

    new_ticker = st.text_input(
        "Stock Symbol",
        placeholder="Example: RELIANCE",
    )


with c2:

    new_quantity = st.number_input(
        "Quantity",
        min_value=0.0,
        value=1.0,
        step=1.0,
    )


with c3:

    new_avg_price = st.number_input(
        "Average Buy Price",
        min_value=0.0,
        value=1000.0,
        step=10.0,
    )


with c4:

    st.write("")

    add_holding = st.button(
        "Add",
        use_container_width=True,
    )


if add_holding:

    ticker = str(
        new_ticker
    ).strip().upper()

    if not ticker:

        st.warning(
            "Enter a stock symbol."
        )

    elif new_quantity <= 0:

        st.warning(
            "Quantity must be greater than zero."
        )

    elif new_avg_price <= 0:

        st.warning(
            "Average price must be greater than zero."
        )

    else:

        existing = None

        for holding in st.session_state.portfolio_holdings:

            if holding["ticker"] == ticker:

                existing = holding
                break

        if existing is not None:

            old_qty = existing["quantity"]
            old_avg = existing["avg_price"]

            total_cost = (
                old_qty * old_avg
                +
                new_quantity * new_avg_price
            )

            total_qty = (
                old_qty + new_quantity
            )

            existing["quantity"] = total_qty

            existing["avg_price"] = (
                total_cost / total_qty
            )

        else:

            st.session_state.portfolio_holdings.append(
                {
                    "ticker": ticker,
                    "quantity": float(new_quantity),
                    "avg_price": float(new_avg_price),
                }
            )

        st.success(
            f"{ticker} added to portfolio."
        )

        st.rerun()


# ============================================================
# PORTFOLIO DATA
# ============================================================

portfolio_rows = []


for holding in st.session_state.portfolio_holdings:

    ticker = holding["ticker"]

    quantity = float(
        holding["quantity"]
    )

    avg_price = float(
        holding["avg_price"]
    )

    stock_data = get_stock_data(
        ticker
    )

    if stock_data is not None:

        current_price = stock_data["price"]

        daily_change = stock_data[
            "daily_change"
        ]

        daily_change_pct = stock_data[
            "daily_change_pct"
        ]

        history = stock_data[
            "history"
        ]

    else:

        current_price = avg_price

        daily_change = 0.0

        daily_change_pct = 0.0

        history = pd.DataFrame()


    invested_value = (
        quantity * avg_price
    )

    current_value = (
        quantity * current_price
    )

    unrealized_pnl = (
        current_value - invested_value
    )

    pnl_pct = (
        unrealized_pnl
        /
        invested_value
        *
        100
        if invested_value != 0
        else 0
    )

    sector = SECTOR_MAP.get(
        ticker,
        "Other",
    )

    portfolio_rows.append(
        {
            "Ticker": ticker,
            "Quantity": quantity,
            "Avg Buy": avg_price,
            "Current Price": current_price,
            "Invested": invested_value,
            "Current Value": current_value,
            "P&L": unrealized_pnl,
            "P&L %": pnl_pct,
            "Daily Change %": daily_change_pct,
            "Sector": sector,
            "_history": history,
        }
    )


portfolio_df = pd.DataFrame(
    portfolio_rows
)


# ============================================================
# EMPTY PORTFOLIO
# ============================================================

if portfolio_df.empty:

    st.info(
        "No portfolio positions have been added yet."
    )

    st.stop()


# ============================================================
# PORTFOLIO METRICS
# ============================================================

total_invested = portfolio_df[
    "Invested"
].sum()

total_value = portfolio_df[
    "Current Value"
].sum()

total_pnl = portfolio_df[
    "P&L"
].sum()

total_pnl_pct = (
    total_pnl
    /
    total_invested
    *
    100
    if total_invested != 0
    else 0
)


# Estimate today's P&L using daily percentage change.

portfolio_df[
    "Today's P&L"
] = (
    portfolio_df["Current Value"]
    *
    portfolio_df["Daily Change %"]
    / 100
)


today_pnl = portfolio_df[
    "Today's P&L"
].sum()


# ============================================================
# OVERVIEW
# ============================================================

st.divider()

st.subheader("📊 Portfolio Overview")


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Portfolio Value",
        f"₹ {total_value:,.2f}",
    )


with c2:

    st.metric(
        "Today's P&L",
        f"₹ {today_pnl:,.2f}",
    )


with c3:

    st.metric(
        "Total P&L",
        f"₹ {total_pnl:,.2f}",
        f"{total_pnl_pct:.2f}%",
    )


with c4:

    st.metric(
        "Holdings",
        len(portfolio_df),
    )


# ============================================================
# HOLDINGS TABLE
# ============================================================

st.divider()

st.subheader("📋 Holdings")


display_columns = [
    "Ticker",
    "Quantity",
    "Avg Buy",
    "Current Price",
    "Invested",
    "Current Value",
    "P&L",
    "P&L %",
    "Sector",
]


display_df = portfolio_df[
    display_columns
].copy()


display_df = display_df.rename(
    columns={
        "Avg Buy": "Avg Buy ₹",
        "Current Price": "Price ₹",
        "Invested": "Invested ₹",
        "Current Value": "Value ₹",
        "P&L": "P&L ₹",
        "P&L %": "P&L %",
    }
)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# REMOVE HOLDING
# ============================================================

st.subheader("🗑️ Manage Holdings")


remove_options = [
    row["ticker"]
    for row in st.session_state.portfolio_holdings
]


if remove_options:

    c1, c2 = st.columns([3, 1])

    with c1:

        remove_ticker = st.selectbox(
            "Remove Stock",
            remove_options,
        )

    with c2:

        st.write("")

        remove_clicked = st.button(
            "Remove",
            use_container_width=True,
        )

        if remove_clicked:

            st.session_state.portfolio_holdings = [
                h
                for h
                in st.session_state.portfolio_holdings
                if h["ticker"] != remove_ticker
            ]

            st.success(
                f"{remove_ticker} removed."
            )

            st.rerun()


# ============================================================
# ALLOCATION
# ============================================================

st.divider()

st.subheader("🥧 Portfolio Allocation")


allocation_df = portfolio_df[
    [
        "Ticker",
        "Current Value",
    ]
].copy()


allocation_df[
    "Allocation %"
] = (
    allocation_df["Current Value"]
    /
    total_value
    *
    100
)


c1, c2 = st.columns(2)


with c1:

    chart_data = allocation_df.set_index(
        "Ticker"
    )["Current Value"]

    st.bar_chart(
        chart_data
    )


with c2:

    allocation_display = allocation_df.copy()

    allocation_display[
        "Allocation %"
    ] = allocation_display[
        "Allocation %"
    ].map(
        lambda x: f"{x:.2f}%"
    )

    st.dataframe(
        allocation_display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SECTOR EXPOSURE
# ============================================================

st.divider()

st.subheader("🏭 Sector Exposure")


sector_df = (
    portfolio_df
    .groupby("Sector")["Current Value"]
    .sum()
    .reset_index()
)


sector_df[
    "Exposure %"
] = (
    sector_df["Current Value"]
    /
    total_value
    *
    100
)


c1, c2 = st.columns(2)


with c1:

    st.bar_chart(
        sector_df.set_index(
            "Sector"
        )["Current Value"]
    )


with c2:

    sector_display = sector_df.copy()

    sector_display[
        "Exposure %"
    ] = sector_display[
        "Exposure %"
    ].map(
        lambda x: f"{x:.2f}%"
    )

    st.dataframe(
        sector_display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CONCENTRATION RISK
# ============================================================

st.divider()

st.subheader("⚠️ Concentration Risk")


largest_position = (
    allocation_df[
        "Allocation %"
    ].max()
)


largest_ticker = (
    allocation_df.loc[
        allocation_df[
            "Allocation %"
        ].idxmax(),
        "Ticker",
    ]
)


if largest_position >= 50:

    risk_label = "HIGH"

    st.error(
        f"High concentration: {largest_ticker} "
        f"represents {largest_position:.1f}% of the portfolio."
    )

elif largest_position >= 30:

    risk_label = "MODERATE"

    st.warning(
        f"Moderate concentration: {largest_ticker} "
        f"represents {largest_position:.1f}% of the portfolio."
    )

else:

    risk_label = "LOW"

    st.success(
        f"Concentration risk is relatively low. "
        f"Largest position: {largest_ticker} "
        f"({largest_position:.1f}%)."
    )


# ============================================================
# RISK METRICS
# ============================================================

st.divider()

st.subheader("📐 Risk Analysis")


returns_series = []


for row in portfolio_rows:

    history = row["_history"]

    if (
        history is not None
        and not history.empty
        and "Close" in history.columns
    ):

        returns = (
            history["Close"]
            .pct_change()
            .dropna()
        )

        if not returns.empty:

            returns_series.append(
                returns
            )


if returns_series:

    returns_df = pd.concat(
        returns_series,
        axis=1,
    )

    returns_df = returns_df.dropna(
        how="all"
    )

    portfolio_returns = (
        returns_df.mean(axis=1)
    )

    annual_volatility = (
        portfolio_returns.std()
        *
        np.sqrt(252)
        *
        100
    )

    cumulative = (
        1 + portfolio_returns
    ).cumprod()

    rolling_max = cumulative.cummax()

    drawdown = (
        cumulative / rolling_max - 1
    )

    max_drawdown = (
        drawdown.min() * 100
    )

    if portfolio_returns.std() != 0:

        sharpe = (
            portfolio_returns.mean()
            /
            portfolio_returns.std()
            *
            np.sqrt(252)
        )

    else:

        sharpe = 0.0

else:

    annual_volatility = 0.0

    max_drawdown = 0.0

    sharpe = 0.0


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Risk Level",
        risk_label,
    )


with c2:

    st.metric(
        "Volatility",
        f"{annual_volatility:.2f}%",
    )


with c3:

    st.metric(
        "Max Drawdown",
        f"{max_drawdown:.2f}%",
    )


with c4:

    st.metric(
        "Sharpe Ratio",
        f"{sharpe:.2f}",
    )


# ============================================================
# PORTFOLIO VALUE VISUALIZATION
# ============================================================

st.divider()

st.subheader("📈 Current Position Values")


value_chart = portfolio_df[
    [
        "Ticker",
        "Invested",
        "Current Value",
    ]
].copy()


value_chart = value_chart.set_index(
    "Ticker"
)


st.bar_chart(
    value_chart
)


# ============================================================
# RESEARCH DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    ⚠️ **Educational / Research Use Only**

    Portfolio values are calculated from market data and the holdings
    entered into this session. This module is not connected to a
    brokerage account and does not execute trades.

    Risk metrics are estimates based on available historical data.
    """
)


st.caption(
    f"Last calculated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)