from __future__ import annotations

from typing import Any, Dict

import streamlit as st
import yfinance as yf


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fundamentals",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# FORMATTING HELPERS
# ============================================================

def _fmt_number(
    value,
    decimals: int = 2,
) -> str:

    if value is None:
        return "N/A"

    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_percent(
    value,
    already_fraction: bool = True,
    decimals: int = 2,
) -> str:

    if value is None:
        return "N/A"

    try:
        value = float(value)

        if already_fraction:
            value *= 100

        return f"{value:,.{decimals}f}%"

    except (TypeError, ValueError):
        return "N/A"


def _fmt_large_number(
    value,
) -> str:

    if value is None:
        return "N/A"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    absolute = abs(value)

    if absolute >= 1e12:
        return f"₹ {value / 1e12:,.2f} T"

    if absolute >= 1e9:
        return f"₹ {value / 1e9:,.2f} B"

    if absolute >= 1e7:
        return f"₹ {value / 1e7:,.2f} Cr"

    if absolute >= 1e5:
        return f"₹ {value / 1e5:,.2f} L"

    return f"₹ {value:,.0f}"


def _safe_value(
    info: Dict[str, Any],
    *keys,
):

    for key in keys:

        value = info.get(key)

        if value is not None:
            return value

    return None


# ============================================================
# FUNDAMENTALS ENGINE
# ============================================================

class FundamentalsEngine:

    def __init__(
        self,
        info: Dict[str, Any],
    ):

        self.info = info or {}

    # --------------------------------------------------------
    # VALUATION
    # --------------------------------------------------------

    def valuation(self):

        return {
            "Trailing P/E":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "trailingPE",
                    )
                ),

            "Forward P/E":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "forwardPE",
                    )
                ),

            "Price / Book":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "priceToBook",
                    )
                ),

            "Price / Sales":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "priceToSalesTrailing12Months",
                    )
                ),

            "EV / EBITDA":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "enterpriseToEbitda",
                    )
                ),

            "EV / Revenue":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "enterpriseToRevenue",
                    )
                ),

            "PEG Ratio":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "pegRatio",
                        "trailingPegRatio",
                    )
                ),
        }

    # --------------------------------------------------------
    # PROFITABILITY
    # --------------------------------------------------------

    def profitability(self):

        return {
            "Return on Equity":
                _fmt_percent(
                    _safe_value(
                        self.info,
                        "returnOnEquity",
                    )
                ),

            "Return on Assets":
                _fmt_percent(
                    _safe_value(
                        self.info,
                        "returnOnAssets",
                    )
                ),

            "Gross Margin":
                _fmt_percent(
                    _safe_value(
                        self.info,
                        "grossMargins",
                    )
                ),

            "Operating Margin":
                _fmt_percent(
                    _safe_value(
                        self.info,
                        "operatingMargins",
                    )
                ),

            "Net Profit Margin":
                _fmt_percent(
                    _safe_value(
                        self.info,
                        "profitMargins",
                    )
                ),
        }

    # --------------------------------------------------------
    # PER SHARE
    # --------------------------------------------------------

    def per_share(self):

        return {
            "EPS (TTM)":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "trailingEps",
                    )
                ),

            "EPS (Forward)":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "forwardEps",
                    )
                ),

            "Book Value / Share":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "bookValue",
                    )
                ),

            "Revenue / Share":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "revenuePerShare",
                    )
                ),
        }

    # --------------------------------------------------------
    # DIVIDENDS
    # --------------------------------------------------------

    def dividends(self):

        return {
            "Dividend Yield":
                _fmt_percent(
                    _safe_value(
                        self.info,
                        "dividendYield",
                    )
                ),

            "Dividend Rate":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "dividendRate",
                    )
                ),

            "Payout Ratio":
                _fmt_percent(
                    _safe_value(
                        self.info,
                        "payoutRatio",
                    )
                ),

            "5Y Avg Dividend Yield":
                _fmt_percent(
                    _safe_value(
                        self.info,
                        "fiveYearAvgDividendYield",
                    ),
                    already_fraction=False,
                ),
        }

    # --------------------------------------------------------
    # FINANCIAL HEALTH
    # --------------------------------------------------------

    def financial_health(self):

        return {
            "Debt / Equity":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "debtToEquity",
                    )
                ),

            "Current Ratio":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "currentRatio",
                    )
                ),

            "Quick Ratio":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "quickRatio",
                    )
                ),

            "Total Cash":
                _fmt_large_number(
                    _safe_value(
                        self.info,
                        "totalCash",
                    )
                ),

            "Total Debt":
                _fmt_large_number(
                    _safe_value(
                        self.info,
                        "totalDebt",
                    )
                ),
        }

    # --------------------------------------------------------
    # GROWTH
    # --------------------------------------------------------

    def growth(self):

        return {
            "Revenue Growth YoY":
                _fmt_percent(
                    _safe_value(
                        self.info,
                        "revenueGrowth",
                    )
                ),

            "Earnings Growth YoY":
                _fmt_percent(
                    _safe_value(
                        self.info,
                        "earningsGrowth",
                    )
                ),

            "Quarterly Earnings Growth":
                _fmt_percent(
                    _safe_value(
                        self.info,
                        "earningsQuarterlyGrowth",
                    )
                ),
        }

    # --------------------------------------------------------
    # MARKET DATA
    # --------------------------------------------------------

    def market_data(self):

        return {
            "Market Cap":
                _fmt_large_number(
                    _safe_value(
                        self.info,
                        "marketCap",
                    )
                ),

            "Beta":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "beta",
                    )
                ),

            "52 Week High":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "fiftyTwoWeekHigh",
                    )
                ),

            "52 Week Low":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "fiftyTwoWeekLow",
                    )
                ),

            "Average Volume":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "averageDailyVolume10Day",
                    ),
                    decimals=0,
                ),

            "Shares Outstanding":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "sharesOutstanding",
                    ),
                    decimals=0,
                ),

            "Face Value":
                _fmt_number(
                    _safe_value(
                        self.info,
                        "faceValue",
                    )
                ),
        }

    # --------------------------------------------------------
    # SNAPSHOT
    # --------------------------------------------------------

    def snapshot(self):

        return {
            "Valuation":
                self.valuation(),

            "Profitability":
                self.profitability(),

            "Per Share":
                self.per_share(),

            "Dividends":
                self.dividends(),

            "Financial Health":
                self.financial_health(),

            "Growth":
                self.growth(),

            "Market Data":
                self.market_data(),
        }

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    def coverage_ratio(self):

        groups = self.snapshot()

        total = 0
        filled = 0

        for group in groups.values():

            for value in group.values():

                total += 1

                if value != "N/A":
                    filled += 1

        if total == 0:
            return 0.0

        return filled / total


# ============================================================
# FETCH FUNDAMENTALS
# ============================================================

@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def fetch_fundamentals(symbol: str):

    ticker = yf.Ticker(symbol)

    info = ticker.info

    if not info:
        raise ValueError(
            f"No fundamental data returned for {symbol}."
        )

    return info


# ============================================================
# SYMBOL NORMALIZATION
# ============================================================

def normalize_symbol(symbol: str) -> str:

    symbol = symbol.strip().upper()

    if not symbol:
        return ""

    if "." not in symbol:

        symbol = f"{symbol}.NS"

    return symbol


# ============================================================
# HEADER
# ============================================================

st.title("📊 Fundamentals")

st.caption(
    "Fundamental snapshot powered by Yahoo Finance."
)


# ============================================================
# SYMBOL INPUT
# ============================================================

default_symbol = "RELIANCE.NS"

if "selected_symbol" in st.session_state:

    selected = st.session_state.get(
        "selected_symbol"
    )

    if selected:
        default_symbol = normalize_symbol(
            str(selected)
        )


symbol = st.text_input(
    "Stock Symbol",
    value=default_symbol,
    placeholder="Example: RELIANCE.NS",
)

symbol = normalize_symbol(symbol)


# ============================================================
# LOAD BUTTON
# ============================================================

load = st.button(
    "🔄 Load Fundamentals",
    type="primary",
    use_container_width=False,
)


# Automatically load the default symbol.
if (
    load
    or "fundamentals_loaded_symbol" not in st.session_state
):

    st.session_state[
        "fundamentals_loaded_symbol"
    ] = symbol


active_symbol = st.session_state.get(
    "fundamentals_loaded_symbol",
    symbol,
)


# ============================================================
# VALIDATE SYMBOL
# ============================================================

if not active_symbol:

    st.info(
        "Enter an NSE stock symbol to view fundamentals."
    )

    st.stop()


# ============================================================
# FETCH DATA
# ============================================================

with st.spinner(
    f"Loading fundamentals for {active_symbol}..."
):

    try:

        info = fetch_fundamentals(
            active_symbol
        )

    except Exception as exc:

        st.error(
            f"Unable to load fundamentals for "
            f"{active_symbol}."
        )

        st.code(
            str(exc)
        )

        st.stop()


# ============================================================
# COMPANY HEADER
# ============================================================

company_name = info.get(
    "longName"
) or info.get(
    "shortName"
) or active_symbol


sector = info.get(
    "sector",
    "N/A",
)

industry = info.get(
    "industry",
    "N/A",
)


st.subheader(
    f"{company_name}"
)

st.caption(
    f"{active_symbol}  •  {sector}  •  {industry}"
)


# ============================================================
# ENGINE
# ============================================================

engine = FundamentalsEngine(
    info
)

snapshot = engine.snapshot()

coverage = engine.coverage_ratio()


# ============================================================
# DATA QUALITY
# ============================================================

coverage_percent = coverage * 100


if coverage >= 0.70:

    coverage_status = "🟢 Good"

elif coverage >= 0.40:

    coverage_status = "🟡 Partial"

else:

    coverage_status = "🔴 Low"


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Fundamental Coverage",
        f"{coverage_percent:.0f}%",
    )


with c2:

    st.metric(
        "Coverage Status",
        coverage_status,
    )


with c3:

    st.metric(
        "Sector",
        sector,
    )


with c4:

    st.metric(
        "Industry",
        industry,
    )


st.divider()


# ============================================================
# QUICK VALUATION CARDS
# ============================================================

st.subheader(
    "Quick Valuation"
)


valuation = snapshot[
    "Valuation"
]


v1, v2, v3, v4, v5 = st.columns(5)


valuation_items = list(
    valuation.items()
)


columns = [
    v1,
    v2,
    v3,
    v4,
    v5,
]


for column, item in zip(
    columns,
    valuation_items[:5],
):

    label, value = item

    with column:

        st.metric(
            label,
            value,
        )


st.divider()


# ============================================================
# CATEGORY TABS
# ============================================================

tabs = st.tabs(
    [
        "💰 Valuation",
        "📈 Profitability",
        "🧾 Per Share",
        "💵 Dividends",
        "🏦 Financial Health",
        "🚀 Growth",
        "📊 Market Data",
    ]
)


category_order = [
    "Valuation",
    "Profitability",
    "Per Share",
    "Dividends",
    "Financial Health",
    "Growth",
    "Market Data",
]


for tab, category in zip(
    tabs,
    category_order,
):

    with tab:

        data = snapshot[
            category
        ]

        rows = []

        for metric, value in data.items():

            rows.append(
                {
                    "Metric": metric,
                    "Value": value,
                }
            )

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# COMPANY DESCRIPTION
# ============================================================

description = info.get(
    "longBusinessSummary"
)


if description:

    st.divider()

    st.subheader(
        "Company Overview"
    )

    st.write(
        description
    )


# ============================================================
# WARNING
# ============================================================

st.divider()

st.caption(
    "⚠️ Fundamental data is sourced from Yahoo Finance. "
    "Some NSE/BSE fields may be unavailable or delayed. "
    "N/A means Yahoo Finance did not provide that field; "
    "the application does not estimate missing values."
)