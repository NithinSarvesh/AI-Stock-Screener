import streamlit as st
import pandas as pd

from stock_fetcher import StockFetcher
from indicators import IndicatorEngine
from support_resistance import SupportResistance
from candlestick import CandlePattern

from core.signal_engine import SignalEngine
from rl.v6_inference import get_v6_signal
from rl.decision_engine import HybridDecisionEngine


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Market Screener",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# STOCK UNIVERSE
# ============================================================

DEFAULT_STOCKS = [
    "RELIANCE",
    "TCS",
    "INFY",
    "ICICIBANK",
    "SBIN",
    "LT",
    "ITC",
]


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🔎 AI Market Screener")

st.caption(
    "Multi-stock quantitative, technical and PPO V6 reinforcement-learning analysis"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Screener Settings")

selected_stocks = st.sidebar.multiselect(
    "Stocks to scan",
    DEFAULT_STOCKS,
    default=DEFAULT_STOCKS,
)

custom_stocks = st.sidebar.text_input(
    "Add other stocks",
    placeholder="Example: HDFCBANK, AXISBANK, MARUTI",
)

if custom_stocks.strip():

    extra_stocks = [
        ticker.strip().upper()
        for ticker in custom_stocks.split(",")
        if ticker.strip()
    ]

    selected_stocks = list(
        dict.fromkeys(
            selected_stocks + extra_stocks
        )
    )

run_scan = st.sidebar.button(
    "🚀 Run AI Screener",
    use_container_width=True,
)

st.sidebar.divider()

st.sidebar.info(
    """
    **Pipeline**

    Quantitative Score
    ↓
    Technical Signal
    ↓
    PPO V6
    ↓
    Hybrid Decision

    PPO is treated as an additional
    signal, not as a guaranteed
    trading decision.
    """
)


# ============================================================
# HELPER
# ============================================================

def scan_stock(ticker):

    try:

        # ----------------------------------------------------
        # DOWNLOAD DATA
        # ----------------------------------------------------

        stock = StockFetcher(ticker)

        data = stock.fetch_all()

        history = data["history"]

        if history is None or history.empty:
            raise ValueError("No historical data returned.")

        # ----------------------------------------------------
        # INDICATORS
        # ----------------------------------------------------

        history = IndicatorEngine(
            history
        ).calculate_all()

        history = history.dropna(
            subset=["Close"]
        )

        if len(history) < 50:
            raise ValueError(
                "Not enough historical data."
            )

        latest = history.iloc[-1]

        # ----------------------------------------------------
        # SUPPORT / RESISTANCE
        # ----------------------------------------------------

        levels = SupportResistance(
            history
        ).calculate()

        # ----------------------------------------------------
        # CANDLE PATTERN
        # ----------------------------------------------------

        pattern = CandlePattern(
            history
        ).detect()

        # ----------------------------------------------------
        # EXISTING SIGNAL ENGINE
        # ----------------------------------------------------

        signal_engine = SignalEngine(
            latest=latest,
            levels=levels,
            pattern=pattern,
        )

        analysis = signal_engine.analyze()

        # ----------------------------------------------------
        # PPO V6
        # ----------------------------------------------------

        rl_result = get_v6_signal(
            history,
            current_position=0.0,
        )

        # ----------------------------------------------------
        # HYBRID ENGINE
        # ----------------------------------------------------

        hybrid_engine = HybridDecisionEngine()

        hybrid_result = hybrid_engine.decide(
            quantitative_score=analysis["score"],
            technical_signal=analysis["signal"],
            rl_result=rl_result,
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return {
            "Ticker": ticker,

            "Price": float(
                latest["Close"]
            ),

            "Quant Score": round(
                float(analysis["score"]),
                1,
            ),

            "Technical": analysis[
                "signal"
            ],

            "PPO": rl_result[
                "name"
            ],

            "Position": rl_result[
                "position"
            ],

            "Hybrid Score": round(
                float(
                    hybrid_result.final_score
                ),
                1,
            ),

            "Final Signal": hybrid_result.final_signal,

            "Confidence": analysis[
                "confidence"
            ],

            "Bull": analysis[
                "bullish_count"
            ],

            "Bear": analysis[
                "bearish_count"
            ],
        }

    except Exception as e:

        return {
            "Ticker": ticker,
            "Price": None,
            "Quant Score": None,
            "Technical": "ERROR",
            "PPO": "ERROR",
            "Position": None,
            "Hybrid Score": None,
            "Final Signal": "ERROR",
            "Confidence": None,
            "Bull": None,
            "Bear": None,
            "Error": str(e),
        }


# ============================================================
# RUN SCAN
# ============================================================

if run_scan:

    if not selected_stocks:

        st.warning(
            "Select at least one stock."
        )

        st.stop()

    st.subheader(
        "📡 Scanning Market"
    )

    results = []

    progress = st.progress(0)

    status = st.empty()

    total = len(
        selected_stocks
    )

    for index, ticker in enumerate(
        selected_stocks
    ):

        status.write(
            f"Analyzing **{ticker}**..."
        )

        result = scan_stock(
            ticker
        )

        results.append(
            result
        )

        progress.progress(
            (index + 1) / total
        )

    status.success(
        f"Scan complete — {total} stocks analyzed."
    )

    df = pd.DataFrame(
        results
    )

    # ========================================================
    # REMOVE ERROR COLUMN FROM MAIN TABLE
    # ========================================================

    display_columns = [
        "Ticker",
        "Price",
        "Quant Score",
        "Technical",
        "PPO",
        "Position",
        "Hybrid Score",
        "Final Signal",
        "Confidence",
        "Bull",
        "Bear",
    ]

    display_df = df[
        [
            column
            for column in display_columns
            if column in df.columns
        ]
    ].copy()

    # ========================================================
    # TOP SUMMARY
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Screener Results"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        strong_buy_count = (
            display_df[
                "Final Signal"
            ]
            .eq("STRONG BUY")
            .sum()
        )

        st.metric(
            "🟢 Strong Buy",
            int(
                strong_buy_count
            ),
        )

    with col2:

        buy_count = (
            display_df[
                "Final Signal"
            ]
            .eq("BUY")
            .sum()
        )

        st.metric(
            "🟢 Buy",
            int(
                buy_count
            ),
        )

    with col3:

        hold_count = (
            display_df[
                "Final Signal"
            ]
            .eq("HOLD")
            .sum()
        )

        st.metric(
            "🟡 Hold",
            int(
                hold_count
            ),
        )

    with col4:

        sell_count = (
            display_df[
                "Final Signal"
            ]
            .isin(
                [
                    "SELL",
                    "STRONG SELL",
                ]
            )
            .sum()
        )

        st.metric(
            "🔴 Sell",
            int(
                sell_count
            ),
        )

    # ========================================================
    # MAIN TABLE
    # ========================================================

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Hybrid score combines quantitative analysis, technical analysis "
        "and PPO V6. It is a ranking signal, not a guaranteed prediction."
    )

    # ========================================================
    # RANKING
    # ========================================================

    st.divider()

    st.subheader(
        "🏆 Hybrid Signal Ranking"
    )

    ranking = display_df.dropna(
        subset=["Hybrid Score"]
    ).sort_values(
        "Hybrid Score",
        ascending=False,
    )

    if not ranking.empty:

        st.dataframe(
            ranking[
                [
                    "Ticker",
                    "Quant Score",
                    "Technical",
                    "PPO",
                    "Hybrid Score",
                    "Final Signal",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("📊 Hybrid Score Comparison")

        chart_data = (
            ranking[
                [
                    "Ticker",
                    "Hybrid Score",
                ]
            ]
            .set_index("Ticker")
        )

        st.bar_chart(
            chart_data
        )

        # ----------------------------------------------------
        # BEST CURRENT SIGNAL
        # ----------------------------------------------------

        best = ranking.iloc[0]

        if best["Final Signal"] in (
            "STRONG BUY",
            "BUY",
        ):

            st.success(
                f"""
                🏆 **Highest Current Hybrid Score: {best['Ticker']}**
                Hybrid Score: **{best['Hybrid Score']}**

                Final Signal: **{best['Final Signal']}**
                """
            )

        elif best["Final Signal"] == "HOLD":

            st.info(
                f"""
                🏆 Highest hybrid score is
                **{best['Ticker']}**, but the current
                system classification is **HOLD**.
                """
            )

    # ========================================================
    # ERROR DETAILS
    # ========================================================

    errors = df[
        df["Final Signal"] == "ERROR"
    ]

    if not errors.empty:

        st.divider()

        st.subheader(
            "⚠️ Scan Errors"
        )

        for _, row in errors.iterrows():

            st.error(
                f"{row['Ticker']}: "
                f"{row.get('Error', 'Unknown error')}"
            )

else:

    # ========================================================
    # INITIAL SCREEN
    # ========================================================

    st.info(
        """
        ### Ready to scan

        Select stocks from the sidebar and click:

        **🚀 Run AI Screener**

        The screener will run:

        **Quantitative Analysis → Technical Analysis → PPO V6 → Hybrid Decision**
        """
    )

    st.subheader(
        "🧠 What this module demonstrates"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            ### 📊 Quantitative

            Each stock receives a
            quantitative score from
            the existing signal engine.
            """
        )

    with c2:

        st.markdown(
            """
            ### 📈 Technical

            Trend, momentum, volume,
            structure and candlestick
            conditions are evaluated.
            """
        )

    with c3:

        st.markdown(
            """
            ### 🤖 Reinforcement Learning

            PPO V6 evaluates the
            30-feature market state and
            selects a portfolio position.
            """
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    ⚠️ **Research / Educational Use Only**

    Screener signals are generated from historical and
    current market data. They are not guaranteed to be
    profitable and should not be treated as financial advice.

    PPO V6 is an additional model signal, not a guarantee
    of future price direction.
    """
)