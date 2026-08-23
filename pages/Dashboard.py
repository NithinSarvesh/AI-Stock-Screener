import streamlit as st
from datetime import datetime

from stock_fetcher import StockFetcher
from indicators import IndicatorEngine
from charts import ChartBuilder
from ai_analysis import AIAnalyzer
from support_resistance import SupportResistance
from candlestick import CandlePattern

from core.signal_engine import SignalEngine
from rl.v6_inference import get_v6_signal
from rl.decision_engine import HybridDecisionEngine


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# AI RESOURCE
# ============================================================

@st.cache_resource
def get_ai():

    return AIAnalyzer()


# ============================================================
# LOAD STOCK DATA
# ============================================================

def load_stock_data(ticker):

    stock = StockFetcher(ticker)

    data = stock.fetch_all()

    history = data["history"]

    history = IndicatorEngine(
        history
    ).calculate_all()

    return {
        "stock_symbol": stock.symbol,
        "history": history,
        "info": data["info"],
        "fast_info": data["fast_info"],
        "news": data["news"],
        "usd_inr": data["usd_inr"],
    }


# ============================================================
# HEADER
# ============================================================

st.title("📊 Stock Dashboard")

st.caption(
    "Technical analysis, unified signals, trade setup and AI analysis"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📈 Stock Search")


query_ticker = st.query_params.get(
    "ticker",
    "",
)

if query_ticker:

    default_ticker = (
        query_ticker
        .upper()
        .strip()
    )

else:

    default_ticker = "RELIANCE"


ticker = st.sidebar.text_input(
    "Enter Stock Symbol",
    value=default_ticker,
)

ticker = ticker.strip().upper()


popular = [
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "LT",
    "ITC",
    "BHARTIARTL",
    "TATAMOTORS",
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
]


quick = st.sidebar.selectbox(
    "Quick Select",
    ["Select a stock..."] + popular,
)


if quick != "Select a stock...":

    if st.sidebar.button(
        "Load Selected Stock",
        use_container_width=True,
    ):

        st.query_params["ticker"] = quick

        st.rerun()


if st.sidebar.button(
    "🔄 Refresh Data",
    use_container_width=True,
):

    st.rerun()


st.sidebar.divider()


st.sidebar.info(
    """
    **Supported Markets**

    🇮🇳 NSE

    🇮🇳 BSE

    🇺🇸 US

    **Examples**

    RELIANCE

    TCS

    INFY

    AAPL

    NVDA

    TSLA
    """
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    with st.spinner(
        f"Loading market data for {ticker}..."
    ):

        data = load_stock_data(ticker)

except Exception as e:

    st.error(
        f"""
        Unable to load **{ticker}**.

        **Error:**

        `{e}`
        """
    )

    st.stop()


# ============================================================
# PREPARE DATA
# ============================================================

history = data["history"]

history = history.dropna(
    subset=["Close"]
)


if history.empty:

    st.error(
        "No valid historical price data was returned."
    )

    st.stop()


if len(history) < 20:

    st.error(
        "Not enough historical data is available."
    )

    st.stop()


latest = history.iloc[-1]

previous = history.iloc[-2]

info = data["info"]

fast_info = data["fast_info"]

usd_inr = data["usd_inr"]

resolved_symbol = data["stock_symbol"]


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

levels = SupportResistance(
    history
).calculate()


# ============================================================
# CANDLE PATTERN
# ============================================================

pattern = CandlePattern(
    history
).detect()


# ============================================================
# UNIFIED SIGNAL ENGINE
# ============================================================

signal_engine = SignalEngine(
    latest=latest,
    levels=levels,
    pattern=pattern,
)

analysis = signal_engine.analyze()


# ============================================================
# PPO V6 + HYBRID DECISION
# ============================================================

# Keep the existing technical signal as the base signal.
# PPO V6 is an additional reinforcement-learning signal.
# The existing technical trade setup is deliberately not replaced
# until the hybrid logic is validated with paper/live data.

try:

    rl_result = get_v6_signal(
        history,
        current_position=0.0,
    )

    hybrid_engine = HybridDecisionEngine()

    hybrid_result = hybrid_engine.decide(
        quantitative_score=analysis["score"],
        technical_signal=analysis["signal"],
        rl_result=rl_result,
    )

    rl_error = None

except Exception as e:

    rl_result = None
    hybrid_result = None
    rl_error = str(e)


# ============================================================
# COMPANY DATA
# ============================================================

company = info.get(
    "longName",
    ticker,
)

sector = info.get(
    "sector",
    "Unknown",
)


# ============================================================
# PRICE DATA
# ============================================================

current_price = float(
    latest["Close"]
)

previous_close = float(
    previous["Close"]
)

price_change = (
    current_price
    - previous_close
)

price_percent = (
    price_change
    / previous_close
) * 100


# ============================================================
# MARKET CAP
# ============================================================

market_cap = info.get(
    "marketCap"
)


if not market_cap:

    try:

        market_cap = (
            fast_info.get(
                "market_cap"
            )
        )

    except Exception:

        market_cap = None


if market_cap:

    if resolved_symbol.endswith(
        (".NS", ".BO")
    ):

        market_cap_display = (
            f"₹ {market_cap / 1e7:,.2f} Cr"
        )

    else:

        if market_cap >= 1e12:

            market_cap_display = (
                f"$ {market_cap / 1e12:,.2f} T"
            )

        else:

            market_cap_display = (
                f"$ {market_cap / 1e9:,.2f} B"
            )

else:

    market_cap_display = "N/A"


# ============================================================
# PE RATIO
# ============================================================

pe_ratio = info.get(
    "trailingPE",
    "N/A",
)


# ============================================================
# COMPANY HEADER
# ============================================================

st.subheader(
    f"{company} ({resolved_symbol})"
)

st.caption(
    f"Sector: {sector}"
)


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    if resolved_symbol.endswith(
        (".NS", ".BO")
    ):

        st.metric(
            "💰 Current Price",
            f"₹ {current_price:,.2f}",
        )

    else:

        inr_price = (
            current_price * usd_inr
        )

        st.metric(
            "💰 Current Price",
            f"$ {current_price:,.2f}",
        )

        st.caption(
            f"≈ ₹ {inr_price:,.2f}"
        )


with col2:

    currency = "₹"

    if not resolved_symbol.endswith(
        (".NS", ".BO")
    ):

        currency = "$"

    st.metric(
        "📈 Today's Change",
        f"{currency} {price_change:.2f}",
        f"{price_percent:.2f}%",
    )


with col3:

    st.metric(
        "🏦 Market Cap",
        market_cap_display,
    )


with col4:

    if isinstance(
        pe_ratio,
        (int, float),
    ):

        st.metric(
            "📊 PE Ratio",
            f"{pe_ratio:.2f}",
        )

    else:

        st.metric(
            "📊 PE Ratio",
            "N/A",
        )


st.divider()

# ============================================================
# FINAL SYSTEM SIGNAL BANNER
# ============================================================

# Keep the original technical signal available for
# technical analysis and trade-setup logic.
signal = analysis["signal"]

score = analysis["score"]

confidence = analysis["confidence"]


# The final displayed signal comes from the hybrid engine
# when PPO V6 is available.
if hybrid_result is not None:

    final_system_signal = hybrid_result.final_signal

else:

    final_system_signal = signal


if final_system_signal == "STRONG BUY":

    st.success(
        f"🟢 **FINAL SYSTEM SIGNAL: {final_system_signal}**"
    )

elif final_system_signal == "BUY":

    st.success(
        f"🟢 **FINAL SYSTEM SIGNAL: {final_system_signal}**"
    )

elif final_system_signal == "STRONG SELL":

    st.error(
        f"🔴 **FINAL SYSTEM SIGNAL: {final_system_signal}**"
    )

elif final_system_signal == "SELL":

    st.error(
        f"🔴 **FINAL SYSTEM SIGNAL: {final_system_signal}**"
    )

else:

    st.warning(
        f"🟡 **FINAL SYSTEM SIGNAL: {final_system_signal}**"
    )

# ============================================================
# PPO / HYBRID SIGNAL
# ============================================================

st.subheader("🤖 PPO V6 Reinforcement Learning")

if rl_error:

    st.warning(
        f"PPO V6 is unavailable. The base technical signal is still active.\n\n"
        f"`{rl_error}`"
    )

elif rl_result and hybrid_result:

    rl_name = rl_result["name"]
    rl_position = rl_result["position"]

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric("PPO Action", rl_name)

    with r2:
        st.metric("RL Position", f"{rl_position:+.1f}")

    with r3:
        st.metric("Hybrid Score", f"{hybrid_result.final_score:.1f}")

    with r4:
        st.metric("Hybrid Signal", hybrid_result.final_signal)

    if hybrid_result.final_signal in ("STRONG BUY", "BUY"):
        st.success("🟢 PPO + existing analysis currently produce a bullish hybrid signal.")
    elif hybrid_result.final_signal in ("STRONG SELL", "SELL"):
        st.error("🔴 PPO + existing analysis currently produce a bearish hybrid signal.")
    else:
        st.warning("🟡 PPO + existing analysis currently produce a HOLD signal.")

    with st.expander("How the hybrid decision was calculated"):
        for line in hybrid_result.explanation:
            st.write("• " + line)

    if rl_result["action_id"] == 2:
        st.caption("PPO is FLAT here. It is not taking a directional position.")

else:

    st.info("PPO V6 did not return a result.")


sig1, sig2, sig3 = st.columns(3)


with sig1:

    st.metric(
        "Quant Score",
        f"{score:.0f} / 100",
    )


with sig2:

    st.metric(
        "Signal Confidence",
        f"{confidence}%",
    )


with sig3:

    st.metric(
        "Signal Agreement",
        f"{analysis['bullish_count']} Bull / "
        f"{analysis['bearish_count']} Bear",
    )


st.progress(
    score / 100
)


# ============================================================
# SIGNAL EXPLANATION
# ============================================================

st.subheader(
    "🧠 Why This Signal?"
)


trend_bias = analysis[
    "trend"
]["bias"]

momentum_bias = analysis[
    "momentum"
]["bias"]

volume_bias = analysis[
    "volume"
]["bias"]

structure_bias = analysis[
    "structure"
]["bias"]

pattern_bias = analysis[
    "pattern"
]["bias"]


b1, b2, b3, b4, b5 = st.columns(5)


with b1:

    st.metric(
        "Trend",
        trend_bias.upper(),
    )


with b2:

    st.metric(
        "Momentum",
        momentum_bias.upper(),
    )


with b3:

    st.metric(
        "Volume",
        volume_bias.upper(),
    )


with b4:

    st.metric(
        "Structure",
        structure_bias.upper(),
    )


with b5:

    st.metric(
        "Pattern",
        pattern_bias.upper(),
    )


# ============================================================
# CONFLICTS
# ============================================================

conflicts = analysis[
    "conflicts"
]


if conflicts:

    st.warning(
        "⚠️ **Signal Conflict Detected**"
    )

    for conflict in conflicts:

        st.write(
            f"• {conflict}"
        )


# ============================================================
# SIGNAL INTERPRETATION
# ============================================================

if signal == "WAIT":

    st.info(
        """
        **WAIT means the evidence is not aligned enough for an
        active trade.**

        This is intentional.

        A high quantitative score does not automatically force a
        BUY. The engine also checks whether the primary trend,
        momentum, volume and structure agree.
        """
    )


elif signal in (
    "BUY",
    "STRONG BUY",
):

    st.success(
        """
        **Bullish conditions are sufficiently aligned for the
        current signal.**

        The trade setup below is generated from the same signal
        engine that produced this direction.
        """
    )


else:

    st.error(
        """
        **Bearish conditions are sufficiently aligned for the
        current signal.**

        The trade setup below is generated from the same signal
        engine that produced this direction.
        """
    )


st.divider()


# ============================================================
# TECHNICAL CHART
# ============================================================

st.subheader(
    "📈 Technical Chart"
)


chart = ChartBuilder(
    history
)

fig = chart.create_dashboard()


st.plotly_chart(
    fig,
    use_container_width=True,
)


last_date = history.index[-1]

try:

    last_date = last_date.strftime(
        "%d %b %Y"
    )

except Exception:

    pass


st.caption(
    f"Last Updated: {last_date}"
)


st.divider()


# ============================================================
# SCORE BREAKDOWN
# ============================================================

st.subheader(
    "📊 Signal Score Breakdown"
)


components = [
    (
        "Trend",
        analysis["trend"]["score"],
        analysis["trend"]["max_score"],
        analysis["trend"]["bias"],
    ),
    (
        "Momentum",
        analysis["momentum"]["score"],
        analysis["momentum"]["max_score"],
        analysis["momentum"]["bias"],
    ),
    (
        "Volume",
        analysis["volume"]["score"],
        analysis["volume"]["max_score"],
        analysis["volume"]["bias"],
    ),
    (
        "Structure",
        analysis["structure"]["score"],
        analysis["structure"]["max_score"],
        analysis["structure"]["bias"],
    ),
    (
        "Pattern",
        analysis["pattern"]["score"],
        analysis["pattern"]["max_score"],
        analysis["pattern"]["bias"],
    ),
    (
        "Volatility",
        analysis["volatility"]["score"],
        analysis["volatility"]["max_score"],
        analysis["volatility"]["bias"],
    ),
]


for name, value, maximum, bias in components:

    c1, c2, c3 = st.columns(
        [2, 1, 1]
    )

    with c1:

        st.write(
            f"**{name}**"
        )

    with c2:

        st.write(
            f"{value:.1f} / {maximum}"
        )

    with c3:

        st.write(
            bias.upper()
        )

    st.progress(
        min(
            value / maximum,
            1.0,
        )
    )


st.divider()


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

st.subheader(
    "📈 Technical Indicators"
)


i1, i2, i3, i4 = st.columns(4)


with i1:

    st.metric(
        "EMA 20",
        f"{latest['EMA20']:.2f}",
    )

    st.metric(
        "EMA 50",
        f"{latest['EMA50']:.2f}",
    )


with i2:

    st.metric(
        "EMA 200",
        f"{latest['EMA200']:.2f}",
    )

    st.metric(
        "VWAP",
        f"{latest['VWAP']:.2f}",
    )


with i3:

    st.metric(
        "RSI",
        f"{latest['RSI']:.2f}",
    )

    st.metric(
        "ADX",
        f"{latest['ADX']:.2f}",
    )


with i4:

    st.metric(
        "ATR",
        f"{latest['ATR']:.2f}",
    )

    st.metric(
        "Stoch RSI",
        f"{latest['STOCH_RSI']:.2f}",
    )


st.divider()


# ============================================================
# CANDLESTICK PATTERN
# ============================================================

st.subheader(
    "🕯 Candlestick Pattern"
)


c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "Pattern",
        pattern["pattern"],
    )


with c2:

    st.metric(
        "Confidence",
        f"{pattern['confidence']}%",
    )


with c3:

    st.write(
        "**Meaning**"
    )

    st.write(
        pattern["meaning"]
    )


st.divider()


# ============================================================
# TRADE SETUP
# ============================================================

st.subheader(
    "🎯 Trade Setup"
)


setup = analysis[
    "setup"
]


if not setup["valid"]:

    st.info(
        """
        **No active trade setup.**

        The current signal is WAIT because the evidence is not
        sufficiently aligned.

        Do not force an entry simply because the quantitative
        score is high.
        """
    )


else:

    currency = "₹"

    if not resolved_symbol.endswith(
        (".NS", ".BO")
    ):

        currency = "$"


    t1, t2, t3 = st.columns(3)


    with t1:

        st.metric(
            "Entry",
            f"{currency} "
            f"{setup['entry']:.2f}",
        )

        st.metric(
            "Stop Loss",
            f"{currency} "
            f"{setup['stop_loss']:.2f}",
        )


    with t2:

        st.metric(
            "Target 1",
            f"{currency} "
            f"{setup['target1']:.2f}",
        )

        st.metric(
            "Target 2",
            f"{currency} "
            f"{setup['target2']:.2f}",
        )


    with t3:

        st.metric(
            "Target 3",
            f"{currency} "
            f"{setup['target3']:.2f}",
        )

        st.metric(
            "Risk / Share",
            f"{currency} "
            f"{setup['risk_per_share']:.2f}",
        )


    st.metric(
        "Risk : Reward to Target 1",
        f"1 : "
        f"{setup['risk_reward_target1']:.2f}",
    )


    st.caption(
        setup["reason"]
    )


st.divider()


# ============================================================
# KEY LEVELS
# ============================================================

st.subheader(
    "🎯 Key Technical Levels"
)


support = levels["support"]

resistance = levels["resistance"]


currency = "₹"

if not resolved_symbol.endswith(
    (".NS", ".BO")
):

    currency = "$"


c1, c2 = st.columns(2)


with c1:

    st.success(
        f"""
        **Support**

        {currency} {support:.2f}
        """
    )


with c2:

    st.error(
        f"""
        **Resistance**

        {currency} {resistance:.2f}
        """
    )


st.divider()


# ============================================================
# MARKET SUMMARY
# ============================================================

st.subheader(
    "🚦 Market Conditions"
)


m1, m2, m3 = st.columns(3)


with m1:

    if latest["Close"] > latest["EMA20"]:

        st.success(
            "🟢 Price Above EMA20"
        )

    else:

        st.error(
            "🔴 Price Below EMA20"
        )


with m2:

    if latest["MACD"] > latest["MACD_SIGNAL"]:

        st.success(
            "🟢 MACD Bullish"
        )

    else:

        st.error(
            "🔴 MACD Bearish"
        )


with m3:

    if latest["ADX"] >= 25:

        st.success(
            "🟢 Strong Trend"
        )

    else:

        st.warning(
            "🟡 Weak Trend"
        )


st.divider()


# ============================================================
# COMPANY INFORMATION
# ============================================================

st.subheader(
    "🏢 Company Overview"
)


left, right = st.columns(2)


with left:

    st.write(
        "**Company:**",
        company,
    )

    st.write(
        "**Sector:**",
        info.get(
            "sector",
            "N/A",
        ),
    )

    st.write(
        "**Industry:**",
        info.get(
            "industry",
            "N/A",
        ),
    )

    st.write(
        "**Country:**",
        info.get(
            "country",
            "N/A",
        ),
    )

    st.write(
        "**Employees:**",
        info.get(
            "fullTimeEmployees",
            "N/A",
        ),
    )


with right:

    st.write(
        "**Market Cap:**",
        market_cap_display,
    )

    st.write(
        "**PE Ratio:**",
        pe_ratio,
    )

    st.write(
        "**Dividend Yield:**",
        info.get(
            "dividendYield",
            "N/A",
        ),
    )

    st.write(
        "**52 Week High:**",
        info.get(
            "fiftyTwoWeekHigh",
            "N/A",
        ),
    )

    st.write(
        "**52 Week Low:**",
        info.get(
            "fiftyTwoWeekLow",
            "N/A",
        ),
    )


st.divider()


# ============================================================
# AI ANALYSIS
# ============================================================

st.subheader(
    "🤖 AI Trading Analysis"
)


try:

    ai = get_ai()

    with st.spinner(
        "Analyzing market data using AI..."
    ):

        report = ai.analyze(
            resolved_symbol,
            info,
            history,
        )

    st.markdown(
        report
    )


except Exception as e:

    st.error(
        f"""
        AI Analysis Failed.

        `{e}`
        """
    )

    report = (
        "AI Analysis could not be generated."
    )


st.download_button(
    label="📥 Download AI Report",
    data=report,
    file_name=f"{ticker}_AI_Report.txt",
    mime="text/plain",
    use_container_width=True,
)


st.divider()


# ============================================================
# NEWS
# ============================================================

st.subheader(
    "📰 Latest Market News"
)


news = data.get(
    "news",
    [],
)


if not news:

    st.info(
        "No recent news available."
    )

else:

    for article in news[:5]:

        content = article.get(
            "content",
            {},
        )

        title = content.get(
            "title",
            "No Title",
        )

        summary = content.get(
            "summary",
            "No summary available.",
        )

        provider = (
            content.get(
                "provider",
                {},
            ).get(
                "displayName",
                "Unknown",
            )
        )

        date = content.get(
            "pubDate",
            "",
        )

        if date:

            try:

                dt = datetime.fromisoformat(
                    date.replace(
                        "Z",
                        "+00:00",
                    )
                )

                date = dt.strftime(
                    "%d %b %Y • %I:%M %p UTC"
                )

            except Exception:

                pass


        url = (
            content.get(
                "canonicalUrl",
                {},
            ).get(
                "url",
                "",
            )
        )


        with st.container(
            border=True
        ):

            st.markdown(
                f"### 📰 {title}"
            )

            st.caption(
                f"{provider} • {date}"
            )

            st.write(
                summary
            )

            if url:

                st.link_button(
                    "📖 Read Full Article",
                    url,
                    use_container_width=True,
                )


st.divider()


# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    """
    ⚠️ **Disclaimer**

    This application is for educational and research purposes only.

    The quantitative score, technical indicators, unified signal,
    trade setup and AI-generated analysis can be wrong.

    Nothing in this application should be considered financial or
    investment advice.

    Historical performance and backtesting do not guarantee future
    results.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()


f1, f2, f3 = st.columns(3)


with f1:

    st.caption(
        "🇮🇳 Market: NSE / BSE / US"
    )


with f2:

    st.caption(
        "🤖 AI: Groq"
    )


with f3:

    st.caption(
        "📈 Data: Yahoo Finance"
    )


st.caption(
    "Stock Assistant • Version 1.3"
)