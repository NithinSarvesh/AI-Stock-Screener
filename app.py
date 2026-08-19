import streamlit as st
from datetime import datetime

from config import Config
from stock_fetcher import StockFetcher
from indicators import IndicatorEngine
from charts import ChartBuilder
from ai_analysis import AIAnalyzer


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="🇮🇳 AI Stock Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CACHE
# --------------------------------------------------

@st.cache_resource
def get_ai():

    return AIAnalyzer()


@st.cache_data(ttl=300)
def load_stock(symbol):

    stock = StockFetcher(symbol)

    data = stock.fetch_all()

    history = IndicatorEngine(
        data["history"]
    ).calculate_all()
    history = history.dropna(
    subset=["Close"]
    ).copy()

    return data, history, stock.symbol

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🇮🇳 AI Stock Trading Assistant")

st.caption(
    "Powered by Groq • Yahoo Finance • Plotly • Technical Analysis"
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📈 Stock Search")

with st.sidebar.form("search_form"):

    ticker = st.text_input(
        "Enter Stock Symbol",
        value="RELIANCE"
    ).strip().upper()

    search_clicked = st.form_submit_button(
        "🔍 Search"
    )

st.sidebar.markdown("---")

st.sidebar.subheader("Quick Select")

quick = st.sidebar.selectbox(
    "Popular Stocks",
    [
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
        "SBIN",
        "LT",
        "ITC",
        "BHARTIARTL",
        "TATAMOTORS"
    ]
)

if st.sidebar.button("Use Selected Stock"):

    ticker = quick
st.sidebar.markdown("---")

if st.sidebar.button(
    "🔄 Refresh",
    use_container_width=True
):
    st.cache_data.clear()

    st.rerun()


st.sidebar.markdown("---")

st.sidebar.info(
"""
Supported Markets

✅ NSE

✅ BSE

✅ US

Examples

RELIANCE

TCS

INFY

AAPL

TSLA
"""
)


# --------------------------------------------------
# LOAD STOCK
# --------------------------------------------------

try:

    with st.spinner("Loading Market Data..."):

        data, history, symbol = load_stock(
            ticker
        )
        stock = StockFetcher(ticker)

        info = data["info"]

except Exception as e:

    message = str(e)

    if (
        "Too Many Requests" in message
        or
        "429" in message
    ):

        st.error(
            """
🚫 Yahoo Finance Rate Limit

Too many requests were sent.

Please wait 30–60 seconds
and try again.
"""
        )

    else:

        st.error(message)

    st.stop()
# --------------------------------------------------
# CURRENT VALUES
# --------------------------------------------------

latest = history.iloc[-1]

previous = history.iloc[-2]

current_price = latest["Close"]

price_change = current_price - previous["Close"]

price_percent = (
    price_change /
    previous["Close"]
) * 100

company = info.get(
    "longName",
    ticker
)

sector = info.get(
    "sector",
    "Unknown"
)

market_cap = info.get("marketCap")

if market_cap:

    market_cap = (
        f"₹ {market_cap / 1e7:,.2f} Cr"
    )

else:

    market_cap = "N/A"

pe_ratio = info.get(
    "trailingPE",
    "N/A"
)
# --------------------------------------------------
# COMPANY HEADER
# --------------------------------------------------

st.subheader(
    f"{company} ({symbol})"
)

st.caption(
    f"Sector : {sector}"
)

st.divider()


# --------------------------------------------------
# METRICS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "💰 Current Price",
        f"₹ {current_price:,.2f}"
    )


with col2:

    st.metric(
        "📈 Today's Change",
        f"{price_change:.2f}",
        f"{price_percent:.2f}%"
    )


with col3:

    if market_cap != "N/A":

    st.metric(
        "🏦 Market Cap",
        f"₹ {market_cap:,.0f}"
    )

else:

    st.metric(
        "🏦 Market Cap",
        "N/A"
    ) 


with col4:

    st.metric(
    "🏦 Market Cap",
    market_cap
)

st.divider()


# --------------------------------------------------
# PRICE CHART
# --------------------------------------------------

st.subheader("📈 Technical Chart")

chart = ChartBuilder(history)

fig = chart.create_dashboard()

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()


# --------------------------------------------------
# LAST MARKET UPDATE
# --------------------------------------------------

last_date = history.index[-1]

try:

    last_date = last_date.strftime(
        "%d %b %Y"
    )

except:

    pass


st.caption(
    f"Last Updated : {last_date}"
)
# --------------------------------------------------
# TECHNICAL INDICATORS
# --------------------------------------------------

st.subheader("📊 Technical Indicators")

i1, i2, i3, i4 = st.columns(4)

with i1:

    st.metric(
        "EMA 20",
        f"{latest['EMA20']:.2f}"
    )

    st.metric(
        "EMA 50",
        f"{latest['EMA50']:.2f}"
    )


with i2:

    st.metric(
        "EMA 200",
        f"{latest['EMA200']:.2f}"
    )

    st.metric(
        "VWAP",
        f"{latest['VWAP']:.2f}"
    )


with i3:

    st.metric(
        "RSI",
        f"{latest['RSI']:.2f}"
    )

    st.metric(
        "ADX",
        f"{latest['ADX']:.2f}"
    )


with i4:

    st.metric(
        "ATR",
        f"{latest['ATR']:.2f}"
    )

    st.metric(
        "Stoch RSI",
        f"{latest['STOCH_RSI']:.2f}"
    )

st.divider()


# --------------------------------------------------
# QUICK MARKET SIGNALS
# --------------------------------------------------

st.subheader("🚦 Quick Market Signals")

s1, s2, s3 = st.columns(3)


# RSI Signal

with s1:

    if latest["RSI"] >= 70:

        st.error(
            "🔴 RSI : Overbought"
        )

    elif latest["RSI"] <= 30:

        st.success(
            "🟢 RSI : Oversold"
        )

    else:

        st.info(
            "🟡 RSI : Neutral"
        )


# EMA Trend

with s2:

    if (
        latest["EMA20"] >
        latest["EMA50"] >
        latest["EMA200"]
    ):

        st.success(
            "🟢 Strong Bullish Trend"
        )

    elif (
        latest["EMA20"] <
        latest["EMA50"] <
        latest["EMA200"]
    ):

        st.error(
            "🔴 Strong Bearish Trend"
        )

    else:

        st.warning(
            "🟡 Sideways Trend"
        )


# ADX

with s3:

    if latest["ADX"] >= 25:

        st.success(
            "🟢 Strong Trend"
        )

    else:

        st.warning(
            "🟡 Weak Trend"
        )

st.divider()


# --------------------------------------------------
# SUPPORT & RESISTANCE
# --------------------------------------------------

st.subheader("🎯 Key Technical Levels")

support = min(
    latest["EMA20"],
    latest["EMA50"]
)

resistance = latest["BB_UPPER"]

c1, c2 = st.columns(2)

with c1:

    st.success(
        f"""
Nearest Support

₹ {support:.2f}
"""
    )

with c2:

    st.error(
        f"""
Nearest Resistance

₹ {resistance:.2f}
"""
    )

st.divider()


# --------------------------------------------------
# MARKET SUMMARY
# --------------------------------------------------

st.subheader("📊 Market Summary")

m1, m2, m3 = st.columns(3)

with m1:

    if latest["Close"] > latest["EMA20"]:

        st.success("✅ Price Above EMA20")

    else:

        st.error("❌ Price Below EMA20")


with m2:

    if latest["MACD"] > latest["MACD_SIGNAL"]:

        st.success("✅ MACD Bullish")

    else:

        st.error("❌ MACD Bearish")


with m3:

    if latest["Close"] > latest["VWAP"]:

        st.success("✅ Above VWAP")

    else:

        st.error("❌ Below VWAP")

st.divider()
# --------------------------------------------------
# COMPANY INFORMATION
# --------------------------------------------------

st.subheader("🏢 Company Overview")

left, right = st.columns(2)

with left:

    st.write(
        "**Company:**",
        company
    )

    st.write(
        "**Sector:**",
        info.get("sector", "N/A")
    )

    st.write(
        "**Industry:**",
        info.get("industry", "N/A")
    )

    st.write(
        "**Country:**",
        info.get("country", "N/A")
    )

    st.write(
        "**Employees:**",
        info.get(
            "fullTimeEmployees",
            "N/A"
        )
    )


with right:

    st.write(
        "**Market Cap:**",
        market_cap
    )

    st.write(
        "**PE Ratio:**",
        pe_ratio
    )

    st.write(
        "**Dividend Yield:**",
        info.get(
            "dividendYield",
            "N/A"
        )
    )

    st.write(
        "**52 Week High:**",
        info.get(
            "fiftyTwoWeekHigh",
            "N/A"
        )
    )

    st.write(
        "**52 Week Low:**",
        info.get(
            "fiftyTwoWeekLow",
            "N/A"
        )
    )

st.divider()


# --------------------------------------------------
# AI ANALYSIS
# --------------------------------------------------

st.subheader("🤖 AI Trading Analysis")

ai = get_ai()

try:

    with st.spinner(
        "Analyzing using AI..."
    ):

        report = ai.analyze(
            symbol,
            info,
            history
        )

    st.markdown(report)

except Exception as e:

    st.error(
        f"AI Analysis Failed\n\n{e}"
    )

    report = (
        "AI Analysis could not be generated."
    )


# --------------------------------------------------
# DOWNLOAD REPORT
# --------------------------------------------------

st.download_button(

    label="📥 Download AI Report",

    data=report,

    file_name=f"{ticker}_AI_Report.txt",

    mime="text/plain",

    use_container_width=True

)

st.divider()
# --------------------------------------------------
# LATEST NEWS
# --------------------------------------------------

st.subheader("📰 Latest Market News")

news = data.get("news", [])
st.write(news)

if not news:

    st.info("No recent news available.")

else:

    for article in news[:5]:

        content = article.get("content", {})

        title = content.get(
            "title",
            "No Title"
        )

        summary = content.get(
            "summary",
            "No summary available."
        )

        provider = content.get(
            "provider",
            {}
        ).get(
            "displayName",
            "Unknown"
        )

        date = content.get(
            "pubDate",
            ""
        )

        if date:

            try:

                dt = datetime.fromisoformat(
                    date.replace("Z", "+00:00")
                )

                date = dt.strftime(
                    "%d %b %Y • %I:%M %p UTC"
                )

            except Exception:

                pass

        url = content.get(
            "canonicalUrl",
            {}
        ).get(
            "url",
            ""
        )

        with st.container(border=True):

            st.markdown(f"### 📰 {title}")

            st.caption(
                f"{provider} • {date}"
            )

            st.write(summary)

            if url:

                st.link_button(
                    "📖 Read Full Article",
                    url,
                    use_container_width=True
                )

st.divider()
# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.warning(
    """
⚠️ **Disclaimer**

This application is built for **educational and research purposes only**.

The AI-generated analysis is based on publicly available market data and
technical indicators. It should **NOT** be considered financial or
investment advice.

Always perform your own research and consult a qualified financial advisor
before making investment decisions.
"""
)

st.divider()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

footer1, footer2, footer3 = st.columns(3)

with footer1:

    st.caption("🇮🇳 Market : NSE / BSE / US")

with footer2:

    st.caption("🤖 AI : Groq")

with footer3:

    st.caption("📈 Data : Yahoo Finance")


st.markdown("---")

st.caption(
    "AI Stock Trading Assistant • Version 1.0"
)