import os
import sys

import streamlit as st
import pandas as pd
import yfinance as yf

from openai import OpenAI


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from config import Config
from indicators import IndicatorEngine
from rl.v6_inference import get_v6_signal
from rl.decision_engine import HybridDecisionEngine


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Copilot",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🤖 AI Copilot")

st.caption(
    "AI explanations powered by Stock Assistant's quantitative, "
    "technical and PPO V6 analysis."
)

st.divider()


# ============================================================
# DATA FUNCTIONS
# ============================================================

@st.cache_data(ttl=300)
def load_stock_data(ticker):

    symbol = ticker.strip().upper()

    if not symbol.endswith(".NS"):
        symbol = symbol + ".NS"

    stock = yf.Ticker(symbol)

    df = stock.history(
        period="1y",
        auto_adjust=False,
    )

    if df is None or df.empty:
        raise ValueError(
            f"No market data found for {symbol}."
        )

    try:

        info = stock.info

    except Exception:

        info = {}

    return symbol, df, info


# ============================================================
# QUANT SCORE
# ============================================================

def get_quant_score(df):

    """
    Build a simple quantitative score from the indicators
    already calculated by Stock Assistant.

    Score range: 0 ... 100
    50 = neutral
    """

    if df is None or df.empty:
        return 50.0

    row = df.iloc[-1]

    score = 50.0

    close = float(row["Close"])

    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    ema200 = float(row["EMA200"])

    rsi = float(row["RSI"])

    macd = float(row["MACD"])
    macd_signal = float(row["MACD_SIGNAL"])

    vwap = float(row["VWAP"])

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    if close > ema20:
        score += 8
    else:
        score -= 8

    if ema20 > ema50:
        score += 8
    else:
        score -= 8

    if ema50 > ema200:
        score += 8
    else:
        score -= 8

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    if rsi >= 60:
        score += 7

    elif rsi >= 50:
        score += 3

    elif rsi <= 40:
        score -= 7

    elif rsi < 50:
        score -= 3

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if macd > macd_signal:
        score += 8
    else:
        score -= 8

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    if close > vwap:
        score += 5
    else:
        score -= 5

    return float(
        max(
            0.0,
            min(
                100.0,
                score,
            ),
        )
    )
# ============================================================
# TECHNICAL SIGNAL
# ============================================================

def derive_technical_signal(row):

    bullish = 0
    bearish = 0

    close = float(row["Close"])

    # Trend
    if close > float(row["EMA20"]):
        bullish += 1
    else:
        bearish += 1

    if float(row["EMA20"]) > float(row["EMA50"]):
        bullish += 1
    else:
        bearish += 1

    if float(row["EMA50"]) > float(row["EMA200"]):
        bullish += 1
    else:
        bearish += 1

    # RSI
    rsi = float(row["RSI"])

    if rsi >= 55:
        bullish += 1
    elif rsi <= 45:
        bearish += 1

    # MACD
    if float(row["MACD"]) > float(row["MACD_SIGNAL"]):
        bullish += 1
    else:
        bearish += 1

    if bullish >= 4:
        return "STRONG BUY"

    if bullish >= 3:
        return "BUY"

    if bearish >= 4:
        return "STRONG SELL"

    if bearish >= 3:
        return "SELL"

    return "WAIT"


# ============================================================
# AI CLIENT
# ============================================================

@st.cache_resource
def get_ai_client():

    api_key = getattr(
        Config,
        "GROQ_API_KEY",
        None,
    )

    base_url = getattr(
        Config,
        "GROQ_BASE_URL",
        None,
    )

    if not api_key:
        return None

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


# ============================================================
# AI PROMPT
# ============================================================

def ask_ai(
    ticker,
    info,
    latest,
    quant_score,
    technical_signal,
    rl_result,
    hybrid_decision,
    question,
):

    client = get_ai_client()

    if client is None:

        return (
            "AI API is not configured.\n\n"
            "The quantitative, technical and PPO V6 "
            "analysis is still available below."
        )

    company = info.get(
        "longName",
        ticker,
    )

    sector = info.get(
        "sector",
        "Unknown",
    )

    prompt = f"""
You are the AI Copilot of a stock-analysis research platform.

You are NOT the trading model.

Your job is to explain the structured analysis produced by
the Stock Assistant engines. Do not invent indicators,
prices, signals or facts that are not provided.

Ticker:
{ticker}

Company:
{company}

Sector:
{sector}

CURRENT MARKET DATA
-------------------

Price:
{float(latest["Close"]):.2f}

EMA20:
{float(latest["EMA20"]):.2f}

EMA50:
{float(latest["EMA50"]):.2f}

EMA200:
{float(latest["EMA200"]):.2f}

RSI:
{float(latest["RSI"]):.2f}

MACD:
{float(latest["MACD"]):.4f}

MACD Signal:
{float(latest["MACD_SIGNAL"]):.4f}

VWAP:
{float(latest["VWAP"]):.2f}

ATR:
{float(latest["ATR"]):.2f}

ADX:
{float(latest["ADX"]):.2f}

Quantitative Score:
{quant_score:.1f}/100

Technical Signal:
{technical_signal}

PPO V6:
{rl_result.get("name", "UNKNOWN")}

PPO Position:
{rl_result.get("position", 0.0):+.1f}

Hybrid Score:
{hybrid_decision.final_score:.1f}/100

Hybrid Final Signal:
{hybrid_decision.final_signal}

USER QUESTION
-------------

{question}

INSTRUCTIONS
------------

Answer clearly and practically.

Explain:

1. What the current system is seeing.
2. Which signals agree.
3. Which signals disagree.
4. What the PPO contributes.
5. What could invalidate the setup.
6. Important risks.

Do not claim that the system can predict the future.

Do not say that the stock WILL rise or WILL fall.

Do not give guaranteed returns.

If the evidence is mixed, explicitly say that it is mixed.

End with:

"System decision: {hybrid_decision.final_signal}"

Educational / research use only.
"""

    response = client.chat.completions.create(

        model=Config.MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise financial research "
                    "assistant. Explain evidence rather than "
                    "making unsupported predictions."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        temperature=0.2,
    )

    return response.choices[0].message.content


# ============================================================
# STOCK INPUT
# ============================================================

st.subheader("📊 Stock Context")

c1, c2 = st.columns([2, 1])

with c1:

    ticker = st.text_input(
        "Stock Symbol",
        value="RELIANCE",
        placeholder="Example: RELIANCE",
    )


with c2:

    load_button = st.button(
        "🔄 Analyze Stock",
        use_container_width=True,
    )


# ============================================================
# ANALYZE STOCK
# ============================================================

if load_button or ticker:

    try:

        with st.spinner(
            "Loading market data and running analysis..."
        ):

            symbol, raw_df, info = load_stock_data(
                ticker
            )

            features = IndicatorEngine(
                raw_df.copy()
            ).calculate_all()

            features = (
                features
                .replace(
                    [float("inf"), float("-inf")],
                    pd.NA,
                )
                .dropna()
            )

            if features.empty:

                raise ValueError(
                    "Not enough data after indicator calculation."
                )

            latest = features.iloc[-1]

            quant_score = get_quant_score(
                features
            )

            technical_signal = (
                derive_technical_signal(
                    latest
                )
            )

            rl_result = get_v6_signal(
                raw_df.copy()
            )

            engine = HybridDecisionEngine()

            hybrid_decision = engine.decide(
                quantitative_score=quant_score,
                technical_signal=technical_signal,
                rl_result=rl_result,
            )


        # ====================================================
        # SIGNAL SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            f"🎯 {symbol.replace('.NS', '')} System Analysis"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Price",
                f"₹{float(latest['Close']):,.2f}",
            )

        with c2:

            st.metric(
                "Quant Score",
                f"{quant_score:.1f}/100",
            )

        with c3:

            st.metric(
                "PPO V6",
                rl_result["name"],
            )

        with c4:

            st.metric(
                "Hybrid Decision",
                hybrid_decision.final_signal,
            )


        # ====================================================
        # ENGINE BREAKDOWN
        # ====================================================

        st.subheader("🧠 Decision Engine Breakdown")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Quantitative",
                f"{quant_score:.1f}",
            )

        with c2:

            st.metric(
                "Technical",
                technical_signal,
            )

        with c3:

            st.metric(
                "PPO Position",
                f"{rl_result['position']:+.1f}",
            )


        st.progress(
            min(
                max(
                    hybrid_decision.final_score / 100,
                    0.0,
                ),
                1.0,
            )
        )

        st.caption(
            f"Hybrid score: "
            f"{hybrid_decision.final_score:.1f}/100"
        )


        # ====================================================
        # INDICATORS
        # ====================================================

        with st.expander(
            "📈 Technical Indicators",
            expanded=False,
        ):

            indicator_data = pd.DataFrame(
                {
                    "Indicator": [
                        "EMA20",
                        "EMA50",
                        "EMA200",
                        "RSI",
                        "MACD",
                        "MACD Signal",
                        "VWAP",
                        "ATR",
                        "ADX",
                    ],
                    "Value": [
                        float(latest["EMA20"]),
                        float(latest["EMA50"]),
                        float(latest["EMA200"]),
                        float(latest["RSI"]),
                        float(latest["MACD"]),
                        float(latest["MACD_SIGNAL"]),
                        float(latest["VWAP"]),
                        float(latest["ATR"]),
                        float(latest["ADX"]),
                    ],
                }
            )

            st.dataframe(
                indicator_data,
                use_container_width=True,
                hide_index=True,
            )


        # ====================================================
        # ASK COPILOT
        # ====================================================

        st.divider()

        st.subheader("🤖 Ask the Copilot")

        question = st.text_area(
            "Your question",
            placeholder=(
                "Example: Why did the system give this signal?"
            ),
            height=110,
        )


        # ====================================================
        # SUGGESTED QUESTIONS
        # ====================================================

        st.caption(
            "Suggested questions:"
        )

        suggestions = [
            "Why did the system give this signal?",
            "Explain the PPO V6 contribution.",
            "Which indicators are bullish or bearish?",
            "What could invalidate this setup?",
            "What are the biggest risks?",
            "Explain the technical setup simply.",
        ]

        cols = st.columns(3)

        for i, suggestion in enumerate(
            suggestions
        ):

            with cols[i % 3]:

                if st.button(
                    suggestion,
                    key=f"suggestion_{i}",
                    use_container_width=True,
                ):

                    question = suggestion


        # ====================================================
        # ASK
        # ====================================================

        if st.button(
            "🤖 Ask AI Copilot",
            use_container_width=True,
        ):

            if not question.strip():

                st.warning(
                    "Enter a question first."
                )

            else:

                with st.spinner(
                    "AI Copilot is analyzing the system output..."
                ):

                    try:

                        answer = ask_ai(
                            ticker=symbol,
                            info=info,
                            latest=latest,
                            quant_score=quant_score,
                            technical_signal=technical_signal,
                            rl_result=rl_result,
                            hybrid_decision=hybrid_decision,
                            question=question,
                        )

                        st.markdown(
                            "### 🤖 Copilot Analysis"
                        )

                        st.markdown(
                            answer
                        )

                    except Exception as e:

                        st.error(
                            f"AI request failed: {e}"
                        )


        # ====================================================
        # SYSTEM EXPLANATION
        # ====================================================

        st.divider()

        st.subheader(
            "🔍 System Explanation"
        )

        for item in hybrid_decision.explanation:

            st.write(
                f"• {item}"
            )


    except Exception as e:

        st.error(
            f"Unable to analyze {ticker.upper()}: {e}"
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    ⚠️ **Educational / Research Use Only**

    AI Copilot explains signals generated by the Stock Assistant.
    It does not guarantee future price movements and does not
    constitute financial or investment advice.
    """
)