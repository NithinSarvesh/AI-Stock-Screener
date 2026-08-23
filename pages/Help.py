import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Help & Guide",
    page_icon="❓",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("❓ Help & Guide")

st.caption(
    "Understand the signals, scores, AI components and technical "
    "terms used throughout Stock Assistant."
)

st.divider()


# ============================================================
# QUICK START
# ============================================================

st.header("🚀 Quick Start")

st.write(
    """
    Stock Assistant combines several analysis systems to produce a
    final stock signal.

    The main decision pipeline is:

    **Market Data → Technical Indicators → Quantitative Score →
    PPO V6 → Hybrid Decision Engine → Final Signal**
    """
)

st.info(
    """
    **Important:** A signal is an analytical result, not a guarantee
    that a stock will rise or fall. Historical performance does not
    guarantee future performance.
    """
)


# ============================================================
# FINAL SIGNALS
# ============================================================

st.header("🎯 Final Signals")

st.write(
    "The final signal is produced by the Hybrid Decision Engine."
)

signal_data = [
    ("🟢 STRONG BUY", "Strong bullish agreement between the analysis components."),
    ("🟢 BUY", "Bullish conditions exist, but the evidence is weaker than STRONG BUY."),
    ("🟡 HOLD", "There is not enough bullish or bearish evidence to recommend a directional trade."),
    ("🔴 SELL", "Bearish conditions exist."),
    ("🔴 STRONG SELL", "Strong bearish agreement between the analysis components."),
]

for signal, description in signal_data:

    with st.expander(signal):

        st.write(description)


# ============================================================
# QUANTITATIVE SCORE
# ============================================================

st.header("📊 Quantitative Score")

st.write(
    """
    The **Quantitative Score** is a numerical summary produced by the
    Stock Assistant's quantitative analysis.

    It converts several measurable market conditions into a single score.
    """
)

st.markdown(
    """
    ### How to read it

    | Score | General interpretation |
    |---|---|
    | **80 – 100** | Very strong bullish quantitative conditions |
    | **60 – 79** | Bullish quantitative conditions |
    | **40 – 59** | Neutral / mixed conditions |
    | **20 – 39** | Weak quantitative conditions |
    | **0 – 19** | Very weak quantitative conditions |

    **Important:** The quantitative score alone does NOT determine the
    final signal.

    Example:

    **Quant = 84**

    means the quantitative system is strongly bullish.

    But if technical analysis says `WAIT`, the final system can still
    produce `HOLD`.
    """
)


# ============================================================
# TECHNICAL SIGNAL
# ============================================================

st.header("📈 Technical Signal")

st.write(
    """
    The Technical Signal summarizes the current technical setup using
    indicators such as trend, momentum, volume and market structure.
    """
)

technical_data = [
    ("STRONG BUY", "Strong bullish technical setup."),
    ("BUY", "Bullish technical setup."),
    ("WAIT", "The technical setup is not strong enough for a directional recommendation."),
    ("SELL", "Bearish technical setup."),
    ("STRONG SELL", "Strong bearish technical setup."),
]

for signal, description in technical_data:

    with st.expander(signal):

        st.write(description)


st.warning(
    """
    **Technical Safety Gate**

    In the current Hybrid Decision Engine, `WAIT` acts as a safety
    condition against a bullish final recommendation.

    Example:

    Quant Score = 67  
    Technical = WAIT  
    PPO = HALF_LONG

    The mathematical hybrid score is **39.3**, but the system keeps
    the final decision at **HOLD** because the technical setup is
    still WAIT.
    """
)


# ============================================================
# PPO
# ============================================================

st.header("🧠 PPO V6 — Reinforcement Learning")

st.write(
    """
    **PPO** stands for **Proximal Policy Optimization**.

    PPO is a reinforcement-learning algorithm. Instead of being given
    a fixed BUY/SELL rule, the model learns a policy from historical
    market interactions.

    The Stock Assistant uses a trained universal PPO V6 model as an
    additional signal inside the Hybrid Decision Engine.
    """
)


# ============================================================
# PPO OBSERVATION
# ============================================================

st.subheader("🔢 What does PPO observe?")

st.write(
    """
    PPO V6 receives a **30-dimensional observation** representing the
    current market state.

    The observation contains normalized information derived from:

    - Price
    - Volume
    - EMA20
    - EMA50
    - EMA200
    - RSI
    - MACD
    - MACD Signal
    - MACD Histogram
    - Bollinger Bands
    - VWAP
    - ATR
    - ADX
    - OBV
    - Stochastic RSI
    - Recent returns
    - Volatility
    - EMA slopes
    - Trend score
    - Current portfolio position
    """
)

st.info(
    """
    **Observation = the information given to the PPO model at a
    particular point in time.**

    The model uses this information to select one of its available
    actions.
    """
)


# ============================================================
# PPO ACTIONS
# ============================================================

st.subheader("🎮 What does a PPO Action mean?")

st.write(
    """
    PPO V6 has **5 possible actions**.

    These actions represent the position the RL model wants to take.
    """
)

ppo_actions = [
    ("0", "SHORT", "-1.0", "The model wants a full short position."),
    ("1", "HALF_SHORT", "-0.5", "The model wants a half-sized short position."),
    ("2", "FLAT", "0.0", "The model wants no directional position."),
    ("3", "HALF_LONG", "+0.5", "The model wants a half-sized long position."),
    ("4", "LONG", "+1.0", "The model wants a full long position."),
]

st.table(
    {
        "Action ID": [x[0] for x in ppo_actions],
        "PPO Action": [x[1] for x in ppo_actions],
        "Position": [x[2] for x in ppo_actions],
        "Meaning": [x[3] for x in ppo_actions],
    }
)


with st.expander("❓ What does FLAT mean?"):

    st.write(
        """
        **FLAT does NOT mean the stock is going down.**

        It means the PPO model does not currently want a long or short
        directional position.

        Example:

        `PPO = FLAT`

        means:

        > "Based on the market state I see, I do not have enough evidence
        > to take a directional position."

        Therefore, FLAT is essentially a **neutral RL signal**.
        """
    )


with st.expander("❓ What does HALF_LONG mean?"):

    st.write(
        """
        `HALF_LONG` corresponds to a position value of **+0.5**.

        It means the PPO model is moderately bullish rather than
        requesting its maximum long position.

        It does NOT mean "50% probability that the stock will rise."

        It represents the model's selected position level.
        """
    )


with st.expander("❓ What does LONG mean?"):

    st.write(
        """
        `LONG` corresponds to **+1.0**.

        It represents the strongest bullish position available to
        the PPO model.
        """
    )


with st.expander("❓ What does SHORT mean?"):

    st.write(
        """
        `SHORT` corresponds to **-1.0**.

        It represents the strongest bearish position available to
        the PPO model.
        """
    )


# ============================================================
# RL SCORE
# ============================================================

st.header("🤖 RL Score")

st.write(
    """
    The Hybrid Decision Engine converts the PPO position into a
    numerical score from **-100 to +100**.
    """
)

st.code(
    """
PPO Position × 100 = RL Score
"""
)

st.markdown(
    """
    | PPO Position | RL Score | Interpretation |
    |---:|---:|---|
    | **-1.0** | **-100** | Strong bearish RL signal |
    | **-0.5** | **-50** | Moderately bearish RL signal |
    | **0.0** | **0** | Neutral RL signal |
    | **+0.5** | **+50** | Moderately bullish RL signal |
    | **+1.0** | **+100** | Strong bullish RL signal |
    """
)


# ============================================================
# HYBRID ENGINE
# ============================================================

st.header("🔀 Hybrid Decision Engine")

st.write(
    """
    The Hybrid Decision Engine combines three independent components:

    **Quantitative + Technical + PPO**
    """
)

st.subheader("⚖️ Current weights")

st.markdown(
    """
    | Component | Weight |
    |---|---:|
    | Quantitative Score | **40%** |
    | Technical Signal | **35%** |
    | PPO V6 | **25%** |
    """
)

st.code(
    """
Hybrid Score =
    Quantitative × 0.40
    + Technical × 0.35
    + RL × 0.25
"""
)


# ============================================================
# HYBRID SCORE EXAMPLE
# ============================================================

st.subheader("🧮 Example: How is Hybrid Score calculated?")

st.write(
    """
    Suppose:

    - Quantitative Score = **84**
    - Technical Signal = **STRONG BUY**
    - PPO = **HALF_LONG**
    """
)

st.code(
    """
Quantitative = 84
Technical = 100
PPO HALF_LONG = +50

Hybrid =
(84 × 0.40)
+ (100 × 0.35)
+ (50 × 0.25)

= 33.6 + 35 + 12.5

= 81.1
"""
)

st.success(
    """
    Hybrid Score = **81.1**

    This is classified as **STRONG BUY**.
    """
)


# ============================================================
# HYBRID SCORE CLASSIFICATION
# ============================================================

st.subheader("📏 Hybrid Score Classification")

st.markdown(
    """
    | Final Score | Signal |
    |---:|---|
    | **≥ 60** | STRONG BUY |
    | **30 to 59.99** | BUY |
    | **-29.99 to 29.99** | HOLD |
    | **-30 to -59.99** | SELL |
    | **≤ -60** | STRONG SELL |
    """
)


# ============================================================
# WAIT EXAMPLE
# ============================================================

st.subheader("🛡️ Why can a score of 39.3 still become HOLD?")

st.write(
    """
    Consider this example:
    """
)

st.code(
    """
Quantitative = 67
Technical = WAIT
PPO = HALF_LONG (+50)

Hybrid Score =
67 × 0.40
+ 0 × 0.35
+ 50 × 0.25

= 39.3
"""
)

st.warning(
    """
    Normally, 39.3 falls into the BUY range.

    However, the current system has a **Technical Safety Gate**.

    Because the technical signal is `WAIT`, the system prevents the
    bullish BUY recommendation and returns:

    **FINAL = HOLD**
    """
)


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

st.header("📚 Technical Indicator Glossary")


indicator_info = {
    "EMA": """
**Exponential Moving Average**

A moving average that gives more weight to recent prices.

- EMA20 → short-term trend
- EMA50 → medium-term trend
- EMA200 → long-term trend

When price and shorter EMAs are above longer EMAs, the trend is
generally considered more bullish.
""",

    "RSI": """
**Relative Strength Index**

Measures momentum on a scale from 0 to 100.

- Above 70 → traditionally considered overbought
- Below 30 → traditionally considered oversold
- Around 50 → relatively neutral momentum

RSI should not be treated as an automatic BUY/SELL indicator.
""",

    "MACD": """
**Moving Average Convergence Divergence**

A momentum and trend indicator based on moving averages.

The system uses:

- MACD
- MACD Signal
- MACD Histogram

A positive histogram generally indicates bullish momentum relative
to the signal line.
""",

    "Bollinger Bands": """
A volatility indicator consisting of:

- Upper Band
- Middle Band
- Lower Band

The distance between the bands reflects market volatility.
""",

    "VWAP": """
**Volume Weighted Average Price**

Represents the average traded price weighted by volume.

It is commonly used to judge whether price is trading above or
below the volume-weighted average price.
""",

    "ATR": """
**Average True Range**

Measures market volatility.

Higher ATR generally means larger price movement and therefore
greater volatility.
""",

    "ADX": """
**Average Directional Index**

Measures trend strength rather than simply bullish or bearish
direction.

A higher ADX generally indicates a stronger trend.
""",

    "OBV": """
**On-Balance Volume**

Combines price direction with volume to help analyze buying and
selling pressure.
""",

    "Stochastic RSI": """
A momentum indicator derived from RSI.

It is more sensitive than standard RSI and can move quickly between
high and low momentum conditions.
""",
}

for name, description in indicator_info.items():

    with st.expander(name):

        st.markdown(description)


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

st.header("📐 Support & Resistance")

st.markdown(
    """
    **Support**

    A price region where buying interest has historically helped
    prevent price from falling further.

    **Resistance**

    A price region where selling pressure has historically limited
    upward movement.

    These are zones, not guaranteed price barriers.
    """
)


# ============================================================
# BACKTESTING TERMS
# ============================================================

st.header("🧪 Backtesting Terms")

backtest_terms = {
    "Backtest": "Testing a strategy against historical market data.",
    "Buy & Hold": "A benchmark where the stock is bought and held instead of actively trading it.",
    "Return": "The percentage gain or loss produced by a strategy during the tested period.",
    "Sharpe Ratio": "A risk-adjusted performance measure comparing return with volatility. Higher is generally better.",
    "Sortino Ratio": "Similar to Sharpe, but focuses more specifically on downside volatility.",
    "Maximum Drawdown": "The largest peak-to-trough decline in portfolio value during the test.",
    "Trade": "A completed position change generated by the strategy.",
    "Win Rate": "The percentage of trades that were profitable.",
    "Transaction Cost": "The simulated cost applied when buying or selling, such as brokerage or slippage assumptions.",
    "Validation Set": "Historical data used during development/model selection rather than as the final unseen test.",
    "Test Set": "Held-out historical data used to evaluate how the model performs on unseen data.",
}

for name, description in backtest_terms.items():

    with st.expander(name):

        st.write(description)


# ============================================================
# MODEL TERMS
# ============================================================

st.header("🧠 Machine Learning Terms")

ml_terms = {
    "Reinforcement Learning": """
A machine-learning approach where an agent learns by interacting
with an environment and receiving rewards or penalties.
""",

    "PPO": """
Proximal Policy Optimization.

The reinforcement-learning algorithm used to train the Stock
Assistant's V6 trading policy.
""",

    "Policy": """
The model's learned strategy for selecting an action given the
current observation.
""",

    "Observation": """
The numerical representation of the current market state provided
to the PPO model.
""",

    "Action": """
The decision selected by PPO from its five available actions:
SHORT, HALF_SHORT, FLAT, HALF_LONG and LONG.
""",

    "Position": """
The numerical directional exposure represented by the PPO action,
ranging from -1.0 to +1.0.
""",

    "Inference": """
Using an already-trained model to generate a prediction/action for
new market data.
""",

    "Training": """
The process through which PPO learns its policy from historical
market interactions.
""",
}

for name, description in ml_terms.items():

    with st.expander(name):

        st.markdown(description)


# ============================================================
# DATA & MARKET TERMS
# ============================================================

st.header("🌐 Market Data Terms")

market_terms = {
    "Ticker": "The stock's market symbol. Example: RELIANCE, TCS or INFY.",
    "OHLC": "Open, High, Low and Close prices for a market period.",
    "Volume": "The amount of shares traded during a period.",
    "Volatility": "A measure of how much price tends to fluctuate.",
    "Market Regime": "The broader market condition, such as trending, sideways or highly volatile.",
    "Data Leakage": "When information from the future accidentally becomes available to a model during training or evaluation.",
}

for name, description in market_terms.items():

    with st.expander(name):

        st.write(description)


# ============================================================
# HOW TO READ THE DASHBOARD
# ============================================================

st.header("🖥️ How to Read a Stock Analysis")

st.markdown(
    """
    When analysing a stock, do **not** look at only one number.

    Use the system in this order:

    ### 1️⃣ Quantitative Score

    Ask:

    > Is the overall quantitative picture bullish or bearish?

    ### 2️⃣ Technical Signal

    Ask:

    > Is the technical setup actually ready for a trade?

    `WAIT` means the system does not currently see a sufficiently
    strong technical setup.

    ### 3️⃣ PPO Action

    Ask:

    > What position does the reinforcement-learning model prefer?

    For example:

    `HALF_LONG` = moderately bullish RL position.

    `FLAT` = neutral RL position.

    ### 4️⃣ Hybrid Score

    Ask:

    > What happens when all three components are combined?

    ### 5️⃣ Final Signal

    Finally look at:

    **STRONG BUY / BUY / HOLD / SELL / STRONG SELL**

    The final signal is the most important output of the hybrid
    decision system.
    """
)


# ============================================================
# IMPORTANT LIMITATIONS
# ============================================================

st.header("⚠️ Important Limitations")

st.warning(
    """
    Stock Assistant is an educational and research system.

    A BUY signal does not guarantee that the price will increase.

    A HOLD signal does not mean the stock is guaranteed to remain
    unchanged.

    PPO actions are model outputs, not probabilities.

    Quantitative scores are not guaranteed predictions.

    Technical indicators can produce false signals.

    Backtest results can differ significantly from future market
    performance.

    The system should not be treated as a guaranteed automated
    trading system or financial adviser.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Stock Assistant • Help & Guide • Educational / Research Use Only"
)