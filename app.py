import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Stock Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.75;
        margin-top: 4px;
    }

    .feature-card {
        padding: 22px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        min-height: 180px;
    }

    .status-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📈 AI Stock Trading Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Quantitative stock analysis • Technical signals • Risk management •
    Backtesting • AI-powered explanations
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# WELCOME
# ============================================================

st.header("Welcome")

st.write(
    """
    Stock Assistant is being developed as a complete market-analysis
    and trading-research platform.

    The application combines market data, technical indicators,
    quantitative scoring, trade setups, risk management, screening,
    portfolio analysis, backtesting and AI-assisted explanations.
    """
)


# ============================================================
# CURRENT VERSION
# ============================================================

st.subheader("🚀 Version 1.3")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        """
        <div class="status-card">
        <h3>📊 Dashboard</h3>
        <p>Technical stock analysis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="status-card">
        <h3>🔎 Screener</h3>
        <p>Market-wide stock scanning</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="status-card">
        <h3>💼 Portfolio</h3>
        <p>Holdings and risk analysis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        """
        <div class="status-card">
        <h3>🧪 Backtesting</h3>
        <p>Historical strategy testing</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()


# ============================================================
# PLATFORM MODULES
# ============================================================

st.header("🧠 Platform Modules")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="feature-card">

        ### 📊 Technical Analysis

        Analyze:

        - EMA 20 / 50 / 200
        - RSI
        - MACD
        - Bollinger Bands
        - VWAP
        - ATR
        - ADX
        - OBV
        - Stochastic RSI
        - Candlestick patterns
        - Support & resistance

        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:

    st.markdown(
        """
        <div class="feature-card">

        ### 🎯 Signal & Risk Engine

        Future modules will combine:

        - Trend
        - Momentum
        - Volume
        - Market structure
        - Volatility
        - Breakouts
        - Divergences
        - Position sizing
        - Risk/reward

        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:

    st.markdown(
        """
        <div class="feature-card">

        ### 🤖 AI Layer

        The AI layer will eventually provide:

        - Signal explanations
        - Market summaries
        - News interpretation
        - Stock comparison
        - Trade-plan explanations
        - Portfolio analysis
        - Natural-language queries

        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()


# ============================================================
# DEVELOPMENT ROADMAP
# ============================================================

st.header("🛠️ Development Roadmap")

roadmap = [
    ("Phase 1", "Architecture & Dashboard migration", "🟢"),
    ("Phase 2", "Signal & Risk Engine", "🟡"),
    ("Phase 3", "Market Screener", "⚪"),
    ("Phase 4", "Breakout & Divergence Detection", "⚪"),
    ("Phase 5", "Portfolio & Trading Journal", "⚪"),
    ("Phase 6", "Backtesting Engine", "⚪"),
    ("Phase 7", "AI Copilot", "⚪"),
    ("Phase 8", "Testing, optimization & polish", "⚪"),
]

for phase, description, status in roadmap:

    col1, col2, col3 = st.columns([1, 5, 1])

    with col1:
        st.write(f"**{phase}**")

    with col2:
        st.write(description)

    with col3:
        st.write(status)


st.divider()


# ============================================================
# HOW TO USE
# ============================================================

st.header("📚 Navigation")

st.write(
    """
    Use the sidebar to navigate between the different modules.

    **Dashboard** is the main stock-analysis workspace.

    The other modules will be progressively implemented as the
    project moves through Version 1.3 development.
    """
)


# ============================================================
# DISCLAIMER
# ============================================================

st.warning(
    """
    ⚠️ **Educational / Research Use Only**

    This application is intended for educational and research purposes.
    It does not provide financial or investment advice.

    Technical indicators, quantitative scores and AI-generated analysis
    can be wrong. Historical backtests do not guarantee future results.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

left, center, right = st.columns(3)

with left:
    st.caption("📈 Stock Assistant")

with center:
    st.caption("Market Data: Yahoo Finance")

with right:
    st.caption("🤖 AI: Groq / OpenAI-compatible API")

st.caption("Version 1.3 • Architecture rebuild")