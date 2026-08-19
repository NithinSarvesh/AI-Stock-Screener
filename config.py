import os
from dotenv import load_dotenv

load_dotenv()


class Config:

    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

    GROQ_BASE_URL = os.environ.get(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1"
)

    MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "openai/gpt-oss-120b"
)

    DEFAULT_TICKER = "AAPL"

    HISTORY_PERIOD = "1y"

    FAST_EMA = 20
    MID_EMA = 50
    LONG_EMA = 200

    RSI_PERIOD = 14

    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

    BOLLINGER_PERIOD = 20

    APP_NAME = "AI Stock Trading Assistant"

    PAGE_ICON = "📈"