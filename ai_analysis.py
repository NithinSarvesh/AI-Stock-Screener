from openai import OpenAI
from config import Config


class AIAnalyzer:

    def __init__(self):
        self.client = OpenAI(
            api_key=Config.GROQ_API_KEY,
            base_url=Config.GROQ_BASE_URL
        )

    def analyze(self, symbol, info, df):

        latest = df.iloc[-1]

        prompt = f"""
You are a senior quantitative analyst working for Goldman Sachs.

Analyze the stock below.

Ticker:
{symbol}

Company:
{info.get("longName","Unknown")}

Sector:
{info.get("sector","Unknown")}

Current Price:
{latest["Close"]:.2f}

Indicators

EMA20
{latest["EMA20"]:.2f}

EMA50
{latest["EMA50"]:.2f}

EMA200
{latest["EMA200"]:.2f}

RSI
{latest["RSI"]:.2f}

MACD
{latest["MACD"]:.2f}

MACD Signal
{latest["MACD_SIGNAL"]:.2f}

VWAP
{latest["VWAP"]:.2f}

Market Cap
{info.get("marketCap","Unknown")}

PE Ratio
{info.get("trailingPE","Unknown")}

Return your answer in this format.

# Trend

# Momentum

# RSI Analysis

# MACD Analysis

# EMA Analysis

# Support

# Resistance

# Risk Level

# Short-Term Outlook

# Long-Term Outlook

# Final Recommendation

State

BUY

HOLD

or

SELL

Then explain WHY.

Educational purpose only.
"""

        response = self.client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert financial analyst."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        return response.choices[0].message.content