class StockScorer:

    def __init__(self, latest):

        self.latest = latest

    def calculate(self):

        score = 0

        reasons = []

        # -------------------------
        # RSI (20 points)
        # -------------------------

        rsi = self.latest["RSI"]

        if 45 <= rsi <= 60:

            score += 20
            reasons.append("✅ Healthy RSI")

        elif 30 <= rsi < 45:

            score += 15
            reasons.append("🟢 RSI Near Oversold")

        elif 60 < rsi <= 70:

            score += 12
            reasons.append("🟡 RSI Slightly High")

        else:

            score += 5
            reasons.append("🔴 RSI Weak")

        # -------------------------
        # EMA Trend (20 points)
        # -------------------------

        if (
            self.latest["EMA20"] >
            self.latest["EMA50"] >
            self.latest["EMA200"]
        ):

            score += 20
            reasons.append("✅ Strong EMA Trend")

        elif self.latest["EMA20"] > self.latest["EMA50"]:

            score += 12
            reasons.append("🟡 EMA Bullish")

        else:

            score += 4
            reasons.append("🔴 EMA Bearish")

        # -------------------------
        # MACD (20)
        # -------------------------

        if self.latest["MACD"] > self.latest["MACD_SIGNAL"]:

            score += 20
            reasons.append("✅ MACD Bullish")

        else:

            score += 5
            reasons.append("🔴 MACD Bearish")

        # -------------------------
        # ADX (15)
        # -------------------------

        if self.latest["ADX"] > 25:

            score += 15
            reasons.append("✅ Strong Trend")

        else:

            score += 8
            reasons.append("🟡 Weak Trend")

        # -------------------------
        # VWAP (10)
        # -------------------------

        if self.latest["Close"] > self.latest["VWAP"]:

            score += 10
            reasons.append("✅ Above VWAP")

        else:

            score += 3
            reasons.append("🔴 Below VWAP")

        # -------------------------
        # Stochastic RSI (10)
        # -------------------------

        if 0.2 < self.latest["STOCH_RSI"] < 0.8:

            score += 10
            reasons.append("✅ Healthy Momentum")

        else:

            score += 4
            reasons.append("🟡 Momentum Extreme")

        # -------------------------
        # Volume (5)
        # -------------------------

        if self.latest["Volume"] > self.latest["AVG_VOLUME"]:

            score += 5
            reasons.append("✅ Strong Buying Volume")

        else:

            score += 2
            reasons.append("🟡 Average Volume")

        # -------------------------
        # Final Recommendation
        # -------------------------

        if score >= 85:

            signal = "🟢 Strong Buy"

        elif score >= 70:

            signal = "🟢 Buy"

        elif score >= 55:

            signal = "🟡 Hold"

        elif score >= 40:

            signal = "🟠 Weak"

        else:

            signal = "🔴 Sell"

        return {

            "score": score,

            "signal": signal,

            "reasons": reasons

        }