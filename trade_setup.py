class TradeSetup:

    def __init__(self, latest):

        self.latest = latest

    def calculate(self):

        close = self.latest["Close"]
        atr = self.latest["ATR"]

        # -------------------------
        # Detect Trend
        # -------------------------

        if self.latest["EMA20"] > self.latest["EMA50"]:

            direction = "🟢 BUY"

            entry = close
            stoploss = close - (1.5 * atr)

            target1 = close + (1.0 * atr)
            target2 = close + (2.0 * atr)
            target3 = close + (3.0 * atr)

        else:

            direction = "🔴 SELL"

            entry = close
            stoploss = close + (1.5 * atr)

            target1 = close - (1.0 * atr)
            target2 = close - (2.0 * atr)
            target3 = close - (3.0 * atr)

        # -------------------------
        # Risk Reward
        # -------------------------

        risk = abs(entry - stoploss)
        reward = abs(target2 - entry)

        if risk == 0:
            rr = 0
        else:
            rr = reward / risk

        # -------------------------
        # Risk Level
        # -------------------------

        if rr >= 2:

            risk_level = "🟢 Low"

        elif rr >= 1.5:

            risk_level = "🟡 Medium"

        else:

            risk_level = "🔴 High"

        # -------------------------
        # Return
        # -------------------------

        return {

            "direction": direction,

            "entry": entry,

            "stoploss": stoploss,

            "target1": target1,

            "target2": target2,

            "target3": target3,

            "rr": rr,

            "risk": risk_level

        }