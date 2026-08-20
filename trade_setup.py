class TradeSetup:

    def __init__(self, latest, support, resistance):

        self.latest = latest
        self.support = support
        self.resistance = resistance

    def calculate(self):

        close = self.latest["Close"]

        # -------------------------
        # Detect Trend
        # -------------------------

        if self.latest["EMA20"] > self.latest["EMA50"]:

            direction = "🟢 BUY"

            entry = close

            stoploss = self.support

            target1 = entry + (self.resistance - entry) * 0.5

            target2 = self.resistance

            target3 = self.resistance + (
                self.resistance - self.support
            ) * 0.5

        else:

            direction = "🔴 SELL"

            entry = close

            stoploss = self.resistance

            target1 = entry - (entry - self.support) * 0.5

            target2 = self.support

            target3 = self.support - (
                self.resistance - self.support
            ) * 0.5

        # -------------------------
        # Risk Reward
        # -------------------------

        risk = abs(entry - stoploss)

        reward = abs(target2 - entry)

        rr = reward / risk if risk else 0

        # -------------------------
        # Risk Level
        # -------------------------

        if rr >= 2:

            risk_level = "🟢 Low"

        elif rr >= 1:

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