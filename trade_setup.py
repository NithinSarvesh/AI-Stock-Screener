class TradeSetup:

    def __init__(self, latest):

        self.latest = latest

    def calculate(self):

        close = self.latest["Close"]
        atr = self.latest["ATR"]

        entry = close

        stoploss = close - (1.5 * atr)

        target1 = close + (1.0 * atr)

        target2 = close + (2.0 * atr)

        target3 = close + (3.0 * atr)

        risk = entry - stoploss

        reward = target2 - entry

        rr = reward / risk if risk else 0

        if rr >= 2:

            risk_level = "🟢 Low"

        elif rr >= 1.5:

            risk_level = "🟡 Medium"

        else:

            risk_level = "🔴 High"

        return {

            "entry": entry,

            "stoploss": stoploss,

            "target1": target1,

            "target2": target2,

            "target3": target3,

            "rr": rr,

            "risk": risk_level

        }