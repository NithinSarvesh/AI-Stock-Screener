class CandlePattern:

    def __init__(self, history):

        self.df = history

    def detect(self):

        if len(self.df) < 3:

            return {
                "pattern": "Unknown",
                "confidence": 0,
                "meaning": "Not enough candles."
            }

        last = self.df.iloc[-1]

        prev = self.df.iloc[-2]

        body = abs(last["Close"] - last["Open"])

        upper = last["High"] - max(last["Close"], last["Open"])

        lower = min(last["Close"], last["Open"]) - last["Low"]

        # Hammer

        if lower > body * 2 and upper < body:

            return {

                "pattern": "Hammer",

                "confidence": 82,

                "meaning": "Possible bullish reversal."

            }

        # Doji

        if body < (last["High"] - last["Low"]) * 0.1:

            return {

                "pattern": "Doji",

                "confidence": 70,

                "meaning": "Market indecision."

            }

        # Bullish Engulfing

        if (

            prev["Close"] < prev["Open"]

            and

            last["Close"] > last["Open"]

            and

            last["Close"] > prev["Open"]

            and

            last["Open"] < prev["Close"]

        ):

            return {

                "pattern": "Bullish Engulfing",

                "confidence": 90,

                "meaning": "Strong bullish reversal."

            }

        # Bearish Engulfing

        if (

            prev["Close"] > prev["Open"]

            and

            last["Close"] < last["Open"]

            and

            last["Open"] > prev["Close"]

            and

            last["Close"] < prev["Open"]

        ):

            return {

                "pattern": "Bearish Engulfing",

                "confidence": 90,

                "meaning": "Strong bearish reversal."

            }

        return {

            "pattern": "None",

            "confidence": 50,

            "meaning": "No major candlestick pattern."

        }