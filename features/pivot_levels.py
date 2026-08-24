"""
Classic floor-trader pivot points.

Computes Support (S1-S3) and Resistance (R1-R3) levels from the
previous completed candle's High, Low and Close — the same inputs
Kite's Streak "Technicals" widget uses for its S1/S2/S3/R1/R2/R3
row.

This is the *classical* pivot formula (the same one used by
TradingView, Investing.com and most charting platforms). Kite's
own internal computation is proprietary and not publicly
documented, so this is a transparent equivalent rather than a
byte-for-byte replication — in practice it lands very close to
what Kite shows for the same timeframe.

Usage:
    levels = PivotCalculator(history).calculate()
    levels["pivot"], levels["s1"], levels["r1"], ...
"""

import pandas as pd


class PivotCalculator:

    def __init__(self, history: pd.DataFrame):
        self.history = history

    def calculate(self, lookback: int = 1) -> dict:
        """
        lookback=1 (default) uses the most recently *completed*
        candle to compute the current levels — e.g. yesterday's
        daily OHLC for today's daily pivots. This mirrors how
        Kite's daily-timeframe pivots are computed.

        Raises ValueError if there isn't enough history yet.
        """

        if len(self.history) < lookback + 1:
            raise ValueError(
                "Not enough history to compute pivot levels."
            )

        prior = self.history.iloc[-(lookback + 1)]

        high = float(prior["High"])
        low = float(prior["Low"])
        close = float(prior["Close"])

        pivot = (high + low + close) / 3

        r1 = 2 * pivot - low
        s1 = 2 * pivot - high

        r2 = pivot + (high - low)
        s2 = pivot - (high - low)

        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)

        return {
            "pivot": pivot,
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "basis_date": self.history.index[-(lookback + 1)],
        }

    def locate_price(self, price: float, levels: dict) -> str:
        """
        Returns which zone the current price sits in, e.g.
        "Between Pivot and R1". Used to draw the marker position
        on the Kite-style support/resistance bar.
        """

        ordered = [
            ("Below S3", levels["s3"]),
            ("S3", levels["s3"]),
            ("S2", levels["s2"]),
            ("S1", levels["s1"]),
            ("Pivot", levels["pivot"]),
            ("R1", levels["r1"]),
            ("R2", levels["r2"]),
            ("R3", levels["r3"]),
            ("Above R3", levels["r3"]),
        ]

        if price < levels["s3"]:
            return "Below S3"
        if price > levels["r3"]:
            return "Above R3"

        boundaries = [
            levels["s3"], levels["s2"], levels["s1"],
            levels["pivot"], levels["r1"], levels["r2"], levels["r3"],
        ]
        labels = ["S3", "S2", "S1", "Pivot", "R1", "R2", "R3"]

        for i in range(len(boundaries) - 1):
            if boundaries[i] <= price <= boundaries[i + 1]:
                return f"Between {labels[i]} and {labels[i + 1]}"

        return "Pivot"
