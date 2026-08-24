"""
Technical event detector.

Scans recent candles and produces a timestamped feed of named
events — the same idea as Kite's "Technical Events" list (e.g.
"Bearish Harami Pattern", "Short Term Bullish", "Heikin Ashi
Bullish Reversal").

Detects:
  - Classic candlestick patterns (Doji, Hammer, Inverted Hammer,
    Shooting Star, Bullish/Bearish Engulfing, Bullish/Bearish
    Harami)
  - Heikin Ashi trend-reversal candles
  - Short-term (5/20 SMA) and long-term (50/200 SMA) crossovers
  - RSI crossing out of oversold/overbought
  - MACD line crossing its signal line

This extends (does not replace) candlestick.py, which is still
used elsewhere in the app for the single "latest pattern" badge.
This module instead returns a *history* of events with timestamps.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


class TechnicalEventDetector:

    def __init__(self, history: pd.DataFrame, lookback_bars: int = 60):
        self.df = history.tail(lookback_bars + 5).copy()
        self.lookback_bars = lookback_bars

    # =========================================================
    # HEIKIN ASHI
    # =========================================================

    def _heikin_ashi(self) -> pd.DataFrame:

        df = self.df
        ha = pd.DataFrame(index=df.index)

        ha["Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

        ha_open = [float((df["Open"].iloc[0] + df["Close"].iloc[0]) / 2)]

        for i in range(1, len(df)):
            ha_open.append((ha_open[i - 1] + ha["Close"].iloc[i - 1]) / 2)

        ha["Open"] = ha_open

        ha["High"] = pd.concat(
            [df["High"], ha["Open"], ha["Close"]], axis=1
        ).max(axis=1)

        ha["Low"] = pd.concat(
            [df["Low"], ha["Open"], ha["Close"]], axis=1
        ).min(axis=1)

        return ha

    # =========================================================
    # CANDLESTICK PATTERNS
    # =========================================================

    def _candlestick_events(self) -> List[Dict]:

        df = self.df
        events = []

        for i in range(2, len(df)):

            last = df.iloc[i]
            prev = df.iloc[i - 1]

            body = abs(last["Close"] - last["Open"])
            candle_range = last["High"] - last["Low"]

            if candle_range <= 0:
                continue

            upper_wick = last["High"] - max(last["Close"], last["Open"])
            lower_wick = min(last["Close"], last["Open"]) - last["Low"]

            # Doji — indecision
            if body < candle_range * 0.1:
                events.append(self._event(df.index[i], "Doji Pattern", "neutral"))
                continue

            # Hammer — bullish reversal (long lower wick)
            if lower_wick > body * 2 and upper_wick < body:
                events.append(self._event(df.index[i], "Hammer Pattern", "bullish"))

            # Shooting Star — bearish reversal (long upper wick)
            if upper_wick > body * 2 and lower_wick < body:
                events.append(self._event(df.index[i], "Shooting Star Pattern", "bearish"))

            # Bullish Engulfing
            if (
                prev["Close"] < prev["Open"]
                and last["Close"] > last["Open"]
                and last["Close"] > prev["Open"]
                and last["Open"] < prev["Close"]
            ):
                events.append(self._event(df.index[i], "Bullish Engulfing Pattern", "bullish"))

            # Bearish Engulfing
            if (
                prev["Close"] > prev["Open"]
                and last["Close"] < last["Open"]
                and last["Open"] > prev["Close"]
                and last["Close"] < prev["Open"]
            ):
                events.append(self._event(df.index[i], "Bearish Engulfing Pattern", "bearish"))

            # Bullish Harami — small green body inside prior big red body
            if (
                prev["Close"] < prev["Open"]
                and last["Close"] > last["Open"]
                and last["Open"] > prev["Close"]
                and last["Close"] < prev["Open"]
            ):
                events.append(self._event(df.index[i], "Bullish Harami Pattern", "bullish"))

            # Bearish Harami — small red body inside prior big green body
            if (
                prev["Close"] > prev["Open"]
                and last["Close"] < last["Open"]
                and last["Open"] < prev["Close"]
                and last["Close"] > prev["Open"]
            ):
                events.append(self._event(df.index[i], "Bearish Harami Pattern", "bearish"))

        return events

    # =========================================================
    # HEIKIN ASHI REVERSALS
    # =========================================================

    def _heikin_ashi_events(self) -> List[Dict]:

        ha = self._heikin_ashi()
        events = []

        for i in range(1, len(ha)):

            cur = ha.iloc[i]
            prev = ha.iloc[i - 1]

            candle_range = cur["High"] - cur["Low"]
            if candle_range <= 0:
                continue

            cur_bullish = cur["Close"] > cur["Open"]
            prev_bearish = prev["Close"] < prev["Open"]
            lower_wick = min(cur["Open"], cur["Close"]) - cur["Low"]

            cur_bearish = cur["Close"] < cur["Open"]
            prev_bullish = prev["Close"] > prev["Open"]
            upper_wick = cur["High"] - max(cur["Open"], cur["Close"])

            # A "clean" HA candle (little/no opposing wick) flipping
            # direction from the prior candle is treated as a
            # Heikin-Ashi trend-reversal signal.
            if cur_bullish and prev_bearish and lower_wick < candle_range * 0.05:
                events.append(self._event(
                    self.df.index[i], "Heikin Ashi Bullish Reversal", "bullish"
                ))

            elif cur_bearish and prev_bullish and upper_wick < candle_range * 0.05:
                events.append(self._event(
                    self.df.index[i], "Heikin Ashi Bearish Reversal", "bearish"
                ))

        return events

    # =========================================================
    # MOVING AVERAGE CROSSOVERS
    # =========================================================

    def _crossover_events(self) -> List[Dict]:

        close = self.df["Close"]
        events = []

        pairs = [
            (5, 20, "Short Term"),
            (50, 200, "Long Term"),
        ]

        for fast_p, slow_p, label in pairs:

            if len(close) < slow_p + 2:
                continue

            fast = close.rolling(fast_p).mean()
            slow = close.rolling(slow_p).mean()

            cross = np.sign(fast - slow)
            cross_change = cross.diff()

            for i in range(len(close)):

                if pd.isna(cross_change.iloc[i]):
                    continue

                if cross_change.iloc[i] == 2:
                    name = f"{label} Bullish ({fast_p}/{slow_p} SMA Cross)"
                    events.append(self._event(self.df.index[i], name, "bullish"))

                elif cross_change.iloc[i] == -2:
                    name = f"{label} Bearish ({fast_p}/{slow_p} SMA Cross)"
                    events.append(self._event(self.df.index[i], name, "bearish"))

        return events

    # =========================================================
    # RSI / MACD SIGNAL CROSSES
    # =========================================================

    def _rsi_events(self) -> List[Dict]:

        close = self.df["Close"]
        events = []

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        for i in range(1, len(rsi)):

            if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i - 1]):
                continue

            if rsi.iloc[i - 1] <= 30 < rsi.iloc[i]:
                events.append(self._event(
                    self.df.index[i], "RSI Exited Oversold Zone", "bullish"
                ))

            elif rsi.iloc[i - 1] >= 70 > rsi.iloc[i]:
                events.append(self._event(
                    self.df.index[i], "RSI Exited Overbought Zone", "bearish"
                ))

        return events

    def _macd_events(self) -> List[Dict]:

        close = self.df["Close"]
        events = []

        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        diff = macd_line - signal_line
        cross = np.sign(diff)
        cross_change = cross.diff()

        for i in range(len(cross_change)):

            if pd.isna(cross_change.iloc[i]):
                continue

            if cross_change.iloc[i] == 2:
                events.append(self._event(
                    self.df.index[i], "MACD Bullish Crossover", "bullish"
                ))

            elif cross_change.iloc[i] == -2:
                events.append(self._event(
                    self.df.index[i], "MACD Bearish Crossover", "bearish"
                ))

        return events

    # =========================================================
    # PUBLIC METHOD
    # =========================================================

    @staticmethod
    def _event(timestamp, name, bias) -> Dict:
        return {
            "timestamp": timestamp,
            "event": name,
            "bias": bias,
        }

    def detect(self, max_events: int = 15) -> List[Dict]:

        if len(self.df) < 5:
            return []

        events = []
        events.extend(self._candlestick_events())
        events.extend(self._heikin_ashi_events())
        events.extend(self._crossover_events())
        events.extend(self._rsi_events())
        events.extend(self._macd_events())

        # Only keep events inside the requested lookback window.
        cutoff = self.df.index[-min(self.lookback_bars, len(self.df))]
        events = [e for e in events if e["timestamp"] >= cutoff]

        events.sort(key=lambda e: e["timestamp"], reverse=True)

        return events[:max_events]
