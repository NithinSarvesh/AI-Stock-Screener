"""
Kite/Streak-style technical summary engine.

Produces the same *shape* of output as the "Technicals" tab shown
in Kite's Streak widget:

  - A Moving Averages table (SMA/EMA at 5, 10, 20, 50, 100, 200),
    each classified Buy / Sell / Neutral against the current price.
  - An Oscillators table (RSI, Stochastic, CCI, ADX, Awesome
    Oscillator, ROC, MACD, Stochastic RSI, Williams %R, Ultimate
    Oscillator), each classified Buy / Sell / Neutral.
  - An overall Bearish / Neutral / Bullish vote count plus a single
    gauge value in [-1, +1] for rendering a Kite-style summary meter.

This is an independent, transparent implementation built on the
`ta` library's public indicator formulas. Kite/Streak's own
internal thresholds are proprietary and are not reproduced here —
this engine gets you the same *kind* of signal (a vote count across
a basket of well-known indicators), openly computed.

Expects `history` to already contain Open/High/Low/Close/Volume
with a DatetimeIndex. If it has already been run through
IndicatorEngine.calculate_all() (RSI, MACD, ADX, STOCH_RSI, ATR
columns present), those are reused instead of recomputed.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ta


MA_PERIODS = [5, 10, 20, 50, 100, 200]


def _safe_last(series: pd.Series, default=None):
    if series is None or len(series) == 0:
        return default
    value = series.iloc[-1]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return float(value)


class TechnicalSummaryEngine:

    def __init__(self, history: pd.DataFrame):
        self.df = history.copy()
        self.close = self.df["Close"]
        self.high = self.df["High"]
        self.low = self.df["Low"]
        self.price = float(self.close.iloc[-1])

    # =========================================================
    # MOVING AVERAGES TABLE
    # =========================================================

    def moving_averages(self) -> List[Dict]:

        rows = []

        for period in MA_PERIODS:

            sma_val = _safe_last(self.close.rolling(period).mean())
            ema_val = _safe_last(self.close.ewm(span=period, adjust=False).mean())

            rows.append({
                "name": f"SMA {period}",
                "value": sma_val,
                "signal": self._classify_vs_price(sma_val),
            })

            rows.append({
                "name": f"EMA {period}",
                "value": ema_val,
                "signal": self._classify_vs_price(ema_val),
            })

        return rows

    def _classify_vs_price(self, ma_value) -> str:

        if ma_value is None:
            return "Neutral"

        if self.price > ma_value:
            return "Buy"

        if self.price < ma_value:
            return "Sell"

        return "Neutral"

    def crossover_signals(self) -> Dict[str, str]:
        """
        Kite's "Short Term (5 & 20 SMA CrossOver)" and
        "Long Term (50 & 200 SMA CrossOver)" rows.
        """

        sma5 = _safe_last(self.close.rolling(5).mean())
        sma20 = _safe_last(self.close.rolling(20).mean())
        sma50 = _safe_last(self.close.rolling(50).mean())
        sma200 = _safe_last(self.close.rolling(200).mean())

        def label(fast, slow):
            if fast is None or slow is None:
                return "------"
            if fast > slow:
                return "Bullish"
            if fast < slow:
                return "Bearish"
            return "------"

        return {
            "short_term": label(sma5, sma20),
            "long_term": label(sma50, sma200),
        }

    # =========================================================
    # OSCILLATORS TABLE
    # =========================================================

    def oscillators(self) -> List[Dict]:

        rows = []
        close, high, low = self.close, self.high, self.low

        # ---------------- RSI ----------------
        if "RSI" in self.df.columns:
            rsi_val = _safe_last(self.df["RSI"])
        else:
            rsi_val = _safe_last(ta.momentum.RSIIndicator(close, window=14).rsi())

        rows.append({
            "name": "RSI (14)",
            "value": rsi_val,
            "signal": self._classify_bounds(rsi_val, sell_above=70, buy_below=30),
        })

        # ---------------- Stochastic %K ----------------
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        stoch_val = _safe_last(stoch.stoch())

        rows.append({
            "name": "Stochastic %K",
            "value": stoch_val,
            "signal": self._classify_bounds(stoch_val, sell_above=80, buy_below=20),
        })

        # ---------------- Stochastic RSI ----------------
        if "STOCH_RSI" in self.df.columns:
            stoch_rsi_raw = _safe_last(self.df["STOCH_RSI"])
        else:
            stoch_rsi_raw = _safe_last(ta.momentum.StochRSIIndicator(close).stochrsi())

        stoch_rsi_val = stoch_rsi_raw * 100 if stoch_rsi_raw is not None and stoch_rsi_raw <= 1.5 else stoch_rsi_raw

        rows.append({
            "name": "Stochastic RSI",
            "value": stoch_rsi_val,
            "signal": self._classify_bounds(stoch_rsi_val, sell_above=80, buy_below=20),
        })

        # ---------------- CCI ----------------
        cci_val = _safe_last(ta.trend.CCIIndicator(high, low, close, window=20).cci())

        rows.append({
            "name": "CCI (20)",
            "value": cci_val,
            "signal": self._classify_bounds(cci_val, sell_above=100, buy_below=-100),
        })

        # ---------------- ADX / Directional ----------------
        if "ADX" in self.df.columns:
            adx_val = _safe_last(self.df["ADX"])
            adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
        else:
            adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
            adx_val = _safe_last(adx_ind.adx())

        plus_di = _safe_last(adx_ind.adx_pos())
        minus_di = _safe_last(adx_ind.adx_neg())

        rows.append({
            "name": "ADX (14)",
            "value": adx_val,
            "signal": self._classify_adx(adx_val, plus_di, minus_di),
        })

        # ---------------- Awesome Oscillator ----------------
        ao_series = ta.momentum.AwesomeOscillatorIndicator(high, low).awesome_oscillator()
        ao_val = _safe_last(ao_series)
        ao_prev = _safe_last(ao_series.iloc[:-1]) if len(ao_series) > 1 else None

        rows.append({
            "name": "Awesome Oscillator",
            "value": ao_val,
            "signal": self._classify_ao(ao_val, ao_prev),
        })

        # ---------------- ROC / Momentum ----------------
        roc_val = _safe_last(ta.momentum.ROCIndicator(close, window=12).roc())

        rows.append({
            "name": "ROC (12)",
            "value": roc_val,
            "signal": "Buy" if (roc_val or 0) > 0 else ("Sell" if (roc_val or 0) < 0 else "Neutral"),
        })

        # ---------------- MACD ----------------
        if "MACD" in self.df.columns and "MACD_SIGNAL" in self.df.columns:
            macd_val = _safe_last(self.df["MACD"])
            macd_signal_val = _safe_last(self.df["MACD_SIGNAL"])
        else:
            macd_ind = ta.trend.MACD(close)
            macd_val = _safe_last(macd_ind.macd())
            macd_signal_val = _safe_last(macd_ind.macd_signal())

        macd_signal = "Neutral"
        if macd_val is not None and macd_signal_val is not None:
            macd_signal = "Buy" if macd_val > macd_signal_val else "Sell"

        rows.append({
            "name": "MACD (12,26)",
            "value": macd_val,
            "signal": macd_signal,
        })

        # ---------------- Williams %R ----------------
        wr_val = _safe_last(ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r())

        rows.append({
            "name": "Williams %R",
            "value": wr_val,
            "signal": self._classify_williams(wr_val),
        })

        # ---------------- Ultimate Oscillator ----------------
        uo_val = _safe_last(ta.momentum.UltimateOscillator(high, low, close).ultimate_oscillator())

        rows.append({
            "name": "Ultimate Oscillator",
            "value": uo_val,
            "signal": self._classify_bounds(uo_val, sell_above=70, buy_below=30),
        })

        return rows

    # ---------------------------------------------------------
    # oscillator classification helpers
    # ---------------------------------------------------------

    @staticmethod
    def _classify_bounds(value, sell_above, buy_below) -> str:
        if value is None:
            return "Neutral"
        if value >= sell_above:
            return "Sell"
        if value <= buy_below:
            return "Buy"
        return "Neutral"

    @staticmethod
    def _classify_adx(adx_val, plus_di, minus_di) -> str:
        if adx_val is None or plus_di is None or minus_di is None:
            return "Neutral"
        if adx_val < 20:
            return "Neutral"
        return "Buy" if plus_di > minus_di else "Sell"

    @staticmethod
    def _classify_ao(value, prev) -> str:
        if value is None or prev is None:
            return "Neutral"
        if value > 0 and value >= prev:
            return "Buy"
        if value < 0 and value <= prev:
            return "Sell"
        return "Neutral"

    @staticmethod
    def _classify_williams(value) -> str:
        if value is None:
            return "Neutral"
        if value <= -80:
            return "Buy"
        if value >= -20:
            return "Sell"
        return "Neutral"

    # =========================================================
    # OVERALL SUMMARY / GAUGE
    # =========================================================

    def summary(self) -> Dict:

        ma_rows = self.moving_averages()
        osc_rows = self.oscillators()

        all_rows = ma_rows + osc_rows

        bullish = sum(1 for r in all_rows if r["signal"] == "Buy")
        bearish = sum(1 for r in all_rows if r["signal"] == "Sell")
        neutral = sum(1 for r in all_rows if r["signal"] == "Neutral")
        total = len(all_rows)

        # gauge_value in [-1, +1]: -1 = fully bearish, +1 = fully bullish
        gauge_value = (bullish - bearish) / total if total else 0.0

        if gauge_value >= 0.5:
            verdict = "Strong Bullish"
        elif gauge_value >= 0.15:
            verdict = "Bullish"
        elif gauge_value <= -0.5:
            verdict = "Strong Bearish"
        elif gauge_value <= -0.15:
            verdict = "Bearish"
        else:
            verdict = "Neutral"

        return {
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "total": total,
            "gauge_value": round(gauge_value, 3),
            "verdict": verdict,
            "moving_averages": ma_rows,
            "oscillators": osc_rows,
            "crossovers": self.crossover_signals(),
        }
