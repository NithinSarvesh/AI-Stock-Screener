from __future__ import annotations

import numpy as np
import pandas as pd
from indicators import IndicatorEngine


MARKET_FEATURES = [
    "CANDLE_BODY", "CANDLE_RANGE", "UPPER_WICK", "LOWER_WICK",
    "BODY_RANGE_RATIO", "CLOSE_POSITION", "GAP", "BODY_CHANGE",
    "RANGE_CHANGE", "EMA20_DISTANCE", "EMA50_DISTANCE", "EMA200_DISTANCE",
    "RSI_NORMALIZED", "MACD_NORMALIZED", "MACD_SIGNAL_NORMALIZED",
    "MACD_HIST_NORMALIZED", "BB_UPPER_DISTANCE", "BB_MIDDLE_DISTANCE",
    "BB_LOWER_DISTANCE", "VWAP_DISTANCE", "ATR_PCT", "ADX_NORMALIZED",
    "OBV_DIRECTION", "STOCH_RSI_NORMALIZED", "RET_1", "RET_5", "RET_20",
    "VOL_20", "EMA20_SLOPE", "EMA50_SLOPE", "TREND_SCORE",
    "VOLUME_RATIO",
]
OBSERVATION_SIZE = 33


def _safe_series(df, col, default=0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the exact 32 market features + position slot used by the champion."""
    if raw is None or raw.empty:
        raise ValueError("Input dataframe is empty.")

    df = raw.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    df = IndicatorEngine(df).calculate_all()

    close = _safe_series(df, "Close")
    open_ = _safe_series(df, "Open")
    high = _safe_series(df, "High")
    low = _safe_series(df, "Low")
    volume = _safe_series(df, "Volume")

    eps = 1e-8

    candle_range = (high - low).replace(0, np.nan)
    body = close - open_

    df["CANDLE_BODY"] = body / close.replace(0, np.nan)
    df["CANDLE_RANGE"] = candle_range / close.replace(0, np.nan)
    df["UPPER_WICK"] = (high - np.maximum(open_, close)) / close.replace(0, np.nan)
    df["LOWER_WICK"] = (np.minimum(open_, close) - low) / close.replace(0, np.nan)
    df["BODY_RANGE_RATIO"] = body / candle_range
    df["CLOSE_POSITION"] = (close - low) / candle_range
    df["GAP"] = open_ / close.shift(1).replace(0, np.nan) - 1.0
    df["BODY_CHANGE"] = df["CANDLE_BODY"].diff()
    df["RANGE_CHANGE"] = df["CANDLE_RANGE"].diff()

    for ema in ["EMA20", "EMA50", "EMA200"]:
        df[f"{ema}_DISTANCE"] = df[ema] / close.replace(0, np.nan) - 1.0

    df["RSI_NORMALIZED"] = (df["RSI"] - 50.0) / 50.0
    df["MACD_NORMALIZED"] = df["MACD"] / close.replace(0, np.nan)
    df["MACD_SIGNAL_NORMALIZED"] = df["MACD_SIGNAL"] / close.replace(0, np.nan)
    df["MACD_HIST_NORMALIZED"] = df["MACD_HISTOGRAM"] / close.replace(0, np.nan)

    for bb in ["BB_UPPER", "BB_MIDDLE", "BB_LOWER"]:
        df[f"{bb}_DISTANCE"] = df[bb] / close.replace(0, np.nan) - 1.0

    df["VWAP_DISTANCE"] = df["VWAP"] / close.replace(0, np.nan) - 1.0
    df["ATR_PCT"] = df["ATR"] / close.replace(0, np.nan)
    df["ADX_NORMALIZED"] = df["ADX"] / 100.0
    df["OBV_DIRECTION"] = np.sign(df["OBV"].diff()).fillna(0.0)

    # IndicatorEngine exposes STOCH_RSI in [0,1] on the current project.
    df["STOCH_RSI_NORMALIZED"] = df["STOCH_RSI"] * 2.0 - 1.0

    df["RET_1"] = close.pct_change(1)
    df["RET_5"] = close.pct_change(5)
    df["RET_20"] = close.pct_change(20)
    df["VOL_20"] = close.pct_change().rolling(20).std()
    df["EMA20_SLOPE"] = df["EMA20"].pct_change(5)
    df["EMA50_SLOPE"] = df["EMA50"].pct_change(10)

    df["TREND_SCORE"] = (
        (close > df["EMA20"]).astype(float)
        + (df["EMA20"] > df["EMA50"]).astype(float)
        + (df["EMA50"] > df["EMA200"]).astype(float)
    ) / 3.0

    avg_volume = _safe_series(df, "AVG_VOLUME", np.nan)
    df["VOLUME_RATIO"] = volume / avg_volume.replace(0, np.nan)

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=MARKET_FEATURES + ["Close"])
        .copy()
    )

    # Keep all features finite and in a sensible numerical range.
    for col in MARKET_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(-10.0, 10.0)

    df = df.dropna(subset=MARKET_FEATURES + ["Close"]).copy()
    if len(df) < 300:
        raise ValueError(f"Only {len(df)} usable rows after feature engineering.")

    return df
