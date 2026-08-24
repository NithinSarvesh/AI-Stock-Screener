from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "RET_1", "RET_5", "RET_20",
    "RSI_N", "MACD_N", "MACD_HIST_N",
    "EMA20_DIST", "EMA50_DIST", "EMA200_DIST",
    "EMA20_SLOPE", "EMA50_SLOPE",
    "BB_POS", "VWAP_DIST",
    "ATR_PCT", "ADX_N",
    "STOCH_RSI_N", "VOL20",
    "VOLUME_RATIO", "TREND_SCORE",
    "CANDLE_BODY", "CANDLE_RANGE",
    "UPPER_WICK", "LOWER_WICK",
    "BODY_RANGE_RATIO", "CLOSE_POSITION",
    "GAP", "OBV_DIRECTION",
    "REGIME_TREND", "REGIME_VOL",
    "POSITION", "HOLD_DAYS", "COOLDOWN", "SIGNAL_SCORE",
]

OBS_SIZE = len(FEATURES)

def _s(s, fill=0.0):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(fill)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    close = _s(x["Close"])
    high = _s(x["High"])
    low = _s(x["Low"])
    op = _s(x["Open"])
    vol = _s(x["Volume"])

    ret1 = close.pct_change()
    ret5 = close.pct_change(5)
    ret20 = close.pct_change(20)

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    prev_close = close.shift()
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(14).mean()

    stoch = (close - close.rolling(14).min()) / (
        close.rolling(14).max() - close.rolling(14).min()
    ).replace(0, np.nan)
    stoch_rsi = stoch.rolling(3).mean()

    avg_vol = vol.rolling(20).mean()
    vol20 = ret1.rolling(20).std()

    body = close - op
    rng = (high - low).replace(0, np.nan)
    upper_wick = high - pd.concat([op, close], axis=1).max(axis=1)
    lower_wick = pd.concat([op, close], axis=1).min(axis=1) - low
    close_position = (close - low) / rng
    gap = op / prev_close.replace(0, np.nan) - 1

    ema20_slope = ema20.pct_change(5)
    ema50_slope = ema50.pct_change(10)

    trend_score = (
        0.35 * np.tanh((ema20 - ema50) / close * 20)
        + 0.35 * np.tanh((ema50 - ema200) / close * 10)
        + 0.30 * np.tanh(ret20 * 5)
    )

    out = pd.DataFrame(index=x.index)
    out["RET_1"] = ret1
    out["RET_5"] = ret5
    out["RET_20"] = ret20
    out["RSI_N"] = (rsi - 50) / 50
    out["MACD_N"] = macd / close.replace(0, np.nan)
    out["MACD_HIST_N"] = macd_hist / close.replace(0, np.nan)
    out["EMA20_DIST"] = close / ema20.replace(0, np.nan) - 1
    out["EMA50_DIST"] = close / ema50.replace(0, np.nan) - 1
    out["EMA200_DIST"] = close / ema200.replace(0, np.nan) - 1
    out["EMA20_SLOPE"] = ema20_slope
    out["EMA50_SLOPE"] = ema50_slope
    out["BB_POS"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan) * 2 - 1
    out["VWAP_DIST"] = close / (high + low + close).rolling(20).mean().replace(0, np.nan) - 1
    out["ATR_PCT"] = atr / close.replace(0, np.nan)
    out["ADX_N"] = adx / 100
    out["STOCH_RSI_N"] = stoch_rsi * 2 - 1
    out["VOL20"] = vol20
    out["VOLUME_RATIO"] = vol / avg_vol.replace(0, np.nan)
    out["TREND_SCORE"] = trend_score
    out["CANDLE_BODY"] = body / close.replace(0, np.nan)
    out["CANDLE_RANGE"] = rng / close.replace(0, np.nan)
    out["UPPER_WICK"] = upper_wick / close.replace(0, np.nan)
    out["LOWER_WICK"] = lower_wick / close.replace(0, np.nan)
    out["BODY_RANGE_RATIO"] = body / rng
    out["CLOSE_POSITION"] = close_position * 2 - 1
    out["GAP"] = gap
    obv = (np.sign(close.diff()).fillna(0) * vol).cumsum()
    out["OBV_DIRECTION"] = np.tanh(obv.diff(5) / (avg_vol * 5).replace(0, np.nan))
    out["REGIME_TREND"] = np.tanh(trend_score * 2)
    out["REGIME_VOL"] = np.tanh((out["ATR_PCT"] - out["ATR_PCT"].rolling(100).median()) * 50)

    # State features are overwritten by the environment every step.
    out["POSITION"] = 0.0
    out["HOLD_DAYS"] = 0.0
    out["COOLDOWN"] = 0.0
    out["SIGNAL_SCORE"] = (
        0.30 * out["RET_5"].clip(-0.1, 0.1) / 0.1
        + 0.25 * out["RET_20"].clip(-0.2, 0.2) / 0.2
        + 0.25 * out["TREND_SCORE"]
        + 0.20 * out["RSI_N"]
    )

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    for c in FEATURES:
        out[c] = _s(out[c])
    out[FEATURES] = out[FEATURES].clip(-10, 10)
    return out
