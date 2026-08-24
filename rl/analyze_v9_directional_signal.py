from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

from stable_baselines3 import PPO

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from indicators import IndicatorEngine
from rl.v6_inference import PPOV6Inference
from rl.trading_env_v9 import StockTradingEnvV9, ACTION_MAP


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "universal_v9",
    "universal_ppo_v9.zip",
)

STOCKS = [
    "RELIANCE",
    "TCS",
    "INFY",
    "SBIN",
    "HDFCBANK",
]


def prepare_stock(symbol):

    ticker = symbol + ".NS"

    print(
        f"Downloading: {ticker}"
    )

    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return None

    if (
        hasattr(df.columns, "nlevels")
        and df.columns.nlevels > 1
    ):
        df.columns = (
            df.columns
            .get_level_values(0)
        )

    df = IndicatorEngine(
        df
    ).calculate_all()

    df = PPOV6Inference.add_context_features(
        df
    )

    return df


def main():

    print("=" * 80)
    print("V9 DIRECTIONAL SIGNAL ANALYSIS")
    print("=" * 80)

    model = PPO.load(
        MODEL_PATH
    )

    records = []

    for symbol in STOCKS:

        print()
        print("=" * 80)
        print(symbol)
        print("=" * 80)

        df = prepare_stock(
            symbol
        )

        if df is None:
            continue

        env = StockTradingEnvV9(
            df,
            random_start=False,
        )

        obs, _ = env.reset(
            seed=42
        )

        while True:

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            action = int(
                action
            )

            current_row = env.df.iloc[
                env.current_step
            ]

            ret5 = float(
                current_row["RET_5"]
            )

            ret20 = float(
                current_row["RET_20"]
            )

            ema20_slope = float(
                current_row["EMA20_SLOPE"]
            )

            ema50_slope = float(
                current_row["EMA50_SLOPE"]
            )

            trend_score = float(
                current_row["TREND_SCORE"]
            )

            directional_signal = np.clip(
                (
                    0.30 * np.tanh(
                        ret5 * 20.0
                    )
                    +
                    0.25 * np.tanh(
                        ret20 * 10.0
                    )
                    +
                    0.20 * np.tanh(
                        ema20_slope * 50.0
                    )
                    +
                    0.15 * np.tanh(
                        ema50_slope * 50.0
                    )
                    +
                    0.10 * (
                        trend_score * 2.0
                        - 1.0
                    )
                ),
                -1.0,
                1.0,
            )

            info = {
                "symbol": symbol,
                "step": env.current_step,
                "action": action,
                "action_name":
                    ACTION_MAP[action]["name"],
                "position":
                    ACTION_MAP[action]["position"],
                "directional_signal":
                    directional_signal,
                "ret5": ret5,
                "ret20": ret20,
                "ema20_slope":
                    ema20_slope,
                "ema50_slope":
                    ema50_slope,
                "trend_score":
                    trend_score,
            }

            records.append(
                info
            )

            obs, reward, terminated, truncated, step_info = env.step(
                action
            )

            if terminated or truncated:
                break

    result = pd.DataFrame(
        records
    )

    print()
    print("=" * 80)
    print("DIRECTIONAL SIGNAL BY ACTION")
    print("=" * 80)

    summary = (
        result
        .groupby("action_name")
        .agg(
            samples=(
                "directional_signal",
                "size",
            ),
            mean_signal=(
                "directional_signal",
                "mean",
            ),
            median_signal=(
                "directional_signal",
                "median",
            ),
            mean_ret5=(
                "ret5",
                "mean",
            ),
            mean_ret20=(
                "ret20",
                "mean",
            ),
            mean_trend_score=(
                "trend_score",
                "mean",
            ),
        )
        .reindex(
            [
                "SHORT",
                "HALF_SHORT",
                "FLAT",
                "HALF_LONG",
                "LONG",
            ]
        )
    )

    print(
        summary.to_string()
    )

    print()
    print("=" * 80)
    print("SIGNAL BUCKET ANALYSIS")
    print("=" * 80)

    result["signal_regime"] = pd.cut(
        result["directional_signal"],
        bins=[
            -1.01,
            -0.50,
            -0.20,
            0.20,
            0.50,
            1.01,
        ],
        labels=[
            "STRONG_BEARISH",
            "BEARISH",
            "NEUTRAL",
            "BULLISH",
            "STRONG_BULLISH",
        ],
    )

    bucket = (
        result
        .groupby(
            [
                "signal_regime",
                "action_name",
            ],
            observed=True,
        )
        .agg(
            samples=(
                "directional_signal",
                "size",
            ),
            avg_signal=(
                "directional_signal",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        bucket.to_string(
            index=False
        )
    )

    output_dir = os.path.join(
        PROJECT_ROOT,
        "models",
        "universal_v9",
        "evaluation",
        "action_analysis",
        "directional_signal",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    result.to_csv(
        os.path.join(
            output_dir,
            "v9_directional_decisions.csv",
        ),
        index=False,
    )

    summary.to_csv(
        os.path.join(
            output_dir,
            "v9_directional_summary.csv",
        )
    )

    bucket.to_csv(
        os.path.join(
            output_dir,
            "v9_directional_buckets.csv",
        ),
        index=False,
    )

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()