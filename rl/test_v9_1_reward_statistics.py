from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from indicators import IndicatorEngine
from rl.v6_inference import PPOV6Inference
from rl.trading_env_v9_1 import (
    StockTradingEnvV91,
    ACTION_MAP,
)


SYMBOL = "RELIANCE"

N_SAMPLES = 100


def prepare():

    df = yf.download(
        SYMBOL + ".NS",
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise RuntimeError(
            "No data downloaded."
        )

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
    print("V9.1 REWARD STATISTICAL TEST")
    print("=" * 80)

    df = prepare()

    records = []

    for seed in range(
        N_SAMPLES
    ):

        env = StockTradingEnvV91(
            df,
            random_start=True,
        )

        obs, _ = env.reset(
            seed=seed
        )

        market_return = None

        for action in range(5):

            test_env = StockTradingEnvV91(
                df,
                random_start=False,
            )

            # Force the same start point selected by
            # the seeded environment.
            test_env.start_step = (
                env.start_step
            )
            test_env.current_step = (
                env.start_step
            )
            test_env.end_step = min(
                env.start_step
                + test_env.episode_length,
                len(test_env.df) - 1,
            )

            test_env.balance = (
                test_env.initial_balance
            )

            test_env.equity = (
                test_env.initial_balance
            )

            test_env.peak_equity = (
                test_env.initial_balance
            )

            test_env.position = 0.0

            (
                _,
                reward,
                _,
                _,
                info,
            ) = test_env.step(
                action
            )

            records.append(
                {
                    "seed": seed,
                    "action": action,
                    "action_name":
                        ACTION_MAP[action]["name"],
                    "market_return":
                        info["market_return"],
                    "strategy_return":
                        info["strategy_return"],
                    "directional_signal":
                        info["directional_signal"],
                    "reward":
                        reward,
                    "alignment":
                        info["directional_alignment"],
                }
            )

    result = pd.DataFrame(
        records
    )

    print()
    print("=" * 80)
    print("AVERAGE REWARD BY ACTION")
    print("=" * 80)

    summary = (
        result
        .groupby(
            "action_name"
        )
        .agg(
            samples=(
                "reward",
                "size",
            ),
            avg_reward=(
                "reward",
                "mean",
            ),
            median_reward=(
                "reward",
                "median",
            ),
            avg_market_return=(
                "market_return",
                "mean",
            ),
            avg_strategy_return=(
                "strategy_return",
                "mean",
            ),
            avg_signal=(
                "directional_signal",
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
    print("PROFITABLE vs UNPROFITABLE ACTIONS")
    print("=" * 80)

    result["profitable"] = (
        result["strategy_return"]
        > 0
    )

    profit_stats = (
        result
        .groupby(
            [
                "action_name",
                "profitable",
            ]
        )
        .agg(
            samples=(
                "reward",
                "size",
            ),
            avg_reward=(
                "reward",
                "mean",
            ),
            avg_strategy_return=(
                "strategy_return",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        profit_stats.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("ACTION REWARD ORDER")
    print("=" * 80)

    ordered = (
        summary["avg_reward"]
        .sort_values(
            ascending=False
        )
    )

    for name, value in ordered.items():

        print(
            f"{name:<12} "
            f"{value:+.6f}"
        )

    output_dir = os.path.join(
        PROJECT_ROOT,
        "models",
        "universal_v9",
        "evaluation",
        "v9_1_reward_analysis",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    result.to_csv(
        os.path.join(
            output_dir,
            "v9_1_reward_samples.csv",
        ),
        index=False,
    )

    summary.to_csv(
        os.path.join(
            output_dir,
            "v9_1_reward_summary.csv",
        )
    )

    print()
    print("=" * 80)
    print("V9.1 REWARD TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()