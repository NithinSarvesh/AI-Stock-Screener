from __future__ import annotations

import os
import sys

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
from rl.multi_stock_env_v9_1 import (
    MultiStockEnvV91,
)


STOCKS = [
    "RELIANCE",
    "TCS",
    "INFY",
    "SBIN",
    "HDFCBANK",
]


def prepare(symbol):

    print(
        f"Downloading: {symbol}.NS"
    )

    df = yf.download(
        symbol + ".NS",
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise RuntimeError(
            f"No data for {symbol}"
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

    print("=" * 75)
    print("MULTI-STOCK V9.1 TEST")
    print("=" * 75)

    data = {}

    for symbol in STOCKS:

        data[symbol] = prepare(
            symbol
        )

    print()
    print(
        "Stocks loaded:",
        list(data.keys())
    )

    env = MultiStockEnvV91(
        data=data,
        episode_length=252,
        random_start=True,
    )

    print()
    print(
        "Observation:",
        env.observation_space
    )

    print(
        "Actions:",
        env.action_space
    )

    observed = []

    print()
    print("=" * 75)
    print("EPISODE STOCK TEST")
    print("=" * 75)

    for episode in range(10):

        obs, info = env.reset(
            seed=episode
        )

        symbol = info[
            "symbol"
        ]

        observed.append(
            symbol
        )

        print(
            f"Episode {episode + 1:02d}: "
            f"{symbol:<10} "
            f"obs={obs.shape}"
        )

        for _ in range(5):

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(2)

            if terminated or truncated:
                break

    print()
    print(
        "Stocks observed:",
        observed
    )

    print(
        "Unique stocks:",
        len(set(observed))
    )

    if env.observation_space.shape != (
        33,
    ):
        raise RuntimeError(
            "Wrong observation shape."
        )

    if env.action_space.n != 5:
        raise RuntimeError(
            "Wrong action count."
        )

    print()
    print("=" * 75)
    print("MULTI-STOCK V9.1 TEST PASSED")
    print("=" * 75)


if __name__ == "__main__":
    main()