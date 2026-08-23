"""
Quick PPO V2 environment test.

Run:

    python rl/test_rl_environment_v2.py
"""

import os

import sys

import numpy as np

import yfinance as yf


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from indicators import IndicatorEngine

from rl.trading_env import StockTradingEnv


TICKER = "RELIANCE"


def main():

    print()
    print("=" * 70)
    print("PPO V2 ENVIRONMENT TEST")
    print("=" * 70)

    print()

    print(
        f"Downloading {TICKER}..."
    )

    data = yf.Ticker(
        TICKER + ".NS"
    ).history(
        period="5y",
        interval="1d",
        auto_adjust=True,
    )

    if data.empty:

        raise ValueError(
            "No market data downloaded."
        )

    data = (
        IndicatorEngine(
            data
        )
        .calculate_all()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    print(
        f"Usable rows: "
        f"{len(data):,}"
    )

    env = StockTradingEnv(

        dataframe=data,

        initial_balance=100000,

        transaction_cost=0.0005,

        random_start=False,
    )

    print()

    print(
        f"Observation space: "
        f"{env.observation_space}"
    )

    print(
        f"Action space: "
        f"{env.action_space}"
    )

    observation, info = (
        env.reset()
    )

    print()

    print(
        f"Observation shape: "
        f"{observation.shape}"
    )

    print(
        f"Observation dtype: "
        f"{observation.dtype}"
    )

    print()

    print(
        "Testing actions..."
    )

    # --------------------------------------------------------
    # FLAT
    # --------------------------------------------------------

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(0)

    print()

    print(
        "Action 0:"
    )

    print(
        f"  Name: "
        f"{info['action_name']}"
    )

    print(
        f"  Position: "
        f"{info['position_name']}"
    )

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(1)

    print()

    print(
        "Action 1:"
    )

    print(
        f"  Name: "
        f"{info['action_name']}"
    )

    print(
        f"  Position: "
        f"{info['position_name']}"
    )

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(2)

    print()

    print(
        "Action 2:"
    )

    print(
        f"  Name: "
        f"{info['action_name']}"
    )

    print(
        f"  Position: "
        f"{info['position_name']}"
    )

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    env.close()

    print()
    print("=" * 70)
    print("PPO V2 ENVIRONMENT TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":

    main()