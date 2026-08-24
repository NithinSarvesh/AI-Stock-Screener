import os
import sys

# -------------------------------------------------------------
# Add project root to Python path
# -------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import yfinance as yf

from indicators import IndicatorEngine
from rl.v6_inference import PPOV6Inference

from rl.trading_env_v9 import (
    StockTradingEnvV9,
    ACTION_MAP,
)


def load_data():

    df = yf.download(
        "RELIANCE.NS",
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
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


def test_actions(df):

    print()
    print("=" * 75)
    print("V9 ACTION SANITY TEST")
    print("=" * 75)

    for action in range(5):

        # IMPORTANT:
        # Create a completely fresh environment for every action.
        env = StockTradingEnvV9(
            df,
            random_start=False,
        )

        obs, info = env.reset(
            seed=42
        )

        (
            next_obs,
            reward,
            terminated,
            truncated,
            step_info,
        ) = env.step(
            action
        )

        print(
            f"{action} | "
            f"{ACTION_MAP[action]['name']:12s} | "
            f"position="
            f"{ACTION_MAP[action]['position']:>5.1f} | "
            f"market_return="
            f"{step_info['market_return']:+.8f} | "
            f"strategy_return="
            f"{step_info['strategy_return']:+.8f} | "
            f"reward="
            f"{reward:+.8f}"
        )


def main():

    print("=" * 75)
    print("V9 ENVIRONMENT TEST")
    print("=" * 75)

    df = load_data()

    print(
        "Prepared rows:",
        len(df),
    )

    env = StockTradingEnvV9(
        df,
        random_start=False,
    )

    obs, info = env.reset(
        seed=42
    )

    print(
        "Observation shape:",
        obs.shape,
    )

    print(
        "Observation space:",
        env.observation_space,
    )

    print(
        "Action space:",
        env.action_space,
    )

    print(
        "Expected observation:",
        obs.shape[0],
    )

    print(
        "Expected action count:",
        env.action_space.n,
    )

    test_actions(df)

    print()
    print("=" * 75)
    print("V9 TEST COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()