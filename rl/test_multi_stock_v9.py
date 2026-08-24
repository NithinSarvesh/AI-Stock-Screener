import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


import yfinance as yf

from indicators import IndicatorEngine
from rl.v6_inference import PPOV6Inference

from rl.multi_stock_env_v9 import (
    MultiStockEnvV9,
)


def prepare(symbol):

    ticker = (
        symbol
        + ".NS"
    )

    print(
        "Downloading:",
        ticker,
    )

    df = yf.download(
        ticker,
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

    df = (
        IndicatorEngine(
            df
        ).calculate_all()
    )

    df = (
        PPOV6Inference
        .add_context_features(
            df
        )
    )

    return df


def main():

    print("=" * 75)
    print("MULTI-STOCK V9 TEST")
    print("=" * 75)

    symbols = [
        "RELIANCE",
        "TCS",
        "INFY",
        "SBIN",
        "HDFCBANK",
    ]

    data = {}

    for symbol in symbols:

        df = prepare(
            symbol
        )

        if (
            df is not None
            and not df.empty
        ):

            data[
                symbol
            ] = df

    print()
    print(
        "Stocks loaded:",
        list(data.keys()),
    )

    env = MultiStockEnvV9(
        data=data,
        episode_length=20,
        random_start=True,
    )

    print()
    print(
        "Observation:",
        env.observation_space,
    )

    print(
        "Actions:",
        env.action_space,
    )

    print()
    print("=" * 75)
    print("EPISODE STOCK TEST")
    print("=" * 75)

    seen = []

    for episode in range(10):

        obs, info = env.reset(
            seed=episode
        )

        symbol = info[
            "symbol"
        ]

        seen.append(
            symbol
        )

        print(
            f"Episode "
            f"{episode + 1:02d}: "
            f"{symbol:10s} "
            f"obs={obs.shape}"
        )

        # Take one FLAT action.
        (
            obs,
            reward,
            terminated,
            truncated,
            step_info,
        ) = env.step(2)

        assert (
            step_info["symbol"]
            == symbol
        )

    print()
    print(
        "Stocks observed:",
        seen,
    )

    print()
    print(
        "Unique stocks:",
        len(set(seen)),
    )

    if len(set(seen)) < 2:

        raise RuntimeError(
            "Stock randomization failed."
        )

    print()
    print("=" * 75)
    print("MULTI-STOCK V9 TEST PASSED")
    print("=" * 75)


if __name__ == "__main__":
    main()