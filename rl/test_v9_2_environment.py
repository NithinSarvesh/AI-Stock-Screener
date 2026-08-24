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
from rl.trading_env_v9_2 import (
    StockTradingEnvV92,
    ACTION_MAP,
)


print("=" * 75)
print("V9.2 ENVIRONMENT TEST")
print("=" * 75)

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

df = (
    IndicatorEngine(
        df
    )
    .calculate_all()
)

df = (
    PPOV6Inference
    .add_context_features(
        df
    )
)

env = StockTradingEnvV92(
    df,
    random_start=False,
)

obs, _ = env.reset(
    seed=42
)

print(
    "Prepared rows:",
    len(env.df)
)

print(
    "Observation:",
    obs.shape
)

print(
    "Observation space:",
    env.observation_space
)

print(
    "Action space:",
    env.action_space
)

print()
print("=" * 75)
print("V9.2 ACTION SANITY TEST")
print("=" * 75)

for action in range(5):

    test_env = StockTradingEnvV92(
        df,
        random_start=False,
    )

    test_env.reset(
        seed=42
    )

    (
        _,
        reward,
        _,
        _,
        info,
    ) = test_env.step(
        action
    )

    print(
        f"{action} | "
        f"{ACTION_MAP[action]['name']:<12} | "
        f"position="
        f"{ACTION_MAP[action]['position']:>5.1f} | "
        f"market_return="
        f"{info['market_return']:+.8f} | "
        f"strategy_return="
        f"{info['strategy_return']:+.8f} | "
        f"signal="
        f"{info['directional_signal']:+.6f} | "
        f"alignment="
        f"{info['directional_alignment']:+.6f} | "
        f"reward="
        f"{reward:+.8f}"
    )

print()
print("=" * 75)
print("V9.2 TEST COMPLETE")
print("=" * 75)