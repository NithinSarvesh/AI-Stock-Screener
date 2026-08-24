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
from rl.trading_env_v9_1 import (
    StockTradingEnvV91,
    ACTION_MAP,
)


print("=" * 75)
print("V9.1 ENVIRONMENT TEST")
print("=" * 75)

df = yf.download(
    "RELIANCE.NS",
    period="5y",
    interval="1d",
    auto_adjust=False,
    progress=False,
    threads=False,
)

if df is None or df.empty:
    raise RuntimeError(
        "Yahoo returned empty data."
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

env = StockTradingEnvV91(
    df,
    random_start=False,
)

print(
    "Prepared rows:",
    len(env.df),
)

print(
    "Observation shape:",
    env.observation_space.shape,
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
    33,
)

print(
    "Expected action count:",
    5,
)

if env.observation_space.shape != (33,):
    raise RuntimeError(
        "Wrong observation shape."
    )

if env.action_space.n != 5:
    raise RuntimeError(
        "Wrong action count."
    )


print()
print("=" * 75)
print("V9.1 ACTION SANITY TEST")
print("=" * 75)

for action in range(5):

    test_env = StockTradingEnvV91(
        df,
        random_start=False,
    )

    obs, _ = test_env.reset(
        seed=42
    )

    (
        obs,
        reward,
        terminated,
        truncated,
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
        f"directional_signal="
        f"{info['directional_signal']:+.6f} | "
        f"alignment="
        f"{info['directional_alignment']:+.6f} | "
        f"reward="
        f"{reward:+.8f}"
    )


print()
print("=" * 75)
print("V9.1 TEST COMPLETE")
print("=" * 75)