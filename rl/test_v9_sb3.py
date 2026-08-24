import os
import sys
import random

import numpy as np
import pandas as pd
import yfinance as yf

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from indicators import IndicatorEngine
from rl.v6_inference import PPOV6Inference
from rl.multi_stock_env_v9 import MultiStockEnvV9


def prepare(symbol):

    print(f"Downloading {symbol}.NS")

    df = yf.download(
        symbol + ".NS",
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    df = IndicatorEngine(df).calculate_all()
    df = PPOV6Inference.add_context_features(df)

    return df


def main():

    print("=" * 70)
    print("V9 SB3 COMPATIBILITY TEST")
    print("=" * 70)

    symbols = [
        "RELIANCE",
        "TCS",
        "INFY",
        "SBIN",
        "HDFCBANK",
    ]

    data = {}

    for symbol in symbols:

        df = prepare(symbol)

        if df is not None and not df.empty:
            data[symbol] = df

    print()
    print("Stocks:", list(data.keys()))

    def make_env():

        env = MultiStockEnvV9(
            data=data,
            initial_balance=100000.0,
            transaction_cost=0.0005,
            episode_length=252,
            random_start=True,
        )

        return Monitor(env)

    env = DummyVecEnv([make_env])

    print()
    print("Observation:", env.observation_space)
    print("Actions:", env.action_space)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=64,
        n_epochs=2,
        learning_rate=3e-4,
        verbose=1,
        seed=42,
    )

    print()
    print("=" * 70)
    print("RUNNING 2,000 PPO TIMESTEPS")
    print("=" * 70)

    model.learn(
        total_timesteps=2000,
        progress_bar=True,
    )

    print()
    print("=" * 70)
    print("SB3 V9 TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()