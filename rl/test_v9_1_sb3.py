from __future__ import annotations

import os
import sys

from stable_baselines3 import PPO
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
from rl.multi_stock_env_v9_1 import MultiStockEnvV91

import yfinance as yf


STOCKS = [
    "RELIANCE",
    "TCS",
    "INFY",
    "SBIN",
    "HDFCBANK",
]


def prepare(symbol):

    print(
        f"Downloading {symbol}.NS"
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

    print("=" * 70)
    print("V9.1 SB3 COMPATIBILITY TEST")
    print("=" * 70)

    data = {}

    for symbol in STOCKS:
        data[symbol] = prepare(
            symbol
        )

    print()
    print(
        "Stocks:",
        list(data.keys())
    )

    def make_env():

        return MultiStockEnvV91(
            data=data,
            initial_balance=100000.0,
            transaction_cost=0.0005,
            episode_length=252,
            random_start=True,
            drawdown_penalty=0.1,
            downside_penalty=0.05,
            opportunity_weight=0.10,
            turnover_penalty=0.02,
            directional_penalty=0.001,
        )

    env = DummyVecEnv(
        [make_env]
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

    if env.observation_space.shape != (
        33,
    ):
        raise RuntimeError(
            "Unexpected observation shape."
        )

    if env.action_space.n != 5:
        raise RuntimeError(
            "Unexpected action count."
        )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
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
    print("SB3 V9.1 TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()