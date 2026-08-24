from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import yfinance as yf

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


# ============================================================================
# CONFIG
# ============================================================================

SEED = 42

HISTORY_PERIOD = "max"

TOTAL_TIMESTEPS = 500_000

INITIAL_BALANCE = 100_000.0

TRANSACTION_COST = 0.0005

EPISODE_LENGTH = 252

DRAWDOWN_PENALTY = 0.10

DOWNSIDE_PENALTY = 0.05

OPPORTUNITY_WEIGHT = 0.10

TURNOVER_PENALTY = 0.02

DIRECTIONAL_PENALTY = 0.001


STOCKS = [
    "RELIANCE",
    "TCS",
    "INFY",
    "SBIN",
    "HDFCBANK",
    "ICICIBANK",
    "ITC",
    "LT",
    "BHARTIARTL",
    "AXISBANK",
]


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "universal_v9_1",
)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "universal_ppo_v9_1",
)

TENSORBOARD_DIR = os.path.join(
    OUTPUT_DIR,
    "tensorboard",
)


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

random.seed(SEED)

np.random.seed(SEED)


# ============================================================================
# DATA
# ============================================================================

def prepare_stock(symbol: str):

    ticker = symbol + ".NS"

    print(
        f"Downloading: {ticker}"
    )

    df = yf.download(
        ticker,
        period=HISTORY_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:

        print(
            "  REJECT: empty data"
        )

        return None

    if (
        hasattr(df.columns, "nlevels")
        and df.columns.nlevels > 1
    ):

        df.columns = (
            df.columns
            .get_level_values(0)
        )

    df = df.dropna(
        how="all"
    )

    if len(df) < 900:

        print(
            f"  REJECT: only {len(df)} rows"
        )

        return None

    df = (
        IndicatorEngine(
            df.copy()
        )
        .calculate_all()
    )

    df = (
        PPOV6Inference
        .add_context_features(
            df
        )
    )

    if df.empty:

        print(
            "  REJECT: no usable features"
        )

        return None

    print(
        f"  ACCEPT: {len(df)} rows"
    )

    return df


def load_universe():

    print()
    print("=" * 80)
    print("BUILDING V9.1 TRAINING UNIVERSE")
    print("=" * 80)

    data = {}

    for index, symbol in enumerate(
        STOCKS,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(STOCKS)}] "
            f"{symbol}"
        )

        prepared = prepare_stock(
            symbol
        )

        if prepared is not None:

            data[symbol] = prepared

    if not data:

        raise RuntimeError(
            "No usable stock datasets."
        )

    print()
    print("=" * 80)
    print(
        f"USABLE STOCKS: {len(data)}"
    )
    print("=" * 80)

    return data


# ============================================================================
# ENVIRONMENT
# ============================================================================

def make_env(data):

    def factory():

        return MultiStockEnvV91(
            data=data,
            initial_balance=INITIAL_BALANCE,
            transaction_cost=TRANSACTION_COST,
            episode_length=EPISODE_LENGTH,
            random_start=True,
            drawdown_penalty=DRAWDOWN_PENALTY,
            downside_penalty=DOWNSIDE_PENALTY,
            opportunity_weight=OPPORTUNITY_WEIGHT,
            turnover_penalty=TURNOVER_PENALTY,
            directional_penalty=DIRECTIONAL_PENALTY,
        )

    return factory


# ============================================================================
# METADATA
# ============================================================================

def save_metadata(
    data,
):

    metadata = {

        "version": "V9.1",

        "seed": SEED,

        "history_period":
            HISTORY_PERIOD,

        "total_timesteps":
            TOTAL_TIMESTEPS,

        "episode_length":
            EPISODE_LENGTH,

        "initial_balance":
            INITIAL_BALANCE,

        "transaction_cost":
            TRANSACTION_COST,

        "drawdown_penalty":
            DRAWDOWN_PENALTY,

        "downside_penalty":
            DOWNSIDE_PENALTY,

        "opportunity_weight":
            OPPORTUNITY_WEIGHT,

        "turnover_penalty":
            TURNOVER_PENALTY,

        "directional_penalty":
            DIRECTIONAL_PENALTY,

        "stocks":
            list(data.keys()),

        "observation_size":
            33,

        "action_count":
            5,

        "actions": {

            "0": "SHORT",

            "1": "HALF_SHORT",

            "2": "FLAT",

            "3": "HALF_LONG",

            "4": "LONG",
        },
    }

    metadata_path = os.path.join(
        OUTPUT_DIR,
        "metadata.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"Metadata saved:\n"
        f"{metadata_path}"
    )


# ============================================================================
# TRAIN
# ============================================================================

def train():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    os.makedirs(
        TENSORBOARD_DIR,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("UNIVERSAL PPO V9.1 TRAINING")
    print("=" * 80)

    print(
        f"Timesteps        : "
        f"{TOTAL_TIMESTEPS:,}"
    )

    print(
        f"History          : "
        f"{HISTORY_PERIOD}"
    )

    print(
        f"Episode length   : "
        f"{EPISODE_LENGTH}"
    )

    print(
        f"Stocks requested : "
        f"{len(STOCKS)}"
    )

    print(
        f"Directional penalty: "
        f"{DIRECTIONAL_PENALTY}"
    )

    print(
        f"Opportunity weight: "
        f"{OPPORTUNITY_WEIGHT}"
    )

    data = load_universe()

    save_metadata(
        data
    )

    env = DummyVecEnv(
        [
            make_env(data)
        ]
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

    print()
    print("=" * 80)
    print("CREATING PPO")
    print("=" * 80)

    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=3e-4,

        n_steps=2048,

        batch_size=64,

        n_epochs=10,

        gamma=0.99,

        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.01,

        verbose=1,

        seed=SEED,

        tensorboard_log=TENSORBOARD_DIR,
    )

    print()
    print("=" * 80)
    print("STARTING V9.1 TRAINING")
    print("=" * 80)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
    )

    model.save(
        MODEL_PATH
    )

    print()
    print("=" * 80)
    print("V9.1 TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Model:\n"
        f"{MODEL_PATH}.zip"
    )

    print(
        f"Stocks used: "
        f"{len(data)}"
    )


if __name__ == "__main__":

    train()