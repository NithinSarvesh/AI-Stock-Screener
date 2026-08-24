"""
Universal PPO V7 Trainer

V7:
- Dynamic Indian stock universe
- 5-position action space
- 31-dimensional observation
- Uses the V7 environment
- Keeps PPO V6 untouched
"""

from __future__ import annotations

import os
import sys
import random

import numpy as np
import pandas as pd
import yfinance as yf

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# =====================================================================
# PROJECT PATH
# =====================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =====================================================================
# IMPORTS
# =====================================================================

from data.universe import get_training_universe
from indicators import IndicatorEngine
from rl.v6_inference import PPOV6Inference
from rl.multi_stock_env_v7 import MultiStockEnvV7

# =====================================================================
# CONFIG
# =====================================================================

SEED = 42

HISTORY_PERIOD = "max"

# First controlled V7 experiment.
TOTAL_TIMESTEPS = 200_000

INITIAL_BALANCE = 100_000.0

TRANSACTION_COST = 0.0005

EPISODE_LENGTH = 252

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "universal_v7",
)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "universal_ppo_v7",
)


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

random.seed(SEED)
np.random.seed(SEED)


# =====================================================================
# DATA PREPARATION
# =====================================================================

def prepare_stock(symbol: str):

    ticker = symbol.upper() + ".NS"

    print(
        f"Downloading: {ticker}"
    )

    try:

        df = yf.download(
            ticker,
            period=HISTORY_PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            print("  REJECT: empty data")
            return None

        # Normalize yfinance MultiIndex.
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

        # -------------------------------------------------------------
        # Same indicator pipeline used by V6
        # -------------------------------------------------------------

        features = (
            IndicatorEngine(
                df.copy()
            ).calculate_all()
        )

        # -------------------------------------------------------------
        # Same context feature pipeline used by V6
        # -------------------------------------------------------------

        features = (
            PPOV6Inference
            .add_context_features(
                features
            )
        )

        if features.empty:

            print(
                "  REJECT: no usable features"
            )

            return None

        print(
            f"  ACCEPT: {len(features)} rows"
        )

        return features

    except Exception as exc:

        print(
            f"  ERROR: {exc}"
        )

        return None


# =====================================================================
# LOAD UNIVERSE
# =====================================================================

def load_universe_data():

    print("=" * 80)
    print("BUILDING V7 TRAINING UNIVERSE")
    print("=" * 80)

    tickers = get_training_universe()

    if not tickers:

        raise RuntimeError(
            "No stocks returned by dynamic universe."
        )

    print()
    print(
        f"Universe candidates: {len(tickers)}"
    )

    data = {}

    for index, symbol in enumerate(
        tickers,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(tickers)}] {symbol}"
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


# =====================================================================
# ENVIRONMENT FACTORY
# =====================================================================

def make_environment(data):

    def factory():

        return MultiStockEnvV7(
            data=data,
            initial_balance=INITIAL_BALANCE,
            transaction_cost=TRANSACTION_COST,
            episode_length=EPISODE_LENGTH,
            random_start=True,
        )

    return factory

# =====================================================================
# TRAIN
# =====================================================================

def train():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("UNIVERSAL PPO V7")
    print("=" * 80)

    print(
        f"Timesteps       : {TOTAL_TIMESTEPS:,}"
    )

    print(
        f"History         : {HISTORY_PERIOD}"
    )

    print(
        f"Episode length  : {EPISODE_LENGTH}"
    )

    print(
        f"Transaction cost: "
        f"{TRANSACTION_COST * 100:.3f}%"
    )

    print()

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    data = load_universe_data()

    # ---------------------------------------------------------------
    # Create environment
    # ---------------------------------------------------------------

    env = DummyVecEnv(
        [
            make_environment(data)
        ]
    )

    print()
    print(
        "Observation space:",
        env.observation_space,
    )

    print(
        "Action space:",
        env.action_space,
    )

    # ---------------------------------------------------------------
    # Verify expected architecture
    # ---------------------------------------------------------------

    expected_shape = (
        30,
    )

    if env.observation_space.shape != expected_shape:

        raise RuntimeError(
            "Unexpected V7 observation shape: "
            f"{env.observation_space.shape}; "
            f"expected {expected_shape}"
        )

    if env.action_space.n != 5:

        raise RuntimeError(
            "Unexpected V7 action count: "
            f"{env.action_space.n}; "
            f"expected 5"
        )

    # ---------------------------------------------------------------
    # PPO
    # ---------------------------------------------------------------

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

        tensorboard_log=os.path.join(
            OUTPUT_DIR,
            "tensorboard",
        ),
    )

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    print()
    print("=" * 80)
    print("STARTING PPO V7 TRAINING")
    print("=" * 80)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    model.save(
        MODEL_PATH
    )

    print()
    print("=" * 80)
    print("V7 TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Model:\n{MODEL_PATH}.zip"
    )

    print(
        f"Stocks used: {len(data)}"
    )

    print(
        "V6 remains untouched."
    )


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    train()