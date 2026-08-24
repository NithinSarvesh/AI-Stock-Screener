"""
Universal PPO V8 Trainer

V8:
- Uses the V8 reward environment.
- Uses the same universal Indian stock universe.
- Uses maximum available daily history.
- Keeps PPO V7 untouched.
- Saves the candidate separately as universal_ppo_v8.
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
from rl.trading_env_v8 import StockTradingEnvV8


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42

HISTORY_PERIOD = "max"

# Controlled first V8 experiment.
TOTAL_TIMESTEPS = 200_000

INITIAL_BALANCE = 100_000.0

TRANSACTION_COST = 0.0005

EPISODE_LENGTH = 252

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "universal_v8",
)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "universal_ppo_v8",
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

            print(
                "  REJECT: empty data"
            )

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
        # Same indicator pipeline used by V7.
        # -------------------------------------------------------------

        features = (
            IndicatorEngine(
                df.copy()
            ).calculate_all()
        )

        # -------------------------------------------------------------
        # Same context pipeline.
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
    print("BUILDING V8 TRAINING UNIVERSE")
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

        return StockTradingEnvV8(
            dataframe=random.choice(
                list(data.values())
            ),
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
    print("UNIVERSAL PPO V8")
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
    # Architecture checks
    # ---------------------------------------------------------------

    expected_shape = (
        30,
    )

    if env.observation_space.shape != expected_shape:

        raise RuntimeError(
            "Unexpected V8 observation shape: "
            f"{env.observation_space.shape}; "
            f"expected {expected_shape}"
        )

    if env.action_space.n != 5:

        raise RuntimeError(
            "Unexpected V8 action count: "
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

        ent_coef=0.02,

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
    print("STARTING PPO V8 TRAINING")
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
    print("V8 TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Model:\n{MODEL_PATH}.zip"
    )

    print(
        f"Stocks used: {len(data)}"
    )

    print(
        "V7 remains untouched."
    )


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    train()