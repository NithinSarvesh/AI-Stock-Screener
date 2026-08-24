"""
Universal PPO V9 Trainer

V9 goals:
- Dynamic Indian stock universe
- Multiple stocks
- Random historical episode starts
- Candle-aware observations
- 33-dimensional observation
- 5-position action space
- Risk-aware reward
- TensorBoard logging
- Checkpointing
"""

from __future__ import annotations

import os
import sys
import random

import numpy as np
import yfinance as yf

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
)

from stable_baselines3.common.monitor import (
    Monitor,
)


# =====================================================================
# PROJECT ROOT
# =====================================================================

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


# =====================================================================
# PROJECT IMPORTS
# =====================================================================

from data.universe import (
    get_training_universe,
)

from indicators import (
    IndicatorEngine,
)

from rl.v6_inference import (
    PPOV6Inference,
)

from rl.trading_env_v9 import (
    OBSERVATION_SIZE,
)

from rl.multi_stock_env_v9 import (
    MultiStockEnvV9,
)


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42

HISTORY_PERIOD = "max"

TOTAL_TIMESTEPS = 100_000

INITIAL_BALANCE = 100_000.0

TRANSACTION_COST = 0.0005

EPISODE_LENGTH = 252

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "universal_v9",
)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "universal_ppo_v9",
)

TENSORBOARD_DIR = os.path.join(
    OUTPUT_DIR,
    "tensorboard",
)

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints",
)


# =====================================================================
# RANDOM SEEDS
# =====================================================================

random.seed(SEED)

np.random.seed(
    SEED
)


# =====================================================================
# DATA PREPARATION
# =====================================================================

def prepare_stock(
    symbol: str,
):

    ticker = (
        symbol.upper()
        + ".NS"
    )

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

        if (
            df is None
            or df.empty
        ):

            print(
                "  REJECT: empty data"
            )

            return None

        # -------------------------------------------------------------
        # Normalize yfinance MultiIndex
        # -------------------------------------------------------------

        if (
            hasattr(
                df.columns,
                "nlevels",
            )
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
                f"  REJECT: only "
                f"{len(df)} rows"
            )

            return None

        # -------------------------------------------------------------
        # Indicator pipeline
        # -------------------------------------------------------------

        features = (
            IndicatorEngine(
                df.copy()
            ).calculate_all()
        )

        # -------------------------------------------------------------
        # Context features
        # -------------------------------------------------------------

        features = (
            PPOV6Inference
            .add_context_features(
                features
            )
        )

        if (
            features is None
            or features.empty
        ):

            print(
                "  REJECT: no features"
            )

            return None

        print(
            f"  ACCEPT: "
            f"{len(features)} rows"
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

def load_universe():

    print()
    print("=" * 80)
    print("BUILDING V9 TRAINING UNIVERSE")
    print("=" * 80)

    symbols = (
        get_training_universe()
    )

    if not symbols:

        raise RuntimeError(
            "Training universe is empty."
        )

    print(
        f"Universe candidates: "
        f"{len(symbols)}"
    )

    data = {}

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(symbols)}] "
            f"{symbol}"
        )

        prepared = prepare_stock(
            symbol
        )

        if prepared is not None:

            data[symbol] = prepared

    if not data:

        raise RuntimeError(
            "No usable stocks."
        )

    print()
    print("=" * 80)
    print(
        f"USABLE STOCKS: "
        f"{len(data)}"
    )
    print("=" * 80)

    return data


# =====================================================================
# MULTI-STOCK ENVIRONMENT
# =====================================================================

def make_environment(data):

    env = MultiStockEnvV9(
        data=data,
        initial_balance=INITIAL_BALANCE,
        transaction_cost=TRANSACTION_COST,
        episode_length=EPISODE_LENGTH,
        random_start=True,
    )

    return Monitor(
        env
    )
# =====================================================================
# MAIN TRAINING
# =====================================================================

def train():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    os.makedirs(
        TENSORBOARD_DIR,
        exist_ok=True,
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("UNIVERSAL PPO V9")
    print("=" * 80)

    print(
        f"Timesteps: "
        f"{TOTAL_TIMESTEPS:,}"
    )

    print(
        f"History: "
        f"{HISTORY_PERIOD}"
    )

    print(
        f"Episode length: "
        f"{EPISODE_LENGTH}"
    )

    print(
        f"Transaction cost: "
        f"{TRANSACTION_COST * 100:.3f}%"
    )

    print(
        f"Expected observations: "
        f"{OBSERVATION_SIZE}"
    )

    print(
        "Actions: 5"
    )

    # -------------------------------------------------------------
    # Load stock data
    # -------------------------------------------------------------

    data = load_universe()

    symbols = list(
        data.keys()
    )

    print()
    print(
        "Training stocks:"
    )

    print(
        ", ".join(symbols)
    )

    # -------------------------------------------------------------
    # Environment factory
    #
    # IMPORTANT:
    # Each reset creates a new stock environment.
    # This prevents V8's fixed-stock problem.
    # -------------------------------------------------------------

    def make_env():

        symbol = random.choice(
            symbols
        )

        return StockTradingEnvV9(
            dataframe=data[symbol],
            initial_balance=INITIAL_BALANCE,
            transaction_cost=TRANSACTION_COST,
            episode_length=EPISODE_LENGTH,
            random_start=True,
        )

    env = DummyVecEnv(
        [
            lambda: make_environment(data)
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

    # -------------------------------------------------------------
    # Architecture validation
    # -------------------------------------------------------------

    if (
        env.observation_space.shape
        != (
            OBSERVATION_SIZE,
        )
    ):

        raise RuntimeError(
            "Observation mismatch: "
            f"{env.observation_space.shape} "
            f"!= "
            f"{(OBSERVATION_SIZE,)}"
        )

    if env.action_space.n != 5:

        raise RuntimeError(
            "Expected 5 actions."
        )

    # -------------------------------------------------------------
    # PPO
    # -------------------------------------------------------------

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

        tensorboard_log=(
            TENSORBOARD_DIR
        ),
    )

    # -------------------------------------------------------------
    # Checkpoint callback
    # -------------------------------------------------------------

    checkpoint_callback = (
        CheckpointCallback(
            save_freq=25_000,
            save_path=CHECKPOINT_DIR,
            name_prefix="universal_v9",
        )
    )

    # -------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("STARTING V9 PILOT")
    print("=" * 80)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # -------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------

    model.save(
        MODEL_PATH
    )

    print()
    print("=" * 80)
    print("V9 TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Model:\n"
        f"{MODEL_PATH}.zip"
    )

    print(
        f"Stocks used: "
        f"{len(data)}"
    )

    print(
        f"Observation size: "
        f"{OBSERVATION_SIZE}"
    )

    print(
        "V8 remains untouched."
    )


if __name__ == "__main__":

    train()