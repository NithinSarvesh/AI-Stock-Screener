"""
Universal PPO V9.2 Trainer

V9.2:
- Dynamic Indian stock universe
- 5-position action space
- 33-dimensional observation
- V9.2 reward environment
- Multi-stock training
"""

from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import yfinance as yf

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# ============================================================================
# PROJECT PATH
# ============================================================================

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


# ============================================================================
# IMPORTS
# ============================================================================

from data.universe import (
    get_training_universe
)

from indicators import (
    IndicatorEngine
)

from rl.v6_inference import (
    PPOV6Inference
)

from rl.multi_stock_env_v9_2 import (
    MultiStockEnvV92
)


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

HISTORY_PERIOD = "max"

TOTAL_TIMESTEPS = 250_000

INITIAL_BALANCE = 100_000.0

TRANSACTION_COST = 0.0005

EPISODE_LENGTH = 252

REQUESTED_STOCKS = 10

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "universal_v9_2",
)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "universal_ppo_v9_2",
)


# ============================================================================
# V9.2 REWARD PARAMETERS
# ============================================================================

DRAWDOWN_PENALTY = 0.05

DOWNSIDE_PENALTY = 0.02

DIRECTIONAL_WEIGHT = 0.003

TURNOVER_PENALTY = 0.005


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

random.seed(
    SEED
)

np.random.seed(
    SEED
)


# ============================================================================
# PREPARE STOCK
# ============================================================================

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

        # ---------------------------------------------------------------
        # Normalize yfinance MultiIndex
        # ---------------------------------------------------------------

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

        # ---------------------------------------------------------------
        # Indicator pipeline
        # ---------------------------------------------------------------

        features = (
            IndicatorEngine(
                df.copy()
            )
            .calculate_all()
        )

        # ---------------------------------------------------------------
        # Context features
        # ---------------------------------------------------------------

        features = (
            PPOV6Inference
            .add_context_features(
                features
            )
        )

        if features.empty:

            print(
                "  REJECT: "
                "no usable features"
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


# ============================================================================
# LOAD UNIVERSE
# ============================================================================

def load_universe_data():

    print()
    print("=" * 80)
    print("BUILDING V9.2 TRAINING UNIVERSE")
    print("=" * 80)

    tickers = (
        get_training_universe()
    )

    if not tickers:

        raise RuntimeError(
            "No stocks returned "
            "by dynamic universe."
        )

    # Limit to requested number.
    tickers = list(
        tickers[
            :REQUESTED_STOCKS
        ]
    )

    print()
    print(
        f"Universe candidates: "
        f"{len(tickers)}"
    )

    data = {}

    for index, symbol in enumerate(
        tickers,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(tickers)}] "
            f"{symbol}"
        )

        prepared = (
            prepare_stock(
                symbol
            )
        )

        if prepared is not None:

            data[symbol] = (
                prepared
            )

    if not data:

        raise RuntimeError(
            "No usable stock datasets."
        )

    print()
    print("=" * 80)
    print(
        f"USABLE STOCKS: "
        f"{len(data)}"
    )
    print("=" * 80)

    return data


# ============================================================================
# ENVIRONMENT FACTORY
# ============================================================================

def make_environment(
    data,
):

    def factory():

        return MultiStockEnvV92(
            data=data,
            initial_balance=(
                INITIAL_BALANCE
            ),
            transaction_cost=(
                TRANSACTION_COST
            ),
            episode_length=(
                EPISODE_LENGTH
            ),
            random_start=True,
            drawdown_penalty=(
                DRAWDOWN_PENALTY
            ),
            downside_penalty=(
                DOWNSIDE_PENALTY
            ),
            directional_weight=(
                DIRECTIONAL_WEIGHT
            ),
            turnover_penalty=(
                TURNOVER_PENALTY
            ),
        )

    return factory


# ============================================================================
# TRAIN
# ============================================================================

def train():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("UNIVERSAL PPO V9.2 TRAINING")
    print("=" * 80)

    print(
        f"Timesteps          : "
        f"{TOTAL_TIMESTEPS:,}"
    )

    print(
        f"History            : "
        f"{HISTORY_PERIOD}"
    )

    print(
        f"Episode length     : "
        f"{EPISODE_LENGTH}"
    )

    print(
        f"Transaction cost   : "
        f"{TRANSACTION_COST * 100:.3f}%"
    )

    print(
        f"Drawdown penalty   : "
        f"{DRAWDOWN_PENALTY}"
    )

    print(
        f"Downside penalty   : "
        f"{DOWNSIDE_PENALTY}"
    )

    print(
        f"Directional weight : "
        f"{DIRECTIONAL_WEIGHT}"
    )

    print(
        f"Turnover penalty   : "
        f"{TURNOVER_PENALTY}"
    )

    print()

    # ========================================================================
    # DATA
    # ========================================================================

    data = (
        load_universe_data()
    )

    # ========================================================================
    # SAVE METADATA
    # ========================================================================

    metadata = {
        "version": "V9.2",
        "seed": SEED,
        "history_period": HISTORY_PERIOD,
        "total_timesteps": (
            TOTAL_TIMESTEPS
        ),
        "initial_balance": (
            INITIAL_BALANCE
        ),
        "transaction_cost": (
            TRANSACTION_COST
        ),
        "episode_length": (
            EPISODE_LENGTH
        ),
        "drawdown_penalty": (
            DRAWDOWN_PENALTY
        ),
        "downside_penalty": (
            DOWNSIDE_PENALTY
        ),
        "directional_weight": (
            DIRECTIONAL_WEIGHT
        ),
        "turnover_penalty": (
            TURNOVER_PENALTY
        ),
        "stocks": list(
            data.keys()
        ),
    }

    metadata_path = os.path.join(
        OUTPUT_DIR,
        "metadata.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"Metadata saved:\n"
        f"{metadata_path}"
    )

    # ========================================================================
    # ENVIRONMENT
    # ========================================================================

    env = DummyVecEnv(
        [
            make_environment(
                data
            )
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

    # ========================================================================
    # ARCHITECTURE CHECKS
    # ========================================================================

    expected_shape = (
        33,
    )

    if (
        env.observation_space.shape
        != expected_shape
    ):

        raise RuntimeError(
            "Unexpected observation "
            f"shape: "
            f"{env.observation_space.shape}; "
            f"expected {expected_shape}"
        )

    if (
        env.action_space.n
        != 5
    ):

        raise RuntimeError(
            "Unexpected action count: "
            f"{env.action_space.n}; "
            f"expected 5"
        )

    # ========================================================================
    # PPO
    # ========================================================================

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

        tensorboard_log=os.path.join(
            OUTPUT_DIR,
            "tensorboard",
        ),
    )

    # ========================================================================
    # TRAIN
    # ========================================================================

    print()
    print("=" * 80)
    print("STARTING V9.2 TRAINING")
    print("=" * 80)

    model.learn(
        total_timesteps=(
            TOTAL_TIMESTEPS
        ),
        progress_bar=True,
    )

    # ========================================================================
    # SAVE
    # ========================================================================

    model.save(
        MODEL_PATH
    )

    print()
    print("=" * 80)
    print("V9.2 TRAINING COMPLETE")
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
        "V9 and V9.1 remain untouched."
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    train()