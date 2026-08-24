"""
Multi-Stock PPO V9 Environment

Selects a different stock and historical starting point
when a new episode begins.

The underlying StockTradingEnvV9 handles:
- candle features
- technical indicators
- reward
- positions
- risk

This wrapper handles:
- stock selection
- episode-level stock randomization
- consistent observation/action spaces
"""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np


from rl.trading_env_v9 import (
    StockTradingEnvV9,
    OBSERVATION_SIZE,
    ACTION_MAP,
)


class MultiStockEnvV9(gym.Env):

    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        data: dict,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.0005,
        episode_length: int = 252,
        random_start: bool = True,
    ):

        super().__init__()

        if not data:

            raise ValueError(
                "MultiStockEnvV9 received empty data."
            )

        self.data = data

        self.symbols = list(
            data.keys()
        )

        if len(self.symbols) < 2:

            raise ValueError(
                "MultiStockEnvV9 requires "
                "at least 2 stocks."
            )

        self.initial_balance = (
            initial_balance
        )

        self.transaction_cost = (
            transaction_cost
        )

        self.episode_length = (
            episode_length
        )

        self.random_start = (
            random_start
        )

        # -------------------------------------------------------------
        # The wrapper exposes exactly the same spaces as the underlying
        # V9 environment.
        # -------------------------------------------------------------

        self.observation_space = (
            gym.spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(
                    OBSERVATION_SIZE,
                ),
                dtype=np.float32,
            )
        )

        self.action_space = (
            gym.spaces.Discrete(
                len(ACTION_MAP)
            )
        )

        self.current_symbol = None

        self.current_env = None

        self.episode_number = 0

    # =================================================================
    # CREATE EPISODE ENVIRONMENT
    # =================================================================

    def _create_environment(self):

        # -------------------------------------------------------------
        # Choose a stock.
        #
        # Every episode gets a fresh stock.
        # -------------------------------------------------------------

        self.current_symbol = random.choice(
            self.symbols
        )

        dataframe = self.data[
            self.current_symbol
        ]

        self.current_env = (
            StockTradingEnvV9(
                dataframe=dataframe,

                initial_balance=(
                    self.initial_balance
                ),

                transaction_cost=(
                    self.transaction_cost
                ),

                episode_length=(
                    self.episode_length
                ),

                random_start=(
                    self.random_start
                ),
            )
        )

    # =================================================================
    # RESET
    # =================================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed
        )

        self.episode_number += 1

        self._create_environment()

        obs, info = (
            self.current_env.reset(
                seed=seed
            )
        )

        info = dict(
            info
        )

        info.update(
            {
                "symbol":
                    self.current_symbol,

                "episode_number":
                    self.episode_number,
            }
        )

        return (
            obs,
            info,
        )

    # =================================================================
    # STEP
    # =================================================================

    def step(
        self,
        action,
    ):

        if self.current_env is None:

            raise RuntimeError(
                "Environment must be reset "
                "before step()."
            )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.current_env.step(
            action
        )

        info = dict(
            info
        )

        info.update(
            {
                "symbol":
                    self.current_symbol,

                "episode_number":
                    self.episode_number,
            }
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # =================================================================
    # OPTIONAL ACCESSORS
    # =================================================================

    @property
    def current_position(self):

        if self.current_env is None:
            return 0.0

        return self.current_env.position

    @property
    def current_equity(self):

        if self.current_env is None:
            return self.initial_balance

        return self.current_env.equity