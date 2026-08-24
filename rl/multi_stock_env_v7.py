"""
Multi-stock wrapper for PPO V7.

The underlying StockTradingEnvV7 handles one stock at a time.
This wrapper changes the stock between episodes so one PPO policy
can learn across the entire training universe.
"""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np


class MultiStockEnvV7(gym.Env):

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
                "Multi-stock dataset is empty."
            )

        self.data = data

        self.symbols = list(
            data.keys()
        )

        self.initial_balance = float(
            initial_balance
        )

        self.transaction_cost = float(
            transaction_cost
        )

        self.episode_length = int(
            episode_length
        )

        self.random_start = bool(
            random_start
        )

        # Create the first underlying environment.
        self.current_symbol = None
        self.env = None

        self.observation_space = None
        self.action_space = None

        self._create_new_environment()

    # -----------------------------------------------------------------
    # CREATE STOCK ENVIRONMENT
    # -----------------------------------------------------------------

    def _create_new_environment(self):

        self.current_symbol = random.choice(
            self.symbols
        )

        from rl.trading_env_v7 import (
            StockTradingEnvV7
        )

        self.env = StockTradingEnvV7(
            dataframe=self.data[
                self.current_symbol
            ],
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost,
            episode_length=self.episode_length,
            random_start=self.random_start,
        )

        self.observation_space = (
            self.env.observation_space
        )

        self.action_space = (
            self.env.action_space
        )

    # -----------------------------------------------------------------
    # RESET
    # -----------------------------------------------------------------

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed
        )

        # New stock for EVERY episode.
        self._create_new_environment()

        observation, info = (
            self.env.reset(
                seed=seed
            )
        )

        info = dict(info)

        info["symbol"] = (
            self.current_symbol
        )

        return observation, info

    # -----------------------------------------------------------------
    # STEP
    # -----------------------------------------------------------------

    def step(self, action):

        return self.env.step(
            action
        )