from __future__ import annotations

import random

import gymnasium as gym
import numpy as np


class MultiStockEnvV91(gym.Env):

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
        drawdown_penalty: float = 0.1,
        downside_penalty: float = 0.05,
        opportunity_weight: float = 0.10,
        turnover_penalty: float = 0.02,
        directional_penalty: float = 0.001,
    ):

        super().__init__()

        if not data:
            raise ValueError(
                "No stock data supplied."
            )

        self.data = data

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

        self.drawdown_penalty = (
            drawdown_penalty
        )

        self.downside_penalty = (
            downside_penalty
        )

        self.opportunity_weight = (
            opportunity_weight
        )

        self.turnover_penalty = (
            turnover_penalty
        )

        self.directional_penalty = (
            directional_penalty
        )

        self.symbols = list(
            self.data.keys()
        )

        # We create one underlying V9.1 environment
        # at reset time for the selected stock.
        self.current_env = None
        self.current_symbol = None

        # Import here to keep the module lightweight.
        from rl.trading_env_v9_1 import (
            StockTradingEnvV91
        )

        self.env_class = (
            StockTradingEnvV91
        )

        # Spaces must match StockTradingEnvV9.1.
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(33,),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Discrete(
            5
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed
        )

        if self.random_start:

            index = int(
                self.np_random.integers(
                    0,
                    len(self.symbols),
                )
            )

            symbol = self.symbols[
                index
            ]

        else:

            symbol = self.symbols[0]

        self.current_symbol = symbol

        self.current_env = (
            self.env_class(
                self.data[symbol],
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
                drawdown_penalty=(
                    self.drawdown_penalty
                ),
                downside_penalty=(
                    self.downside_penalty
                ),
                opportunity_weight=(
                    self.opportunity_weight
                ),
                turnover_penalty=(
                    self.turnover_penalty
                ),
                directional_penalty=(
                    self.directional_penalty
                ),
            )
        )

        observation, info = (
            self.current_env.reset(
                seed=seed
            )
        )

        info["symbol"] = (
            self.current_symbol
        )

        return observation, info

    def step(
        self,
        action,
    ):

        if self.current_env is None:
            raise RuntimeError(
                "Environment must be reset before step."
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

        info["symbol"] = (
            self.current_symbol
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )