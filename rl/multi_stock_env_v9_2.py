from __future__ import annotations

import gymnasium as gym
import numpy as np


class MultiStockEnvV92(gym.Env):

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
        drawdown_penalty: float = 0.05,
        downside_penalty: float = 0.02,
        directional_weight: float = 0.003,
        turnover_penalty: float = 0.005,
    ):

        super().__init__()

        if not data:
            raise ValueError(
                "No stock data supplied."
            )

        self.data = data

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

        self.drawdown_penalty = float(
            drawdown_penalty
        )

        self.downside_penalty = float(
            downside_penalty
        )

        self.directional_weight = float(
            directional_weight
        )

        self.turnover_penalty = float(
            turnover_penalty
        )

        self.symbols = list(
            self.data.keys()
        )

        if not self.symbols:
            raise ValueError(
                "No symbols available."
            )

        self.current_env = None
        self.current_symbol = None

        # Import here so the wrapper remains lightweight.
        from rl.trading_env_v9_2 import (
            StockTradingEnvV92
        )

        self.env_class = StockTradingEnvV92

        # Must match StockTradingEnvV92.
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

            symbol = self.symbols[index]

        else:

            symbol = self.symbols[0]

        self.current_symbol = symbol

        self.current_env = self.env_class(
            self.data[symbol],

            initial_balance=self.initial_balance,

            transaction_cost=self.transaction_cost,

            episode_length=self.episode_length,

            random_start=False,

            drawdown_penalty=self.drawdown_penalty,

            downside_penalty=self.downside_penalty,

            directional_weight=self.directional_weight,

            turnover_penalty=self.turnover_penalty,
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

    def render(self):
        if self.current_env is not None:
            return self.current_env.render()

        return None

    def close(self):
        if self.current_env is not None:
            self.current_env.close()

        self.current_env = None