from __future__ import annotations

import gymnasium as gym
import numpy as np

from champion_env import ChampionTradingEnv


class MultiStockChampionEnv(gym.Env):
    """Select one stock per episode; PPO learns one universal policy."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: dict,
        episode_length: int = 252,
        random_stock: bool = True,
        random_start: bool = True,
        transaction_cost: float = 0.0005,
        drawdown_penalty: float = 0.02,
        downside_penalty: float = 0.01,
        turnover_penalty: float = 0.001,
    ):
        super().__init__()

        if not data:
            raise ValueError("No stock data supplied.")

        self.data = data
        self.symbols = list(data.keys())
        self.episode_length = episode_length
        self.random_stock = random_stock
        self.random_start = random_start
        self.kwargs = dict(
            episode_length=episode_length,
            random_start=random_start,
            transaction_cost=transaction_cost,
            drawdown_penalty=drawdown_penalty,
            downside_penalty=downside_penalty,
            turnover_penalty=turnover_penalty,
        )

        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(33,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(5)

        self.current_env = None
        self.current_symbol = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.random_stock:
            idx = int(self.np_random.integers(0, len(self.symbols)))
        else:
            idx = 0

        self.current_symbol = self.symbols[idx]
        self.current_env = ChampionTradingEnv(
            self.data[self.current_symbol],
            **self.kwargs,
        )

        obs, info = self.current_env.reset(seed=seed)
        info["symbol"] = self.current_symbol
        return obs, info

    def step(self, action):
        if self.current_env is None:
            raise RuntimeError("Call reset() before step().")

        obs, reward, terminated, truncated, info = self.current_env.step(action)
        info["symbol"] = self.current_symbol
        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        self.current_env = None
