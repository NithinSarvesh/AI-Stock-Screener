from __future__ import annotations

import gymnasium as gym
import numpy as np

from rl.champion_v10_env import ChampionV10Env

class MultiStockChampionV10(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data: dict[str, object], **env_kwargs):
        super().__init__()
        if not data:
            raise ValueError("No stock data")
        self.data = data
        self.symbols = list(data.keys())
        self.env_kwargs = env_kwargs
        self.current_symbol = None
        self.current_env = None

        sample = self.symbols[0]
        self.current_env = ChampionV10Env(self.data[sample], **env_kwargs)
        self.observation_space = self.current_env.observation_space
        self.action_space = self.current_env.action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        idx = int(self.np_random.integers(0, len(self.symbols)))
        self.current_symbol = self.symbols[idx]
        self.current_env = ChampionV10Env(self.data[self.current_symbol], **self.env_kwargs)
        obs, info = self.current_env.reset(seed=seed)
        info["symbol"] = self.current_symbol
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        info["symbol"] = self.current_symbol
        return obs, reward, terminated, truncated, info
