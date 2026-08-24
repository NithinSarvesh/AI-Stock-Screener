from __future__ import annotations

from typing import Any
import gymnasium as gym
import numpy as np
import pandas as pd

from champion_features import MARKET_FEATURES, OBSERVATION_SIZE


ACTION_MAP = {
    0: {"name": "SHORT", "position": -1.0},
    1: {"name": "HALF_SHORT", "position": -0.5},
    2: {"name": "FLAT", "position": 0.0},
    3: {"name": "HALF_LONG", "position": 0.5},
    4: {"name": "LONG", "position": 1.0},
}


class ChampionTradingEnv(gym.Env):
    """
    Clean economic RL environment.

    Reward is portfolio growth (log equity return) with only modest,
    economically interpretable risk/turnover penalties.
    No handcrafted directional reward is used.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataframe: pd.DataFrame,
        episode_length: int = 252,
        random_start: bool = True,
        transaction_cost: float = 0.0005,
        drawdown_penalty: float = 0.02,
        downside_penalty: float = 0.01,
        turnover_penalty: float = 0.001,
    ):
        super().__init__()

        if dataframe is None or dataframe.empty:
            raise ValueError("ChampionTradingEnv received empty data.")

        self.df = dataframe.reset_index(drop=False).copy()
        self.episode_length = int(episode_length)
        self.random_start = bool(random_start)
        self.transaction_cost = float(transaction_cost)
        self.drawdown_penalty = float(drawdown_penalty)
        self.downside_penalty = float(downside_penalty)
        self.turnover_penalty = float(turnover_penalty)

        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(5)

        self.current_step = 0
        self.start_step = 0
        self.end_step = 0
        self.position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.trade_count = 0
        self.total_turnover = 0.0

    @staticmethod
    def _f(v, default=0.0):
        try:
            x = float(v)
            return x if np.isfinite(x) else default
        except Exception:
            return default

    def _obs(self):
        row = self.df.iloc[self.current_step]
        vals = [self._f(row.get(c, 0.0)) for c in MARKET_FEATURES]
        vals.append(self.position)
        return np.clip(np.asarray(vals, dtype=np.float32), -10.0, 10.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        max_start = len(self.df) - self.episode_length - 1
        if max_start < 0:
            raise ValueError(
                f"Need at least {self.episode_length + 1} rows, got {len(self.df)}."
            )

        if self.random_start and max_start > 0:
            self.start_step = int(self.np_random.integers(0, max_start + 1))
        else:
            self.start_step = 0

        self.current_step = self.start_step
        self.end_step = min(
            self.start_step + self.episode_length,
            len(self.df) - 1,
        )

        self.position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.trade_count = 0
        self.total_turnover = 0.0

        return self._obs(), {
            "step": self.current_step,
            "position": self.position,
            "equity": self.equity,
        }

    def step(self, action):
        action = int(action)
        if action not in ACTION_MAP:
            raise ValueError(f"Invalid action {action}")

        target = ACTION_MAP[action]["position"]

        current = self.df.iloc[self.current_step]
        nxt = self.df.iloc[self.current_step + 1]

        close0 = max(self._f(current["Close"]), 1e-8)
        close1 = max(self._f(nxt["Close"]), 1e-8)
        market_return = close1 / close0 - 1.0

        turnover = abs(target - self.position)
        cost = turnover * self.transaction_cost

        strategy_return = target * market_return
        equity_return = strategy_return - cost

        previous_equity = self.equity
        self.equity *= max(1e-8, 1.0 + equity_return)
        self.peak_equity = max(self.peak_equity, self.equity)

        drawdown = self.equity / self.peak_equity - 1.0
        downside = max(0.0, -strategy_return)

        # Core objective = actual portfolio growth.
        reward = np.log(max(self.equity, 1e-8) / max(previous_equity, 1e-8))

        # Small stabilizers. They cannot dominate returns.
        reward -= self.drawdown_penalty * max(0.0, -drawdown)
        reward -= self.downside_penalty * downside
        reward -= self.turnover_penalty * turnover

        if turnover > 1e-12:
            self.trade_count += 1
        self.total_turnover += turnover

        self.position = target
        self.current_step += 1

        terminated = self.current_step >= self.end_step
        truncated = False

        info = {
            "market_return": market_return,
            "strategy_return": strategy_return,
            "equity": self.equity,
            "position": self.position,
            "turnover": turnover,
            "transaction_cost": cost,
            "drawdown": drawdown,
            "reward": float(reward),
            "trade_count": self.trade_count,
        }

        return self._obs(), float(reward), terminated, truncated, info
