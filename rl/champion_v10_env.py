from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from rl.champion_v10_features import FEATURES, OBS_SIZE

ACTION_TO_POSITION = {
    0: -1.0,
    1: -0.5,
    2: 0.0,
    3: 0.5,
    4: 1.0,
}

class ChampionV10Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dataframe: pd.DataFrame,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.0005,
        episode_length: int = 252,
        random_start: bool = True,
        min_hold_days: int = 3,
        cooldown_days: int = 2,
        turnover_penalty: float = 0.001,
        drawdown_penalty: float = 0.01,
        downside_penalty: float = 0.01,
        signal_threshold: float = 0.08,
    ):
        super().__init__()
        if dataframe is None or dataframe.empty:
            raise ValueError("Empty dataframe")

        self.df = dataframe.copy()
        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.episode_length = int(episode_length)
        self.random_start = bool(random_start)
        self.min_hold_days = int(min_hold_days)
        self.cooldown_days = int(cooldown_days)
        self.turnover_penalty = float(turnover_penalty)
        self.drawdown_penalty = float(drawdown_penalty)
        self.downside_penalty = float(downside_penalty)
        self.signal_threshold = float(signal_threshold)

        self.observation_space = spaces.Box(-10, 10, (OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

        self.current_step = 0
        self.start_step = 0
        self.end_step = 0
        self.position = 0.0
        self.hold_days = 0
        self.cooldown = 0
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.previous_equity = self.initial_balance
        self.trade_count = 0

    def _base_signal(self) -> float:
        return float(np.clip(self.df.iloc[self.current_step]["SIGNAL_SCORE"], -1, 1))

    def _allowed_position(self, requested: float, signal: float) -> float:
        # Weak signals cannot justify large exposure.
        if abs(signal) < self.signal_threshold:
            return 0.0

        max_pos = min(1.0, abs(signal) * 1.8)
        if signal > 0:
            allowed = [0.0, 0.5, 1.0]
            return min(max_pos, requested) if requested > 0 else 0.0
        allowed = [0.0, -0.5, -1.0]
        return max(-max_pos, requested) if requested < 0 else 0.0

    def _observation(self):
        row = self.df.iloc[self.current_step]
        vals = row[FEATURES].astype(float).to_numpy()
        vals[FEATURES.index("POSITION")] = self.position
        vals[FEATURES.index("HOLD_DAYS")] = min(self.hold_days / 20.0, 1.0)
        vals[FEATURES.index("COOLDOWN")] = min(self.cooldown / max(self.cooldown_days, 1), 1.0)
        vals[FEATURES.index("SIGNAL_SCORE")] = self._base_signal()
        return np.clip(np.nan_to_num(vals, nan=0.0, posinf=10, neginf=-10), -10, 10).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        max_start = max(0, len(self.df) - self.episode_length - 1)
        if self.random_start and max_start > 0:
            self.start_step = int(self.np_random.integers(0, max_start + 1))
        else:
            self.start_step = 0

        self.current_step = self.start_step
        self.end_step = min(len(self.df) - 1, self.start_step + self.episode_length)
        self.position = 0.0
        self.hold_days = 0
        self.cooldown = 0
        self.equity = self.initial_balance
        self.previous_equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.trade_count = 0

        return self._observation(), {
            "equity": self.equity,
            "position": self.position,
        }

    def step(self, action):
        action = int(action)
        requested = ACTION_TO_POSITION[action]

        if self.current_step >= self.end_step:
            return self._observation(), 0.0, True, False, {}

        signal = self._base_signal()
        old_position = self.position

        # Prevent rapid flip-flopping.
        if self.cooldown > 0 and requested != old_position:
            requested = old_position

        if self.hold_days < self.min_hold_days and old_position != 0 and requested != old_position:
            requested = old_position

        new_position = self._allowed_position(requested, signal)

        # If we are in a position, don't exit on a tiny signal unless it persists.
        if old_position != 0 and new_position == 0 and abs(signal) < self.signal_threshold * 1.5:
            new_position = old_position

        next_step = self.current_step + 1
        close_now = float(self.df.iloc[self.current_step]["Close"])
        close_next = float(self.df.iloc[next_step]["Close"])
        market_return = close_next / close_now - 1.0

        strategy_return = new_position * market_return
        turnover = abs(new_position - old_position)
        costs = turnover * self.transaction_cost

        self.previous_equity = self.equity
        self.equity *= max(1e-6, 1.0 + strategy_return - costs)
        self.peak_equity = max(self.peak_equity, self.equity)

        log_return = float(np.log(max(self.equity, 1e-9) / max(self.previous_equity, 1e-9)))
        drawdown = max(0.0, 1.0 - self.equity / max(self.peak_equity, 1e-9))
        downside = max(0.0, -strategy_return)

        # Main objective = portfolio growth.
        reward = log_return
        reward -= self.turnover_penalty * turnover
        reward -= self.drawdown_penalty * drawdown
        reward -= self.downside_penalty * downside

        if turnover > 0:
            self.trade_count += 1
            self.cooldown = self.cooldown_days
        else:
            self.cooldown = max(0, self.cooldown - 1)

        self.position = new_position
        if self.position == old_position and self.position != 0:
            self.hold_days += 1
        elif self.position != 0:
            self.hold_days = 1
        else:
            self.hold_days = 0

        self.current_step = next_step
        terminated = self.current_step >= self.end_step

        info = {
            "strategy_return": strategy_return,
            "market_return": market_return,
            "position": self.position,
            "trade": turnover > 0,
            "equity": self.equity,
            "drawdown": drawdown,
            "signal": signal,
            "trade_count": self.trade_count,
        }

        return self._observation(), float(reward), terminated, False, info
