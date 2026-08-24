"""
PPO V7 Trading Environment

V7 keeps the 5-position action architecture used by PPO V6:

    0 = SHORT       (-1.0)
    1 = HALF_SHORT  (-0.5)
    2 = FLAT         (0.0)
    3 = HALF_LONG   (+0.5)
    4 = LONG        (+1.0)

V7 is deliberately separate from trading_env.py so that the
working V6 environment remains untouched.

The environment accepts one prepared OHLCV dataframe at a time.
The V7 trainer will handle the multi-stock universe.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd


# =====================================================================
# ACTION MAP
# =====================================================================

ACTION_MAP = {
    0: -1.0,    # SHORT
    1: -0.5,    # HALF SHORT
    2: 0.0,     # FLAT
    3: 0.5,     # HALF LONG
    4: 1.0,     # LONG
}


ACTION_NAMES = {
    0: "SHORT",
    1: "HALF_SHORT",
    2: "FLAT",
    3: "HALF_LONG",
    4: "LONG",
}


# =====================================================================
# ENVIRONMENT
# =====================================================================

class StockTradingEnvV7(gym.Env):

    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        dataframe: pd.DataFrame,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.0005,
        episode_length: int = 252,
        random_start: bool = True,
    ):

        super().__init__()

        if dataframe is None or dataframe.empty:
            raise ValueError(
                "Trading dataframe is empty."
            )

        self.df = dataframe.copy()

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

        # -------------------------------------------------------------
        # Required features
        # -------------------------------------------------------------

        self.feature_columns = [

            "Open",
            "High",
            "Low",
            "Close",
            "Volume",

            "EMA20",
            "EMA50",
            "EMA200",

            "RSI",

            "MACD",
            "MACD_SIGNAL",
            "MACD_HISTOGRAM",

            "BB_UPPER",
            "BB_MIDDLE",
            "BB_LOWER",

            "VWAP",

            "AVG_VOLUME",

            "ATR",

            "ADX",

            "OBV",

            "STOCH_RSI",

            # Context features
            "RET_1",
            "RET_5",
            "RET_20",
            "VOL_20",
            "EMA20_SLOPE",
            "EMA50_SLOPE",
            "ATR_PCT",
            "TREND_SCORE",
        ]

        missing = [
            column
            for column in self.feature_columns
            if column not in self.df.columns
        ]

        if missing:

            raise ValueError(
                "Missing V7 features: "
                + ", ".join(missing)
            )

        # -------------------------------------------------------------
        # Clean data
        # -------------------------------------------------------------

        self.df = (
            self.df
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna(
                subset=self.feature_columns
            )
            .reset_index(
                drop=True
            )
        )

        if len(self.df) < 50:

            raise ValueError(
                "Not enough valid rows for V7."
            )

        # -------------------------------------------------------------
        # Observation
        #
        # 30 market/context features
        # + current position
        #
        # This gives V7 30 features from the feature list plus
        # the portfolio position.
        # -------------------------------------------------------------

        self.market_feature_count = len(
            self.feature_columns
        )

        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(
                self.market_feature_count + 1,
            ),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Discrete(
            5
        )

        # -------------------------------------------------------------
        # Runtime state
        # -------------------------------------------------------------

        self.current_step = 0
        self.start_step = 0
        self.end_step = 0

        self.cash = self.initial_balance
        self.position = 0.0
        self.equity = self.initial_balance

        self.previous_equity = (
            self.initial_balance
        )

        self.total_reward = 0.0

    # =================================================================
    # OBSERVATION
    # =================================================================

    def _build_observation(self):

        row = self.df.iloc[
            self.current_step
        ]

        close = max(
            float(row["Close"]),
            1e-8,
        )

        values = []

        # -------------------------------------------------------------
        # OHLC
        # -------------------------------------------------------------

        values.append(0.0)

        values.append(
            np.clip(
                float(row["High"]) / close - 1.0,
                -5,
                5,
            )
        )

        values.append(
            np.clip(
                float(row["Low"]) / close - 1.0,
                -5,
                5,
            )
        )

        values.append(0.0)

        # -------------------------------------------------------------
        # Volume
        # -------------------------------------------------------------

        values.append(
            np.clip(
                np.log1p(
                    max(
                        float(row["Volume"]),
                        1.0,
                    )
                    /
                    max(
                        float(row["AVG_VOLUME"]),
                        1.0,
                    )
                ),
                -10,
                10,
            )
        )

        # -------------------------------------------------------------
        # EMA
        # -------------------------------------------------------------

        for column in [
            "EMA20",
            "EMA50",
            "EMA200",
        ]:

            values.append(
                np.clip(
                    float(row[column]) / close - 1.0,
                    -5,
                    5,
                )
            )

        # -------------------------------------------------------------
        # RSI
        # -------------------------------------------------------------

        values.append(
            np.clip(
                (
                    float(row["RSI"]) - 50.0
                ) / 50.0,
                -1,
                1,
            )
        )

        # -------------------------------------------------------------
        # MACD
        # -------------------------------------------------------------

        for column in [
            "MACD",
            "MACD_SIGNAL",
            "MACD_HISTOGRAM",
        ]:

            values.append(
                np.clip(
                    float(row[column]) / close,
                    -1,
                    1,
                )
            )

        # -------------------------------------------------------------
        # Bollinger Bands
        # -------------------------------------------------------------

        for column in [
            "BB_UPPER",
            "BB_MIDDLE",
            "BB_LOWER",
        ]:

            values.append(
                np.clip(
                    float(row[column]) / close - 1.0,
                    -5,
                    5,
                )
            )

        # -------------------------------------------------------------
        # VWAP
        # -------------------------------------------------------------

        values.append(
            np.clip(
                float(row["VWAP"]) / close - 1.0,
                -5,
                5,
            )
        )

        # -------------------------------------------------------------
        # Average volume
        # -------------------------------------------------------------

        values.append(0.0)

        # -------------------------------------------------------------
        # ATR
        # -------------------------------------------------------------

        values.append(
            np.clip(
                float(row["ATR"]) / close,
                0,
                1,
            )
        )

        # -------------------------------------------------------------
        # ADX
        # -------------------------------------------------------------

        values.append(
            np.clip(
                float(row["ADX"]) / 100.0,
                0,
                1,
            )
        )

        # -------------------------------------------------------------
        # OBV
        # -------------------------------------------------------------

        values.append(
            np.sign(
                float(row["OBV"])
            )
        )

        # -------------------------------------------------------------
        # Stochastic RSI
        # -------------------------------------------------------------

        values.append(
            np.clip(
                float(row["STOCH_RSI"]) * 2.0 - 1.0,
                -1,
                1,
            )
        )

        # -------------------------------------------------------------
        # Context
        # -------------------------------------------------------------

        values.extend([

            np.clip(
                float(row["RET_1"]),
                -1,
                1,
            ),

            np.clip(
                float(row["RET_5"]),
                -1,
                1,
            ),

            np.clip(
                float(row["RET_20"]),
                -1,
                1,
            ),

            np.clip(
                float(row["VOL_20"]),
                0,
                1,
            ),

            np.clip(
                float(row["EMA20_SLOPE"]),
                -1,
                1,
            ),

            np.clip(
                float(row["EMA50_SLOPE"]),
                -1,
                1,
            ),

            np.clip(
                float(row["ATR_PCT"]),
                0,
                1,
            ),

            np.clip(
                float(row["TREND_SCORE"]) * 2.0 - 1.0,
                -1,
                1,
            ),
        ])

        # -------------------------------------------------------------
        # Current position
        # -------------------------------------------------------------

        values.append(
            float(self.position)
        )

        observation = np.asarray(
            values,
            dtype=np.float32,
        )

        observation = np.nan_to_num(
            observation,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

        observation = np.clip(
            observation,
            -10,
            10,
        ).astype(
            np.float32
        )

        expected = (
            self.market_feature_count + 1
        )

        if observation.shape != (
            expected,
        ):

            raise RuntimeError(
                f"Invalid observation shape: "
                f"{observation.shape}; "
                f"expected ({expected},)"
            )

        return observation

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

        max_start = max(
            0,
            len(self.df)
            - self.episode_length
            - 1,
        )

        if (
            self.random_start
            and max_start > 0
        ):

            self.start_step = int(
                self.np_random.integers(
                    0,
                    max_start + 1,
                )
            )

        else:

            self.start_step = 0

        self.current_step = (
            self.start_step
        )

        self.end_step = min(
            self.start_step
            + self.episode_length,
            len(self.df) - 1,
        )

        self.cash = (
            self.initial_balance
        )

        self.position = 0.0

        self.equity = (
            self.initial_balance
        )

        self.previous_equity = (
            self.initial_balance
        )

        self.total_reward = 0.0

        return (
            self._build_observation(),
            {},
        )

    # =================================================================
    # STEP
    # =================================================================

    def step(
        self,
        action,
    ):

        action = int(action)

        if action not in ACTION_MAP:

            raise ValueError(
                f"Invalid action: {action}"
            )

        target_position = ACTION_MAP[
            action
        ]

        current_price = max(
            float(
                self.df.iloc[
                    self.current_step
                ]["Close"]
            ),
            1e-8,
        )

        # -------------------------------------------------------------
        # Position change
        # -------------------------------------------------------------

        position_change = (
            target_position
            - self.position
        )

        turnover = abs(
            position_change
        )

        transaction_cost = (
            turnover
            * self.transaction_cost
            * self.equity
        )

        # -------------------------------------------------------------
        # Move forward one bar
        # -------------------------------------------------------------

        next_step = min(
            self.current_step + 1,
            len(self.df) - 1,
        )

        next_price = max(
            float(
                self.df.iloc[
                    next_step
                ]["Close"]
            ),
            1e-8,
        )

        price_return = (
            next_price / current_price
            - 1.0
        )

        # -------------------------------------------------------------
        # Portfolio return
        # -------------------------------------------------------------

        position_return = (
            target_position
            * price_return
        )

        pnl = (
            self.equity
            * position_return
        )

        self.equity = (
            self.equity
            + pnl
            - transaction_cost
        )

        self.position = (
            target_position
        )

        self.previous_equity = (
            self.equity
        )

        self.current_step = (
            next_step
        )

        # -------------------------------------------------------------
        # Reward
        #
        # Log equity return is more numerically stable than raw P&L.
        # -------------------------------------------------------------

        equity_return = (
            self.equity
            / max(
                self.initial_balance,
                1e-8,
            )
        )

        reward = (
            position_return
            - (
                transaction_cost
                /
                max(
                    self.initial_balance,
                    1e-8,
                )
            )
        )

        self.total_reward += reward

        # -------------------------------------------------------------
        # Termination
        # -------------------------------------------------------------

        terminated = (
            self.current_step
            >= self.end_step
        )

        truncated = (
            self.equity <= 0
        )

        info = {

            "action": action,

            "action_name":
                ACTION_NAMES[action],

            "position":
                self.position,

            "price_return":
                price_return,

            "position_return":
                position_return,

            "transaction_cost":
                transaction_cost,

            "equity":
                self.equity,

            "equity_return":
                equity_return,

            "total_reward":
                self.total_reward,
        }

        return (
            self._build_observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )