"""
Stock Trading Environment V9.1

V9.1 changes:
- Keeps V9's 32 market features + current position = 33 observations.
- Keeps the same 5-position action space.
- Keeps V9's P&L, downside, drawdown, opportunity and turnover concepts.
- Fixes the directional-alignment penalty so it does not disappear
  when realized return is close to zero.

IMPORTANT:
This environment is an experiment. It does not modify V9.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


# =====================================================================
# ACTIONS
# =====================================================================

ACTION_MAP = {
    0: {
        "name": "SHORT",
        "position": -1.0,
    },
    1: {
        "name": "HALF_SHORT",
        "position": -0.5,
    },
    2: {
        "name": "FLAT",
        "position": 0.0,
    },
    3: {
        "name": "HALF_LONG",
        "position": 0.5,
    },
    4: {
        "name": "LONG",
        "position": 1.0,
    },
}


# =====================================================================
# MARKET FEATURES
# =====================================================================

MARKET_FEATURES = [
    "CANDLE_BODY",
    "CANDLE_RANGE",
    "UPPER_WICK",
    "LOWER_WICK",
    "BODY_RANGE_RATIO",
    "CLOSE_POSITION",
    "GAP",
    "BODY_CHANGE",
    "RANGE_CHANGE",
    "EMA20_DISTANCE",
    "EMA50_DISTANCE",
    "EMA200_DISTANCE",
    "RSI_NORMALIZED",
    "MACD_NORMALIZED",
    "MACD_SIGNAL_NORMALIZED",
    "MACD_HIST_NORMALIZED",
    "BB_UPPER_DISTANCE",
    "BB_MIDDLE_DISTANCE",
    "BB_LOWER_DISTANCE",
    "VWAP_DISTANCE",
    "ATR_PCT",
    "ADX_NORMALIZED",
    "OBV_DIRECTION",
    "STOCH_RSI_NORMALIZED",
    "RET_1",
    "RET_5",
    "RET_20",
    "VOL_20",
    "EMA20_SLOPE",
    "EMA50_SLOPE",
    "TREND_SCORE",
    "VOLUME_RATIO",
]

OBSERVATION_SIZE = 33


# =====================================================================
# ENVIRONMENT
# =====================================================================

class StockTradingEnvV91(gym.Env):

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

        # -------------------------------------------------------------
        # V9 risk parameters
        # -------------------------------------------------------------

        drawdown_penalty: float = 0.1,
        downside_penalty: float = 0.05,
        opportunity_weight: float = 0.10,
        turnover_penalty: float = 0.02,

        # -------------------------------------------------------------
        # V9.1 directional alignment.
        #
        # This penalty is deliberately small.
        # We do NOT want to hard-code trading rules.
        # -------------------------------------------------------------

        directional_penalty: float = 0.001,

    ):

        super().__init__()

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

        self.drawdown_penalty = float(
            drawdown_penalty
        )

        self.downside_penalty = float(
            downside_penalty
        )

        self.opportunity_weight = float(
            opportunity_weight
        )

        self.turnover_penalty = float(
            turnover_penalty
        )

        self.directional_penalty = float(
            directional_penalty
        )

        # -------------------------------------------------------------
        # Spaces
        # -------------------------------------------------------------

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(
                OBSERVATION_SIZE,
            ),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(
            5
        )

        # -------------------------------------------------------------
        # State
        # -------------------------------------------------------------

        self.current_step = 0

        self.start_step = 0

        self.end_step = 0

        self.balance = (
            self.initial_balance
        )

        self.equity = (
            self.initial_balance
        )

        self.peak_equity = (
            self.initial_balance
        )

        self.position = 0.0

        self.previous_position = 0.0

        self.trade_count = 0

        self.total_turnover = 0.0

    # =================================================================
    # NORMALIZATION
    # =================================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ):

        try:

            result = float(
                value
            )

            if not np.isfinite(
                result
            ):
                return default

            return result

        except Exception:

            return default

    @staticmethod
    def _clip(
        value: float,
        low: float = -10.0,
        high: float = 10.0,
    ):

        return float(
            np.clip(
                value,
                low,
                high,
            )
        )

    # =================================================================
    # OBSERVATION
    # =================================================================

    def _build_observation(
        self,
    ):

        row = self.df.iloc[
            self.current_step
        ]

        values = []

        for feature in MARKET_FEATURES:

            value = self._safe_float(
                row.get(
                    feature,
                    0.0,
                )
            )

            values.append(
                self._clip(
                    value
                )
            )

        # Current position.
        values.append(
            self._clip(
                self.position
            )
        )

        observation = np.asarray(
            values,
            dtype=np.float32,
        )

        return observation

    # =================================================================
    # DIRECTIONAL SIGNAL
    # =================================================================

    def _directional_signal(
        self,
        row,
    ):

        ret5 = self._safe_float(
            row.get(
                "RET_5",
                0.0,
            )
        )

        ret20 = self._safe_float(
            row.get(
                "RET_20",
                0.0,
            )
        )

        ema20_slope = self._safe_float(
            row.get(
                "EMA20_SLOPE",
                0.0,
            )
        )

        ema50_slope = self._safe_float(
            row.get(
                "EMA50_SLOPE",
                0.0,
            )
        )

        trend_score = self._safe_float(
            row.get(
                "TREND_SCORE",
                0.0,
            )
        )

        signal = (

            0.30
            * np.tanh(
                ret5 * 20.0
            )

            +

            0.25
            * np.tanh(
                ret20 * 10.0
            )

            +

            0.20
            * np.tanh(
                ema20_slope * 50.0
            )

            +

            0.15
            * np.tanh(
                ema50_slope * 50.0
            )

            +

            0.10
            * (
                trend_score * 2.0
                - 1.0
            )
        )

        return float(
            np.clip(
                signal,
                -1.0,
                1.0,
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

        total_rows = len(
            self.df
        )

        minimum_required = (
            self.episode_length
            + 2
        )

        if total_rows < minimum_required:

            raise ValueError(
                "Insufficient dataframe rows. "
                f"Need at least "
                f"{minimum_required}, "
                f"got {total_rows}."
            )

        max_start = (
            total_rows
            - self.episode_length
            - 1
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
            total_rows - 1,
        )

        self.balance = (
            self.initial_balance
        )

        self.equity = (
            self.initial_balance
        )

        self.peak_equity = (
            self.initial_balance
        )

        self.position = 0.0

        self.previous_position = 0.0

        self.trade_count = 0

        self.total_turnover = 0.0

        observation = (
            self._build_observation()
        )

        info = {
            "equity":
                self.equity,

            "position":
                self.position,

            "step":
                self.current_step,
        }

        return observation, info

    # =================================================================
    # STEP
    # =================================================================

    def step(
        self,
        action,
    ):

        action = int(
            action
        )

        if action not in ACTION_MAP:

            raise ValueError(
                f"Invalid action: {action}"
            )

        target_position = float(
            ACTION_MAP[action][
                "position"
            ]
        )

        current_row = self.df.iloc[
            self.current_step
        ]

        next_row = self.df.iloc[
            self.current_step + 1
        ]

        # -------------------------------------------------------------
        # Market return.
        # -------------------------------------------------------------

        current_close = self._safe_float(
            current_row.get(
                "Close",
                0.0,
            )
        )

        next_close = self._safe_float(
            next_row.get(
                "Close",
                current_close,
            )
        )

        if (
            current_close <= 0.0
            or next_close <= 0.0
        ):

            market_return = 0.0

        else:

            market_return = (
                next_close
                / current_close
                - 1.0
            )

        # -------------------------------------------------------------
        # Strategy return.
        # -------------------------------------------------------------

        strategy_return = (
            target_position
            * market_return
        )

        # -------------------------------------------------------------
        # Turnover.
        # -------------------------------------------------------------

        turnover = abs(
            target_position
            - self.position
        )

        transaction_cost = (
            turnover
            * self.transaction_cost
        )

        if turnover > 1e-12:

            self.trade_count += 1

        self.total_turnover += (
            turnover
        )

        # -------------------------------------------------------------
        # Base P&L reward.
        # -------------------------------------------------------------

        reward = strategy_return

        # -------------------------------------------------------------
        # Downside penalty.
        # -------------------------------------------------------------

        downside = max(
            0.0,
            -strategy_return,
        )

        reward -= (
            self.downside_penalty
            * downside
        )

        # -------------------------------------------------------------
        # Update equity.
        # -------------------------------------------------------------

        equity_return = (
            strategy_return
            - transaction_cost
        )

        self.equity *= (
            1.0
            + equity_return
        )

        self.balance = (
            self.equity
        )

        # -------------------------------------------------------------
        # Drawdown.
        # -------------------------------------------------------------

        self.peak_equity = max(
            self.peak_equity,
            self.equity,
        )

        drawdown = 0.0

        if self.peak_equity > 0:

            drawdown = (
                self.equity
                / self.peak_equity
                - 1.0
            )

        reward -= (
            self.drawdown_penalty
            * max(
                0.0,
                -drawdown,
            )
        )

        # -------------------------------------------------------------
        # Directional signal.
        # -------------------------------------------------------------

        directional_signal = (
            self._directional_signal(
                current_row
            )
        )

        # -------------------------------------------------------------
        # V9.1 CHANGE
        #
        # Penalize directional disagreement independently of realized
        # return.
        #
        # This fixes V9's problem where the directional penalty became
        # approximately zero whenever the market return was tiny.
        #
        # We still keep the penalty small so the model can override
        # the signal when actual price behavior proves it wrong.
        # -------------------------------------------------------------

        directional_alignment = (
            target_position
            * directional_signal
        )

        disagreement = max(
            0.0,
            -directional_alignment,
        )

        reward -= (
            self.directional_penalty
            * disagreement
        )

        # -------------------------------------------------------------
        # Opportunity penalty.
        #
        # Penalize being flat/underexposed when a directional move
        # actually occurs.
        # -------------------------------------------------------------

        missed_move = max(
            0.0,
            abs(market_return)
            * (
                1.0
                - abs(target_position)
            ),
        )

        reward -= (
            self.opportunity_weight
            * missed_move
        )

        # -------------------------------------------------------------
        # Turnover penalty.
        # -------------------------------------------------------------

        reward -= (
            self.turnover_penalty
            * turnover
        )

        # -------------------------------------------------------------
        # Transaction cost.
        # -------------------------------------------------------------

        reward -= transaction_cost

        # -------------------------------------------------------------
        # Advance.
        # -------------------------------------------------------------

        self.previous_position = (
            self.position
        )

        self.position = (
            target_position
        )

        self.current_step += 1

        terminated = (
            self.current_step
            >= self.end_step
        )

        truncated = False

        observation = (
            self._build_observation()
        )

        info = {
            "market_return":
                market_return,

            "strategy_return":
                strategy_return,

            "equity":
                self.equity,

            "position":
                self.position,

            "target_position":
                target_position,

            "turnover":
                turnover,

            "transaction_cost":
                transaction_cost,

            "drawdown":
                drawdown,

            "directional_signal":
                directional_signal,

            "directional_alignment":
                directional_alignment,

            "directional_disagreement":
                disagreement,

            "reward":
                reward,

            "trade_count":
                self.trade_count,
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )