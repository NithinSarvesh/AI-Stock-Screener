"""
PPO V9 Trading Environment

Goals:
- 5-position action space
- Explicit candle-structure features
- Technical + momentum + volatility + trend context
- Correct observation dimension generated from feature list
- Multi-stock compatible
- Risk-aware reward
- No hard-coded 30/31 dimensional mismatch

Actions:
    0 = SHORT       (-1.0)
    1 = HALF_SHORT  (-0.5)
    2 = FLAT         (0.0)
    3 = HALF_LONG   (+0.5)
    4 = LONG        (+1.0)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd


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
# V9 MARKET FEATURES
# =====================================================================

MARKET_FEATURES = [

    # -------------------------------------------------------------
    # Candle geometry
    # -------------------------------------------------------------

    "CANDLE_BODY",
    "CANDLE_RANGE",
    "UPPER_WICK",
    "LOWER_WICK",
    "BODY_RANGE_RATIO",
    "CLOSE_POSITION",
    "GAP",

    # Previous candle relationships
    "BODY_CHANGE",
    "RANGE_CHANGE",

    # -------------------------------------------------------------
    # Technical indicators
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Market context
    # -------------------------------------------------------------

    "RET_1",
    "RET_5",
    "RET_20",

    "VOL_20",

    "EMA20_SLOPE",
    "EMA50_SLOPE",

    "TREND_SCORE",

    # -------------------------------------------------------------
    # Volume
    # -------------------------------------------------------------

    "VOLUME_RATIO",
]


# Position is appended dynamically.
OBSERVATION_SIZE = len(
    MARKET_FEATURES
) + 1


# =====================================================================
# ENVIRONMENT
# =====================================================================

class StockTradingEnvV9(gym.Env):

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
        # Reward parameters
        # -------------------------------------------------------------

        drawdown_penalty: float = 0.01,
        downside_penalty: float = 0.02,

        # Penalizes exposure when the model is not aligned with
        # the short-term directional signal.
        neutral_exposure_penalty: float = 0.005,

        # Penalizes unnecessary position switching.
        turnover_penalty: float = 0.01,

        # Small reward for correctly staying flat when the market
        # has weak directional movement.
        flat_reward_weight: float = 0.0,
    ):

        super().__init__()

        if dataframe is None:
            raise ValueError(
                "V9 dataframe is None."
            )

        if dataframe.empty:
            raise ValueError(
                "V9 dataframe is empty."
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

        self.drawdown_penalty = float(
            drawdown_penalty
        )

        self.downside_penalty = float(
            downside_penalty
        )

        self.neutral_exposure_penalty = float(
            neutral_exposure_penalty
        )

        self.turnover_penalty = float(
            turnover_penalty
        )

        self.flat_reward_weight = float(
            flat_reward_weight
        )

        # -------------------------------------------------------------
        # Required source columns
        # -------------------------------------------------------------

        required_columns = [

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

            "ATR",
            "ADX",
            "OBV",
            "STOCH_RSI",

            "RET_1",
            "RET_5",
            "RET_20",

            "VOL_20",

            "EMA20_SLOPE",
            "EMA50_SLOPE",

            "TREND_SCORE",
        ]

        missing = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing:

            raise ValueError(
                "V9 missing columns: "
                + ", ".join(missing)
            )

        # -------------------------------------------------------------
        # Clean
        # -------------------------------------------------------------

        self.df = (
            self.df
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .copy()
        )

        # =============================================================
        # CANDLE FEATURES
        # =============================================================

        close = self.df["Close"].astype(float)
        open_ = self.df["Open"].astype(float)
        high = self.df["High"].astype(float)
        low = self.df["Low"].astype(float)

        previous_close = (
            close.shift(1)
        )

        body = (
            close - open_
        )

        candle_range = (
            high - low
        ).clip(
            lower=1e-8
        )

        upper_wick = (
            high
            - np.maximum(
                open_,
                close,
            )
        )

        lower_wick = (
            np.minimum(
                open_,
                close,
            )
            - low
        )

        self.df["CANDLE_BODY"] = (
            body / close.abs().clip(
                lower=1e-8
            )
        )

        self.df["CANDLE_RANGE"] = (
            candle_range / close.abs().clip(
                lower=1e-8
            )
        )

        self.df["UPPER_WICK"] = (
            upper_wick / close.abs().clip(
                lower=1e-8
            )
        )

        self.df["LOWER_WICK"] = (
            lower_wick / close.abs().clip(
                lower=1e-8
            )
        )

        self.df["BODY_RANGE_RATIO"] = (
            body.abs()
            / candle_range
        )

        self.df["CLOSE_POSITION"] = (
            (close - low)
            / candle_range
        )

        self.df["GAP"] = (
            close / previous_close
            - 1.0
        )

        previous_body = (
            body.shift(1)
        )

        previous_range = (
            candle_range.shift(1)
        )

        self.df["BODY_CHANGE"] = (
            body.abs()
            / previous_body.abs().clip(
                lower=1e-8
            )
            - 1.0
        )

        self.df["RANGE_CHANGE"] = (
            candle_range
            / previous_range.clip(
                lower=1e-8
            )
            - 1.0
        )

        # =============================================================
        # NORMALIZED TECHNICAL FEATURES
        # =============================================================

        self.df["EMA20_DISTANCE"] = (
            self.df["EMA20"]
            / close
            - 1.0
        )

        self.df["EMA50_DISTANCE"] = (
            self.df["EMA50"]
            / close
            - 1.0
        )

        self.df["EMA200_DISTANCE"] = (
            self.df["EMA200"]
            / close
            - 1.0
        )

        self.df["RSI_NORMALIZED"] = (
            self.df["RSI"]
            - 50.0
        ) / 50.0

        self.df["MACD_NORMALIZED"] = (
            self.df["MACD"]
            / close
        )

        self.df[
            "MACD_SIGNAL_NORMALIZED"
        ] = (
            self.df["MACD_SIGNAL"]
            / close
        )

        self.df[
            "MACD_HIST_NORMALIZED"
        ] = (
            self.df["MACD_HISTOGRAM"]
            / close
        )

        self.df[
            "BB_UPPER_DISTANCE"
        ] = (
            self.df["BB_UPPER"]
            / close
            - 1.0
        )

        self.df[
            "BB_MIDDLE_DISTANCE"
        ] = (
            self.df["BB_MIDDLE"]
            / close
            - 1.0
        )

        self.df[
            "BB_LOWER_DISTANCE"
        ] = (
            self.df["BB_LOWER"]
            / close
            - 1.0
        )

        self.df["VWAP_DISTANCE"] = (
            self.df["VWAP"]
            / close
            - 1.0
        )

        self.df["ATR_PCT"] = (
            self.df["ATR"]
            / close
        )

        self.df["ADX_NORMALIZED"] = (
            self.df["ADX"]
            / 100.0
        )

        self.df["OBV_DIRECTION"] = (
            np.sign(
                self.df["OBV"]
                .diff()
                .fillna(0)
            )
        )

        self.df[
            "STOCH_RSI_NORMALIZED"
        ] = (
            self.df["STOCH_RSI"] * 2.0
            - 1.0
        )

        self.df["VOLUME_RATIO"] = (
            self.df["Volume"]
            /
            self.df["Volume"]
            .rolling(20)
            .mean()
            .replace(0, np.nan)
        )

        # -------------------------------------------------------------
        # Final cleaning
        # -------------------------------------------------------------

        self.df = (
            self.df
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
            .reset_index(
                drop=True
            )
        )

        if len(self.df) < 100:

            raise ValueError(
                "Not enough valid rows after V9 "
                "feature construction."
            )

        # =============================================================
        # OBSERVATION SPACE
        # =============================================================

        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(
                OBSERVATION_SIZE,
            ),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Discrete(
            len(ACTION_MAP)
        )

        # =============================================================
        # STATE
        # =============================================================

        self.current_step = 0
        self.start_step = 0
        self.end_step = 0

        self.position = 0.0

        self.equity = (
            self.initial_balance
        )

        self.peak_equity = (
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

        feature_values = []

        for feature in MARKET_FEATURES:

            value = float(
                row[feature]
            )

            # Percentage/ratio features can occasionally become
            # extreme. Bound everything consistently.
            value = float(
                np.clip(
                    value,
                    -10.0,
                    10.0,
                )
            )

            feature_values.append(
                value
            )

        feature_values.append(
            float(
                self.position
            )
        )

        observation = np.asarray(
            feature_values,
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
            -10.0,
            10.0,
        )

        expected = OBSERVATION_SIZE

        if observation.shape != (
            expected,
        ):

            raise RuntimeError(
                "V9 observation mismatch: "
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

        self.position = 0.0

        self.equity = (
            self.initial_balance
        )

        self.peak_equity = (
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
                f"Invalid V9 action: {action}"
            )

        target_position = ACTION_MAP[
            action
        ]["position"]

        current_row = self.df.iloc[
            self.current_step
        ]

        next_step = min(
            self.current_step + 1,
            len(self.df) - 1,
        )

        next_row = self.df.iloc[
            next_step
        ]

        current_price = max(
            float(
                current_row["Close"]
            ),
            1e-8,
        )

        next_price = max(
            float(
                next_row["Close"]
            ),
            1e-8,
        )

        market_return = (
            next_price
            / current_price
            - 1.0
        )

        old_position = (
            self.position
        )

        position_change = (
            target_position
            - old_position
        )

        turnover = abs(
            position_change
        )

        # =============================================================
        # TRANSACTION COST
        # =============================================================

        transaction_cost = (
            turnover
            * self.transaction_cost
            * self.equity
        )

        normalized_cost = (
            transaction_cost
            / max(
                self.equity,
                1e-8,
            )
        )

        # =============================================================
        # STRATEGY RETURN
        # =============================================================

        strategy_return = (
            target_position
            * market_return
        )

        old_equity = (
            self.equity
        )

        self.equity *= (
            1.0
            + strategy_return
        )

        self.equity -= (
            transaction_cost
        )

        self.position = (
            target_position
        )

        # =============================================================
        # DRAW DOWN
        # =============================================================

        self.peak_equity = max(
            self.peak_equity,
            self.equity,
        )

        drawdown = (
            self.equity
            / max(
                self.peak_equity,
                1e-8,
            )
            - 1.0
        )

        # =============================================================
        # MARKET DIRECTION SIGNAL
        # =============================================================

        ret5 = float(
            current_row["RET_5"]
        )

        ret20 = float(
            current_row["RET_20"]
        )

        ema20_slope = float(
            current_row["EMA20_SLOPE"]
        )

        ema50_slope = float(
            current_row["EMA50_SLOPE"]
        )

        trend_score = float(
            current_row["TREND_SCORE"]
        )

        # Combine medium-term direction signals.
        directional_signal = np.clip(
            (
                0.30 * np.tanh(ret5 * 20.0)
                +
                0.25 * np.tanh(ret20 * 10.0)
                +
                0.20 * np.tanh(ema20_slope * 50.0)
                +
                0.15 * np.tanh(ema50_slope * 50.0)
                +
                0.10 * (
                    trend_score * 2.0
                    - 1.0
                )
            ),
            -1.0,
            1.0,
        )

        # =============================================================
        # CORE REWARD
        # =============================================================

        # Reward is based on actual realized portfolio return.
        reward = strategy_return

        # =============================================================
        # DOWN-SIDE PENALTY
        # =============================================================

        if strategy_return < 0:

            reward -= (
                self.downside_penalty
                * abs(strategy_return)
            )

        # =============================================================
        # DRAW-DOWN PENALTY
        # =============================================================

        reward -= (
            self.drawdown_penalty
            * abs(drawdown)
        )

        # =============================================================
        # DIRECTIONAL ALIGNMENT
        # =============================================================

        # Penalize large exposure that contradicts the
        # medium-term directional context.
        exposure_misalignment = (
            abs(target_position)
            * max(
                0.0,
                -(
                    target_position
                    * directional_signal
                )
            )
        )

        reward -= (
            self.neutral_exposure_penalty
            * exposure_misalignment
            * abs(strategy_return)
        )

        # =============================================================
        # FLAT DECISION
        # =============================================================

        # When directional signal is weak, FLAT is a valid decision.
        

        # =============================================================
        # TURNOVER
        # =============================================================

        reward -= (
            self.turnover_penalty
            * turnover
            * self.transaction_cost
        )

        # Transaction cost itself.
        reward -= (
            normalized_cost
        )

        reward = float(
            np.clip(
                reward,
                -1.0,
                1.0,
            )
        )

        self.total_reward += (
            reward
        )

        # =============================================================
        # ADVANCE
        # =============================================================

        self.current_step = (
            next_step
        )

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
                ACTION_MAP[action]["name"],

            "position":
                self.position,

            "market_return":
                market_return,

            "strategy_return":
                strategy_return,

            "directional_signal":
                float(
                    directional_signal
                ),

            "drawdown":
                float(drawdown),

            "equity":
                float(self.equity),

            "transaction_cost":
                float(transaction_cost),

            "reward":
                reward,

            "total_reward":
                self.total_reward,

            "old_equity":
                old_equity,
        }

        return (
            self._build_observation(),
            reward,
            terminated,
            truncated,
            info,
        )