"""
PPO V6 inference layer.

Loads the trained universal PPO V6 model and converts
current market data into the exact 30-feature observation
used during training.
"""

import os
import sys
import numpy as np
import pandas as pd

from stable_baselines3 import PPO


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from indicators import IndicatorEngine


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "universal_v6",
    "universal_ppo_v6.zip",
)


# ---------------------------------------------------------------------
# ACTION DEFINITIONS
# ---------------------------------------------------------------------

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


class PPOV6Inference:

    def __init__(self, model_path=MODEL_PATH):

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"PPO V6 model not found:\n{model_path}"
            )

        print("Loading PPO V6...")
        self.model = PPO.load(model_path)

        print("PPO V6 loaded successfully.")
        print("Observation space:", self.model.observation_space)
        print("Action space:", self.model.action_space)

    # -----------------------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------------------

    @staticmethod
    def add_context_features(df):

        df = df.copy()

        close = df["Close"].astype(float)

        df["RET_1"] = close.pct_change(1)
        df["RET_5"] = close.pct_change(5)
        df["RET_20"] = close.pct_change(20)

        df["VOL_20"] = (
            close.pct_change()
            .rolling(20)
            .std()
        )

        df["EMA20_SLOPE"] = (
            df["EMA20"].pct_change(5)
        )

        df["EMA50_SLOPE"] = (
            df["EMA50"].pct_change(10)
        )

        df["ATR_PCT"] = (
            df["ATR"] /
            close.replace(0, np.nan)
        )

        df["TREND_SCORE"] = (
            (close > df["EMA20"]).astype(float)
            +
            (df["EMA20"] > df["EMA50"]).astype(float)
            +
            (df["EMA50"] > df["EMA200"]).astype(float)
        ) / 3.0

        return (
            df
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .copy()
        )

    # -----------------------------------------------------------------
    # BUILD EXACT 30-DIMENSION OBSERVATION
    # -----------------------------------------------------------------

    @staticmethod
    def build_observation(row, current_position=0.0):

        close = max(float(row["Close"]), 1e-8)

        values = [

            # Open
            0.0,

            # High
            np.clip(
                float(row["High"]) / close - 1,
                -5,
                5
            ),

            # Low
            np.clip(
                float(row["Low"]) / close - 1,
                -5,
                5
            ),

            # Close
            0.0,

            # Volume relative to average volume
            np.clip(
                np.log1p(
                    max(float(row["Volume"]), 1)
                    /
                    max(float(row["AVG_VOLUME"]), 1)
                ),
                -10,
                10
            ),
        ]

        # EMA20 / EMA50 / EMA200
        for column in [
            "EMA20",
            "EMA50",
            "EMA200",
        ]:
            values.append(
                np.clip(
                    float(row[column]) / close - 1,
                    -5,
                    5
                )
            )

        # RSI
        values.append(
            np.clip(
                (float(row["RSI"]) - 50) / 50,
                -1,
                1
            )
        )

        # MACD family
        for column in [
            "MACD",
            "MACD_SIGNAL",
            "MACD_HISTOGRAM",
        ]:
            values.append(
                np.clip(
                    float(row[column]) / close,
                    -1,
                    1
                )
            )

        # Bollinger Bands
        for column in [
            "BB_UPPER",
            "BB_MIDDLE",
            "BB_LOWER",
        ]:
            values.append(
                np.clip(
                    float(row[column]) / close - 1,
                    -5,
                    5
                )
            )

        # VWAP
        values.append(
            np.clip(
                float(row["VWAP"]) / close - 1,
                -5,
                5
            )
        )

        # AVG_VOLUME
        values.append(0.0)

        # ATR
        values.append(
            np.clip(
                float(row["ATR"]) / close,
                0,
                1
            )
        )

        # ADX
        values.append(
            np.clip(
                float(row["ADX"]) / 100,
                0,
                1
            )
        )

        # OBV
        values.append(
            np.sign(float(row["OBV"]))
        )

        # Stochastic RSI
        values.append(
            np.clip(
                float(row["STOCH_RSI"]) * 2 - 1,
                -1,
                1
            )
        )

        # -------------------------------------------------------------
        # CONTEXT FEATURES
        # -------------------------------------------------------------

        values.extend([

            np.clip(
                float(row["RET_1"]),
                -1,
                1
            ),

            np.clip(
                float(row["RET_5"]),
                -1,
                1
            ),

            np.clip(
                float(row["RET_20"]),
                -1,
                1
            ),

            np.clip(
                float(row["VOL_20"]),
                0,
                1
            ),

            np.clip(
                float(row["EMA20_SLOPE"]),
                -1,
                1
            ),

            np.clip(
                float(row["EMA50_SLOPE"]),
                -1,
                1
            ),

            np.clip(
                float(row["ATR_PCT"]),
                0,
                1
            ),

            np.clip(
                float(row["TREND_SCORE"] * 2 - 1),
                -1,
                1
            ),

            # Current portfolio position
            float(current_position),
        ])

        observation = np.nan_to_num(
            np.asarray(
                values,
                dtype=np.float32
            ),
            nan=0,
            posinf=10,
            neginf=-10,
        )

        observation = np.clip(
            observation,
            -10,
            10
        ).astype(np.float32)

        if observation.shape != (30,):
            raise ValueError(
                f"Invalid observation shape: "
                f"{observation.shape}; expected (30,)"
            )

        return observation

    # -----------------------------------------------------------------
    # PREDICT
    # -----------------------------------------------------------------

    def predict_from_dataframe(
        self,
        df,
        current_position=0.0,
    ):

        if df is None or df.empty:
            raise ValueError("Market dataframe is empty.")

        # Calculate the same indicators used during training
        features = IndicatorEngine(
            df.copy()
        ).calculate_all()

        features = self.add_context_features(
            features
        )

        if features.empty:
            raise ValueError(
                "No usable rows after indicator calculation."
            )

        latest = features.iloc[-1]

        observation = self.build_observation(
            latest,
            current_position=current_position,
        )

        # Deterministic prediction
        action, _ = self.model.predict(
            observation,
            deterministic=True,
        )

        action = int(
            np.asarray(action).item()
        )

        result = ACTION_MAP[action].copy()

        result.update({

            "action_id": action,

            "position": result["position"],

            "current_price":
                float(latest["Close"]),

            "observation_size":
                int(observation.shape[0]),

            "observation":
                observation.tolist(),

            "model":
                "PPO V6",

        })

        return result


# ---------------------------------------------------------------------
# SIMPLE FUNCTION FOR OTHER PROJECT FILES
# ---------------------------------------------------------------------

_model = None


def get_v6_signal(
    df,
    current_position=0.0,
):

    global _model

    if _model is None:
        _model = PPOV6Inference()

    return _model.predict_from_dataframe(
        df,
        current_position=current_position,
    )