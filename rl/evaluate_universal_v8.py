"""
Universal PPO V8 Evaluation

Evaluates V8 across the same dynamic Indian-stock universe.

Measures:
- PPO return
- Buy & Hold return
- Excess return
- Sharpe
- Maximum drawdown
- Trades
- Action distribution
- Position exposure
"""

from __future__ import annotations

import os
import sys
import json
from collections import Counter

import numpy as np
import pandas as pd
import yfinance as yf

from stable_baselines3 import PPO


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =====================================================================
# IMPORTS
# =====================================================================

from data.universe import get_training_universe
from indicators import IndicatorEngine
from rl.v6_inference import PPOV6Inference


# =====================================================================
# CONFIG
# =====================================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "universal_v8",
    "universal_ppo_v8.zip",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "universal_v8",
)

RESULT_CSV = os.path.join(
    OUTPUT_DIR,
    "universal_v8_evaluation.csv",
)

RESULT_JSON = os.path.join(
    OUTPUT_DIR,
    "universal_v8_evaluation.json",
)

INITIAL_BALANCE = 100000.0

TRANSACTION_COST = 0.0005

HISTORY_PERIOD = "max"

MIN_ROWS = 900


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
# DATA
# =====================================================================

def prepare_stock(symbol):

    ticker = symbol.upper() + ".NS"

    print(
        f"Downloading: {ticker}"
    )

    try:

        df = yf.download(
            ticker,
            period=HISTORY_PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:

            print(
                "  REJECT: empty"
            )

            return None

        if (
            hasattr(df.columns, "nlevels")
            and df.columns.nlevels > 1
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        df = df.dropna(
            how="all"
        )

        if len(df) < MIN_ROWS:

            print(
                f"  REJECT: {len(df)} rows"
            )

            return None

        features = (
            IndicatorEngine(
                df.copy()
            ).calculate_all()
        )

        features = (
            PPOV6Inference
            .add_context_features(
                features
            )
        )

        if features.empty:

            print(
                "  REJECT: no features"
            )

            return None

        print(
            f"  ACCEPT: {len(features)} rows"
        )

        return features

    except Exception as exc:

        print(
            f"  ERROR: {exc}"
        )

        return None


# =====================================================================
# OBSERVATION
# =====================================================================

def build_observation(
    row,
    position,
):

    close = max(
        float(row["Close"]),
        1e-8,
    )

    values = []

    # OHLC
    values.append(0.0)

    values.append(
        np.clip(
            float(row["High"]) / close - 1,
            -5,
            5,
        )
    )

    values.append(
        np.clip(
            float(row["Low"]) / close - 1,
            -5,
            5,
        )
    )

    values.append(0.0)

    # Volume
    values.append(
        np.clip(
            np.log1p(
                max(
                    float(row["Volume"]),
                    1,
                )
                /
                max(
                    float(row["AVG_VOLUME"]),
                    1,
                )
            ),
            -10,
            10,
        )
    )

    # EMA
    for column in [
        "EMA20",
        "EMA50",
        "EMA200",
    ]:

        values.append(
            np.clip(
                float(row[column]) / close - 1,
                -5,
                5,
            )
        )

    # RSI
    values.append(
        np.clip(
            (
                float(row["RSI"]) - 50
            ) / 50,
            -1,
            1,
        )
    )

    # MACD
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

    # Bollinger
    for column in [
        "BB_UPPER",
        "BB_MIDDLE",
        "BB_LOWER",
    ]:

        values.append(
            np.clip(
                float(row[column]) / close - 1,
                -5,
                5,
            )
        )

    # VWAP
    values.append(
        np.clip(
            float(row["VWAP"]) / close - 1,
            -5,
            5,
        )
    )

    # AVG volume placeholder
    values.append(0.0)

    # ATR
    values.append(
        np.clip(
            float(row["ATR"]) / close,
            0,
            1,
        )
    )

    # ADX
    values.append(
        np.clip(
            float(row["ADX"]) / 100,
            0,
            1,
        )
    )

    # OBV
    values.append(
        np.sign(
            float(row["OBV"])
        )
    )

    # Stoch RSI
    values.append(
        np.clip(
            float(row["STOCH_RSI"]) * 2 - 1,
            -1,
            1,
        )
    )

    # Context
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
            float(row["TREND_SCORE"]) * 2 - 1,
            -1,
            1,
        ),
    ])

    # Current position
    values.append(
        float(position)
    )

    observation = np.asarray(
        values,
        dtype=np.float32,
    )

    observation = np.nan_to_num(
        observation,
        nan=0,
        posinf=10,
        neginf=-10,
    )

    observation = np.clip(
        observation,
        -10,
        10,
    )

    if observation.shape != (30,):

        raise RuntimeError(
            f"Invalid observation shape: "
            f"{observation.shape}"
        )

    return observation.astype(
        np.float32
    )


# =====================================================================
# METRICS
# =====================================================================

def calculate_sharpe(returns):

    returns = np.asarray(
        returns,
        dtype=float,
    )

    if len(returns) < 2:

        return 0.0

    std = np.std(
        returns,
        ddof=1,
    )

    if std == 0:

        return 0.0

    return float(
        np.sqrt(252)
        * np.mean(returns)
        / std
    )


def calculate_drawdown(equity):

    equity = np.asarray(
        equity,
        dtype=float,
    )

    peak = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity
        / np.maximum(
            peak,
            1e-8,
        )
        - 1
    )

    return float(
        np.min(drawdown)
        * 100
    )


# =====================================================================
# EVALUATE
# =====================================================================

def evaluate_stock(
    model,
    df,
):

    # Use the final 30% as test data.
    test_start = int(
        len(df) * 0.70
    )

    test = df.iloc[
        test_start:
    ].reset_index(
        drop=True
    )

    if len(test) < 100:

        return None

    equity = INITIAL_BALANCE

    equity_curve = [
        equity
    ]

    returns = []

    position = 0.0

    actions = []

    transitions = Counter()

    trades = 0

    for i in range(
        len(test) - 1
    ):

        row = test.iloc[i]

        observation = build_observation(
            row,
            position,
        )

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        action = int(
            np.asarray(
                action
            ).item()
        )

        if action not in ACTION_MAP:

            raise RuntimeError(
                f"Invalid action: {action}"
            )

        target = ACTION_MAP[
            action
        ]["position"]

        old_position = position

        if target != old_position:

            trades += 1

            transitions[
                f"{old_position}->{target}"
            ] += 1

        current_close = float(
            test.iloc[i]["Close"]
        )

        next_close = float(
            test.iloc[i + 1]["Close"]
        )

        market_return = (
            next_close
            / current_close
            - 1
        )

        turnover = abs(
            target
            - old_position
        )

        cost = (
            turnover
            * TRANSACTION_COST
        )

        strategy_return = (
            target
            * market_return
            - cost
        )

        equity *= (
            1
            + strategy_return
        )

        returns.append(
            strategy_return
        )

        equity_curve.append(
            equity
        )

        actions.append(
            action
        )

        position = target

    # -------------------------------------------------------------
    # Buy and hold
    # -------------------------------------------------------------

    first_price = float(
        test.iloc[0]["Close"]
    )

    last_price = float(
        test.iloc[-1]["Close"]
    )

    buy_hold_return = (
        last_price
        / first_price
        - 1
    )

    strategy_return = (
        equity
        / INITIAL_BALANCE
        - 1
    )

    action_counts = Counter(
        actions
    )

    total_actions = max(
        len(actions),
        1,
    )

    action_distribution = {

        ACTION_MAP[action]["name"]:
            (
                action_counts[action]
                / total_actions
                * 100
            )

        for action in ACTION_MAP
    }

    long_exposure = (
        sum(
            ACTION_MAP[a]["position"] > 0
            for a in actions
        )
        / total_actions
        * 100
    )

    short_exposure = (
        sum(
            ACTION_MAP[a]["position"] < 0
            for a in actions
        )
        / total_actions
        * 100
    )

    flat_exposure = (
        sum(
            ACTION_MAP[a]["position"] == 0
            for a in actions
        )
        / total_actions
        * 100
    )

    return {

        "return_pct":
            strategy_return * 100,

        "buy_hold_return_pct":
            buy_hold_return * 100,

        "excess_vs_bh_pct":
            (
                strategy_return
                - buy_hold_return
            )
            * 100,

        "sharpe":
            calculate_sharpe(
                returns
            ),

        "max_drawdown_pct":
            calculate_drawdown(
                equity_curve
            ),

        "trades":
            trades,

        "long_exposure_pct":
            long_exposure,

        "short_exposure_pct":
            short_exposure,

        "flat_exposure_pct":
            flat_exposure,

        "actions":
            action_distribution,

        "transitions":
            dict(transitions),
    }


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 80)
    print("UNIVERSAL PPO V8 EVALUATION")
    print("=" * 80)

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            MODEL_PATH
        )

    model = PPO.load(
        MODEL_PATH
    )

    tickers = (
        get_training_universe()
    )

    print(
        f"Universe candidates: {len(tickers)}"
    )

    results = []

    for index, symbol in enumerate(
        tickers,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(tickers)}] "
            f"{symbol}"
        )

        df = prepare_stock(
            symbol
        )

        if df is None:

            continue

        try:

            result = evaluate_stock(
                model,
                df,
            )

            if result is None:

                continue

            result[
                "ticker"
            ] = symbol

            results.append(
                result
            )

            actions = result[
                "actions"
            ]

            print(
                f"  PPO: "
                f"{result['return_pct']:.2f}% | "
                f"B&H: "
                f"{result['buy_hold_return_pct']:.2f}% | "
                f"Excess: "
                f"{result['excess_vs_bh_pct']:.2f}%"
            )

            print(
                f"  Sharpe: "
                f"{result['sharpe']:.2f} | "
                f"DD: "
                f"{result['max_drawdown_pct']:.2f}% | "
                f"Trades: "
                f"{result['trades']}"
            )

            print(
                "  Actions: "
                f"SHORT={actions['SHORT']:.1f}% | "
                f"HS={actions['HALF_SHORT']:.1f}% | "
                f"FLAT={actions['FLAT']:.1f}% | "
                f"HL={actions['HALF_LONG']:.1f}% | "
                f"LONG={actions['LONG']:.1f}%"
            )

        except Exception as exc:

            print(
                f"  FAILED: {exc}"
            )

    if not results:

        raise RuntimeError(
            "No evaluation results."
        )

    df_results = pd.DataFrame(
        results
    )

    df_results.to_csv(
        RESULT_CSV,
        index=False,
    )

    # -------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------

    aggregate = {

        "stocks":
            len(results),

        "mean_return_pct":
            float(
                df_results[
                    "return_pct"
                ].mean()
            ),

        "median_return_pct":
            float(
                df_results[
                    "return_pct"
                ].median()
            ),

        "mean_buy_hold_pct":
            float(
                df_results[
                    "buy_hold_return_pct"
                ].mean()
            ),

        "mean_excess_vs_bh_pct":
            float(
                df_results[
                    "excess_vs_bh_pct"
                ].mean()
            ),

        "stocks_positive":
            int(
                (
                    df_results[
                        "return_pct"
                    ] > 0
                ).sum()
            ),

        "stocks_beating_bh":
            int(
                (
                    df_results[
                        "excess_vs_bh_pct"
                    ] > 0
                ).sum()
            ),

        "mean_sharpe":
            float(
                df_results[
                    "sharpe"
                ].mean()
            ),

        "worst_drawdown_pct":
            float(
                df_results[
                    "max_drawdown_pct"
                ].min()
            ),

        "mean_trades":
            float(
                df_results[
                    "trades"
                ].mean()
            ),

        "mean_long_exposure_pct":
            float(
                df_results[
                    "long_exposure_pct"
                ].mean()
            ),

        "mean_short_exposure_pct":
            float(
                df_results[
                    "short_exposure_pct"
                ].mean()
            ),

        "mean_flat_exposure_pct":
            float(
                df_results[
                    "flat_exposure_pct"
                ].mean()
            ),
    }

    # -------------------------------------------------------------
    # Aggregate action usage
    # -------------------------------------------------------------

    action_totals = {
        name: 0.0
        for name in [
            "SHORT",
            "HALF_SHORT",
            "FLAT",
            "HALF_LONG",
            "LONG",
        ]
    }

    for result in results:

        for name, value in result[
            "actions"
        ].items():

            action_totals[name] += value

    for name in action_totals:

        action_totals[name] /= len(
            results
        )

    summary = {

        "model":
            MODEL_PATH,

        "aggregate":
            aggregate,

        "average_action_distribution":
            action_totals,

        "results":
            results,
    }

    with open(
        RESULT_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("=" * 80)
    print("V8 EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"Stocks: "
        f"{aggregate['stocks']}"
    )

    print(
        f"Mean PPO return: "
        f"{aggregate['mean_return_pct']:.2f}%"
    )

    print(
        f"Mean Buy & Hold: "
        f"{aggregate['mean_buy_hold_pct']:.2f}%"
    )

    print(
        f"Mean excess vs B&H: "
        f"{aggregate['mean_excess_vs_bh_pct']:.2f}%"
    )

    print(
        f"Positive stocks: "
        f"{aggregate['stocks_positive']}/"
        f"{aggregate['stocks']}"
    )

    print(
        f"Beat B&H: "
        f"{aggregate['stocks_beating_bh']}/"
        f"{aggregate['stocks']}"
    )

    print(
        f"Mean Sharpe: "
        f"{aggregate['mean_sharpe']:.2f}"
    )

    print(
        f"Worst DD: "
        f"{aggregate['worst_drawdown_pct']:.2f}%"
    )

    print(
        f"Mean trades: "
        f"{aggregate['mean_trades']:.1f}"
    )

    print()
    print("AVERAGE ACTION DISTRIBUTION")

    for name, value in action_totals.items():

        print(
            f"{name:12s}: "
            f"{value:.2f}%"
        )

    print()
    print(
        "CSV:",
        RESULT_CSV,
    )

    print(
        "JSON:",
        RESULT_JSON,
    )


if __name__ == "__main__":

    main()