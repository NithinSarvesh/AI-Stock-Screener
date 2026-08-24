"""
Universal PPO V7 Evaluator

Evaluates the trained V7 policy on the dynamic Indian-stock
universe using chronological out-of-sample data.

This is NOT a profitability guarantee.
"""

from __future__ import annotations

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from stable_baselines3 import PPO


# =====================================================================
# PROJECT PATH
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

HISTORY_PERIOD = "max"

MIN_USABLE_ROWS = 900

INITIAL_BALANCE = 100000.0

TRANSACTION_COST = 0.0005

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

MODEL_ROOT = os.path.join(
    PROJECT_ROOT,
    "models",
    "universal_v7",
)

MODEL_PATH = os.path.join(
    MODEL_ROOT,
    "universal_ppo_v7.zip",
)

RESULT_CSV = os.path.join(
    MODEL_ROOT,
    "universal_v7_evaluation.csv",
)

RESULT_JSON = os.path.join(
    MODEL_ROOT,
    "universal_v7_evaluation.json",
)

ACTION_DIR = os.path.join(
    MODEL_ROOT,
    "actions",
)

os.makedirs(
    ACTION_DIR,
    exist_ok=True,
)

warnings.filterwarnings(
    "ignore"
)


# =====================================================================
# DATA PREPARATION
# =====================================================================

def clean(df):

    if df is None or df.empty:
        return None

    df = df.copy()

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):

        df.columns = (
            df.columns
            .get_level_values(0)
        )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(
        c in df.columns
        for c in required
    ):
        return None

    df = df[
        required
    ].copy()

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .sort_index()
    )

    df = df[
        df["Close"] > 0
    ]

    return df


def prepare_stock(symbol):

    ticker = (
        symbol.upper()
        + ".NS"
    )

    print(
        f"Downloading: {ticker}"
    )

    try:

        raw = yf.download(
            ticker,
            period=HISTORY_PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        raw = clean(raw)

        if (
            raw is None
            or len(raw) < MIN_USABLE_ROWS
        ):

            print(
                "  REJECT: insufficient data"
            )

            return None

        features = (
            IndicatorEngine(
                raw.copy()
            )
            .calculate_all()
        )

        features = (
            PPOV6Inference
            .add_context_features(
                features
            )
        )

        if features.empty:

            print(
                "  REJECT: no usable features"
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
# CHRONOLOGICAL SPLIT
# =====================================================================

def split(df):

    train_end = int(
        len(df)
        * TRAIN_RATIO
    )

    validation_end = int(
        len(df)
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    return (
        df.iloc[:train_end],
        df.iloc[
            train_end:validation_end
        ],
        df.iloc[
            validation_end:
        ],
    )


# =====================================================================
# METRICS
# =====================================================================

def sharpe(returns):

    returns = np.asarray(
        returns,
        dtype=float,
    )

    if (
        len(returns) < 2
        or np.std(
            returns,
            ddof=1,
        ) == 0
    ):
        return 0.0

    return float(
        np.sqrt(252)
        * np.mean(returns)
        / np.std(
            returns,
            ddof=1,
        )
    )


def max_drawdown(equity):

    equity = np.asarray(
        equity,
        dtype=float,
    )

    peak = np.maximum.accumulate(
        equity
    )

    return float(
        np.min(
            equity
            / np.maximum(
                peak,
                1e-12,
            )
            - 1
        )
    )


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

    values = [

        0.0,

        np.clip(
            float(row["High"])
            / close
            - 1,
            -5,
            5,
        ),

        np.clip(
            float(row["Low"])
            / close
            - 1,
            -5,
            5,
        ),

        0.0,

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
        ),
    ]

    for column in [
        "EMA20",
        "EMA50",
        "EMA200",
    ]:

        values.append(
            np.clip(
                float(row[column])
                / close
                - 1,
                -5,
                5,
            )
        )

    values.append(
        np.clip(
            (
                float(row["RSI"])
                - 50
            )
            / 50,
            -1,
            1,
        )
    )

    for column in [
        "MACD",
        "MACD_SIGNAL",
        "MACD_HISTOGRAM",
    ]:

        values.append(
            np.clip(
                float(row[column])
                / close,
                -1,
                1,
            )
        )

    for column in [
        "BB_UPPER",
        "BB_MIDDLE",
        "BB_LOWER",
    ]:

        values.append(
            np.clip(
                float(row[column])
                / close
                - 1,
                -5,
                5,
            )
        )

    values.extend([

        np.clip(
            float(row["VWAP"])
            / close
            - 1,
            -5,
            5,
        ),

        0.0,

        np.clip(
            float(row["ATR"])
            / close,
            0,
            1,
        ),

        np.clip(
            float(row["ADX"])
            / 100,
            0,
            1,
        ),

        np.sign(
            float(row["OBV"])
        ),

        np.clip(
            float(row["STOCH_RSI"])
            * 2
            - 1,
            -1,
            1,
        ),

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
            float(row["TREND_SCORE"] * 2 - 1),
            -1,
            1,
        ),

        float(position),
    ])

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
    ).astype(
        np.float32
    )

    if observation.shape != (
        30,
    ):

        raise RuntimeError(
            f"Invalid observation shape: "
            f"{observation.shape}"
        )

    return observation


# =====================================================================
# POLICY EVALUATION
# =====================================================================

def evaluate_policy(
    model,
    df,
):

    position = 0.0

    equity = INITIAL_BALANCE

    equity_curve = [
        equity
    ]

    returns = []

    actions = []

    positions = []

    trades = 0

    transaction_costs = 0.0

    transitions = {}

    for i in range(
        len(df) - 1
    ):

        row = df.iloc[i]

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

        action_map = {
            0: -1.0,
            1: -0.5,
            2: 0.0,
            3: 0.5,
            4: 1.0,
        }

        target = action_map[
            action
        ]

        old_position = position

        change = abs(
            target
            - old_position
        )

        cost = (
            change
            * TRANSACTION_COST
        )

        if change > 0:
            trades += 1

        transition = (
            f"{old_position}"
            f"->{target}"
        )

        transitions[
            transition
        ] = (
            transitions.get(
                transition,
                0,
            )
            + 1
        )

        next_close = float(
            df.iloc[i + 1][
                "Close"
            ]
        )

        current_close = float(
            row["Close"]
        )

        market_return = (
            next_close
            / current_close
            - 1
        )

        net_return = (
            target
            * market_return
            - cost
        )

        net_return = float(
            np.clip(
                net_return,
                -0.99,
                10,
            )
        )

        equity *= (
            1
            + net_return
        )

        transaction_costs += (
            cost
            * equity
        )

        returns.append(
            net_return
        )

        equity_curve.append(
            equity
        )

        actions.append(
            action
        )

        positions.append(
            target
        )

        position = target

    return {

        "initial_equity":
            INITIAL_BALANCE,

        "final_equity":
            equity,

        "return_pct":
            (
                equity
                / INITIAL_BALANCE
                - 1
            )
            * 100,

        "sharpe":
            sharpe(
                returns
            ),

        "max_drawdown_pct":
            max_drawdown(
                equity_curve
            )
            * 100,

        "trades":
            trades,

        "transaction_costs":
            transaction_costs,

        "long_exposure_pct":
            (
                sum(
                    p > 0
                    for p in positions
                )
                / len(positions)
                * 100
            ),

        "short_exposure_pct":
            (
                sum(
                    p < 0
                    for p in positions
                )
                / len(positions)
                * 100
            ),

        "flat_exposure_pct":
            (
                positions.count(0)
                / len(positions)
                * 100
            ),

        "average_position":
            (
                float(
                    np.mean(
                        positions
                    )
                )
                if positions
                else 0.0
            ),

        "transitions":
            transitions,
    }


# =====================================================================
# BENCHMARK
# =====================================================================

def benchmark(
    df,
    mode,
):

    returns = (
        df["Close"]
        .pct_change()
        .fillna(0)
        .to_numpy()[1:]
    )

    if mode == "cash":

        strategy_returns = (
            np.zeros_like(
                returns
            )
        )

    elif mode == "long":

        strategy_returns = returns

    else:

        strategy_returns = -returns

    equity = INITIAL_BALANCE

    equity_curve = [
        equity
    ]

    for value in strategy_returns:

        equity *= (
            1 + value
        )

        equity_curve.append(
            equity
        )

    return {

        "return_pct":
            (
                equity
                / INITIAL_BALANCE
                - 1
            )
            * 100,

        "sharpe":
            sharpe(
                strategy_returns
            ),

        "max_drawdown_pct":
            max_drawdown(
                equity_curve
            )
            * 100,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 80)
    print(
        "UNIVERSAL PPO V7 EVALUATION"
    )
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
        f"Universe candidates: "
        f"{len(tickers)}"
    )

    rows = []

    all_actions = {}

    successful = 0

    for ticker in tickers:

        print()
        print("-" * 80)
        print(
            "EVALUATING:",
            ticker,
        )

        try:

            df = prepare_stock(
                ticker
            )

            if df is None:
                continue

            train, validation, test = (
                split(df)
            )

            for name, part in [
                (
                    "validation",
                    validation,
                ),
                (
                    "test",
                    test,
                ),
            ]:

                if len(part) < 100:

                    continue

                result = evaluate_policy(
                    model,
                    part,
                )

                bh = benchmark(
                    part,
                    "long",
                )

                cash = benchmark(
                    part,
                    "cash",
                )

                short = benchmark(
                    part,
                    "short",
                )

                result.update({

                    "ticker":
                        ticker,

                    "period":
                        name,

                    "rows":
                        len(part),

                    "buy_hold_return_pct":
                        bh[
                            "return_pct"
                        ],

                    "cash_return_pct":
                        cash[
                            "return_pct"
                        ],

                    "short_only_return_pct":
                        short[
                            "return_pct"
                        ],

                    "excess_vs_bh_pct":
                        (
                            result[
                                "return_pct"
                            ]
                            - bh[
                                "return_pct"
                            ]
                        ),
                })

                rows.append(
                    result
                )

                print(
                    f"{name.upper()}: "
                    f"PPO "
                    f"{result['return_pct']:.2f}% | "
                    f"B&H "
                    f"{bh['return_pct']:.2f}% | "
                    f"Sharpe "
                    f"{result['sharpe']:.2f} | "
                    f"DD "
                    f"{result['max_drawdown_pct']:.2f}% | "
                    f"Trades "
                    f"{result['trades']} | "
                    f"L/F/S "
                    f"{result['long_exposure_pct']:.1f}/"
                    f"{result['flat_exposure_pct']:.1f}/"
                    f"{result['short_exposure_pct']:.1f}%"
                )

                if name == "test":

                    all_actions[
                        ticker
                    ] = result[
                        "transitions"
                    ]

            successful += 1

        except Exception as exc:

            print(
                "FAILED:",
                ticker,
                exc,
            )

    if not rows:

        raise RuntimeError(
            "No evaluation results."
        )

    output = pd.DataFrame(
        rows
    )

    output.to_csv(
        RESULT_CSV,
        index=False,
    )

    tests = [
        row
        for row in rows
        if row.get(
            "period"
        ) == "test"
    ]

    validations = [
        row
        for row in rows
        if row.get(
            "period"
        ) == "validation"
    ]

    summary = {

        "model":
            MODEL_PATH,

        "successful_stocks":
            successful,

        "validation":
            validations,

        "test":
            tests,
    }

    if tests:

        summary[
            "test_aggregate"
        ] = {

            "mean_return_pct":
                float(
                    np.mean(
                        [
                            x[
                                "return_pct"
                            ]
                            for x in tests
                        ]
                    )
                ),

            "median_return_pct":
                float(
                    np.median(
                        [
                            x[
                                "return_pct"
                            ]
                            for x in tests
                        ]
                    )
                ),

            "mean_excess_vs_bh_pct":
                float(
                    np.mean(
                        [
                            x[
                                "excess_vs_bh_pct"
                            ]
                            for x in tests
                        ]
                    )
                ),

            "stocks_positive":
                int(
                    sum(
                        x[
                            "return_pct"
                        ] > 0
                        for x in tests
                    )
                ),

            "stocks_beating_bh":
                int(
                    sum(
                        x[
                            "excess_vs_bh_pct"
                        ] > 0
                        for x in tests
                    )
                ),

            "mean_sharpe":
                float(
                    np.mean(
                        [
                            x[
                                "sharpe"
                            ]
                            for x in tests
                        ]
                    )
                ),

            "worst_drawdown_pct":
                float(
                    min(
                        x[
                            "max_drawdown_pct"
                        ]
                        for x in tests
                    )
                ),

            "mean_trades":
                float(
                    np.mean(
                        [
                            x[
                                "trades"
                            ]
                            for x in tests
                        ]
                    )
                ),
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
            default=str,
        )

    print()
    print("=" * 80)
    print(
        "V7 EVALUATION COMPLETE"
    )
    print("=" * 80)

    print(
        "CSV:",
        RESULT_CSV,
    )

    print(
        "JSON:",
        RESULT_JSON,
    )

    if tests:

        aggregate = summary[
            "test_aggregate"
        ]

        print()

        print(
            f"Test mean return: "
            f"{aggregate['mean_return_pct']:.2f}%"
        )

        print(
            f"Mean excess vs B&H: "
            f"{aggregate['mean_excess_vs_bh_pct']:.2f}%"
        )

        print(
            f"Positive: "
            f"{aggregate['stocks_positive']}/"
            f"{len(tests)}"
        )

        print(
            f"Beat B&H: "
            f"{aggregate['stocks_beating_bh']}/"
            f"{len(tests)}"
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


if __name__ == "__main__":
    main()