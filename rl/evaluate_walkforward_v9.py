"""
Universal PPO V9 Walk-Forward Evaluator

Purpose:
- Load the frozen V9 PPO model.
- Evaluate it chronologically on unseen data.
- Evaluate the complete test period, not one random 252-day episode.
- Compare strategy performance with Buy & Hold.
- Record every model action.
- Produce per-stock and aggregate metrics.

IMPORTANT:
This script does NOT train the model.
"""

from __future__ import annotations

import json
import os
import sys
import random

import numpy as np
import pandas as pd
import yfinance as yf

from stable_baselines3 import PPO


# =====================================================================
# PROJECT ROOT
# =====================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =====================================================================
# PROJECT IMPORTS
# =====================================================================

from indicators import IndicatorEngine

from rl.v6_inference import PPOV6Inference

from rl.trading_env_v9 import (
    StockTradingEnvV9,
    ACTION_MAP,
)


# =====================================================================
# CONFIG
# =====================================================================

SEED = 42

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "universal_v9",
    "universal_ppo_v9.zip",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "universal_v9",
    "evaluation",
    "walkforward",
)

# ---------------------------------------------------------------------
# FIRST TEST
#
# We intentionally start with only 5 stocks.
# After this passes, we'll expand the universe.
# ---------------------------------------------------------------------

EVALUATION_STOCKS = [
    "RELIANCE",
    "TCS",
    "INFY",
    "SBIN",
    "HDFCBANK",
]

# ---------------------------------------------------------------------
# Download enough history to create a chronological split.
# ---------------------------------------------------------------------

HISTORY_PERIOD = "5y"

# ---------------------------------------------------------------------
# Chronological split.
#
# Last 20% is completely unseen test data.
# The preceding 20% is validation.
# The first 60% is training-history context.
#
# NOTE:
# The PPO model is already trained.
# We are NOT retraining here.
# ---------------------------------------------------------------------

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

INITIAL_BALANCE = 100_000.0

TRANSACTION_COST = 0.0005

DETERMINISTIC = True


# =====================================================================
# RANDOMNESS
# =====================================================================

random.seed(SEED)
np.random.seed(SEED)


# =====================================================================
# DATA PREPARATION
# =====================================================================

def prepare_stock(
    symbol: str,
):
    """
    Download and prepare stock data using the exact feature
    pipeline used during V9 training.
    """

    ticker = (
        symbol.upper()
        + ".NS"
    )

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

        if (
            df is None
            or df.empty
        ):

            print(
                "  REJECT: empty data"
            )

            return None

        # -------------------------------------------------------------
        # Normalize yfinance MultiIndex.
        # -------------------------------------------------------------

        if (
            hasattr(
                df.columns,
                "nlevels",
            )
            and df.columns.nlevels > 1
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        df = df.dropna(
            how="all"
        )

        if len(df) < 500:

            print(
                f"  REJECT: only "
                f"{len(df)} rows"
            )

            return None

        # -------------------------------------------------------------
        # Same feature pipeline as V9 training.
        # -------------------------------------------------------------

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

        if (
            features is None
            or features.empty
        ):

            print(
                "  REJECT: no usable features"
            )

            return None

        features = (
            features
            .dropna()
            .copy()
        )

        print(
            f"  ACCEPT: "
            f"{len(features)} rows"
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

def split_data(
    df: pd.DataFrame,
):
    """
    Split chronologically.

    Nothing is shuffled.
    """

    n = len(df)

    train_end = int(
        n * TRAIN_RATIO
    )

    validation_end = int(
        n
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    train_df = (
        df.iloc[
            :train_end
        ]
        .copy()
    )

    validation_df = (
        df.iloc[
            train_end:validation_end
        ]
        .copy()
    )

    test_df = (
        df.iloc[
            validation_end:
        ]
        .copy()
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# =====================================================================
# METRICS
# =====================================================================

def max_drawdown(
    equity_curve,
):
    equity = np.asarray(
        equity_curve,
        dtype=float,
    )

    if len(equity) == 0:
        return 0.0

    peak = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity / peak
        - 1.0
    )

    return float(
        drawdown.min()
    )


def sharpe_ratio(
    returns,
):
    returns = np.asarray(
        returns,
        dtype=float,
    )

    if len(returns) < 2:
        return 0.0

    std = returns.std()

    if std <= 1e-12:
        return 0.0

    return float(
        np.sqrt(252.0)
        * returns.mean()
        / std
    )


def sortino_ratio(
    returns,
):
    returns = np.asarray(
        returns,
        dtype=float,
    )

    if len(returns) < 2:
        return 0.0

    downside = returns[
        returns < 0
    ]

    if len(downside) == 0:
        return 0.0

    downside_std = downside.std()

    if downside_std <= 1e-12:
        return 0.0

    return float(
        np.sqrt(252.0)
        * returns.mean()
        / downside_std
    )


# =====================================================================
# TEST-PERIOD EVALUATION
# =====================================================================

def evaluate_test_period(
    model,
    symbol: str,
    test_df: pd.DataFrame,
):
    """
    Evaluate the complete chronological test period.

    No random starting point.
    No training.
    No model updates.
    """

    if len(test_df) < 30:

        raise ValueError(
            "Test period is too short."
        )

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # The environment receives ONLY the test data.
    #
    # The model never sees the earlier training/validation rows
    # during this evaluation episode.
    # -------------------------------------------------------------

    env = StockTradingEnvV9(
        dataframe=test_df,
        initial_balance=INITIAL_BALANCE,
        transaction_cost=TRANSACTION_COST,
        episode_length=len(test_df) - 1,
        random_start=False,
    )

    obs, info = env.reset(
        seed=SEED
    )

    done = False

    equity_curve = [
        INITIAL_BALANCE
    ]

    strategy_returns = []

    market_returns = []

    actions = []

    trades = []

    previous_position = 0.0

    step_number = 0

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=DETERMINISTIC,
        )

        action = int(
            np.asarray(
                action
            )
            .reshape(-1)[0]
        )

        if action not in ACTION_MAP:

            raise RuntimeError(
                f"Invalid action: "
                f"{action}"
            )

        action_name = ACTION_MAP[
            action
        ]["name"]

        current_position = (
            ACTION_MAP[
                action
            ]["position"]
        )

        actions.append(
            action
        )

        if (
            current_position
            != previous_position
        ):

            trades.append(
                {
                    "symbol":
                        symbol,
                    "step":
                        step_number,
                    "action":
                        action,
                    "action_name":
                        action_name,
                    "position":
                        current_position,
                }
            )

        (
            obs,
            reward,
            terminated,
            truncated,
            step_info,
        ) = env.step(
            action
        )

        done = (
            terminated
            or truncated
        )

        step_number += 1

        previous_position = (
            current_position
        )

        # ---------------------------------------------------------
        # Environment diagnostics.
        # ---------------------------------------------------------

        if not isinstance(
            step_info,
            dict,
        ):

            raise RuntimeError(
                "Environment returned "
                "non-dict step info."
            )

        market_return = float(
            step_info.get(
                "market_return",
                0.0,
            )
        )

        strategy_return = float(
            step_info.get(
                "strategy_return",
                0.0,
            )
        )

        equity = float(
            step_info.get(
                "equity",
                equity_curve[-1],
            )
        )

        market_returns.append(
            market_return
        )

        strategy_returns.append(
            strategy_return
        )

        equity_curve.append(
            equity
        )

    # =================================================================
    # PERFORMANCE
    # =================================================================

    final_equity = float(
        equity_curve[-1]
    )

    strategy_return = (
        final_equity
        / INITIAL_BALANCE
        - 1.0
    )

    market_returns_np = np.asarray(
        market_returns,
        dtype=float,
    )

    buy_hold_equity = (
        INITIAL_BALANCE
        * np.prod(
            1.0
            + market_returns_np
        )
    )

    buy_hold_return = (
        buy_hold_equity
        / INITIAL_BALANCE
        - 1.0
    )

    strategy_returns_np = np.asarray(
        strategy_returns,
        dtype=float,
    )

    # =================================================================
    # ACTION DISTRIBUTION
    # =================================================================

    action_counts = {
        action: 0
        for action in ACTION_MAP
    }

    for action in actions:

        action_counts[
            action
        ] += 1

    total_actions = len(
        actions
    )

    action_distribution = {}

    for action, count in (
        action_counts.items()
    ):

        name = ACTION_MAP[
            action
        ]["name"]

        percentage = (
            count
            / total_actions
            * 100.0
            if total_actions > 0
            else 0.0
        )

        action_distribution[
            name
        ] = float(
            percentage
        )

    # =================================================================
    # RESULT
    # =================================================================

    result = {
        "symbol":
            symbol,

        "test_rows":
            int(len(test_df)),

        "evaluation_steps":
            int(len(strategy_returns)),

        "initial_equity":
            float(INITIAL_BALANCE),

        "final_equity":
            final_equity,

        "strategy_return":
            float(strategy_return),

        "buy_hold_return":
            float(buy_hold_return),

        "excess_vs_buy_hold":
            float(
                strategy_return
                - buy_hold_return
            ),

        "max_drawdown":
            float(
                max_drawdown(
                    equity_curve
                )
            ),

        "sharpe":
            float(
                sharpe_ratio(
                    strategy_returns_np
                )
            ),

        "sortino":
            float(
                sortino_ratio(
                    strategy_returns_np
                )
            ),

        "trade_count":
            int(len(trades)),

        "action_distribution":
            action_distribution,
    }

    return (
        result,
        trades,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print()
    print("=" * 80)
    print("UNIVERSAL PPO V9 WALK-FORWARD EVALUATION")
    print("=" * 80)

    print(
        f"Model:\n{MODEL_PATH}"
    )

    print(
        f"Stocks: "
        f"{len(EVALUATION_STOCKS)}"
    )

    print(
        f"History: "
        f"{HISTORY_PERIOD}"
    )

    print(
        f"Split: "
        f"{TRAIN_RATIO:.0%} train / "
        f"{VALIDATION_RATIO:.0%} validation / "
        f"{TEST_RATIO:.0%} test"
    )

    # =================================================================
    # MODEL
    # =================================================================

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )

    print()
    print(
        "Loading V9..."
    )

    model = PPO.load(
        MODEL_PATH
    )

    print(
        "V9 loaded."
    )

    print(
        "Observation:",
        model.observation_space,
    )

    print(
        "Actions:",
        model.action_space,
    )

    # =================================================================
    # OUTPUT
    # =================================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    results = []

    all_trades = []

    # =================================================================
    # STOCK LOOP
    # =================================================================

    for index, symbol in enumerate(
        EVALUATION_STOCKS,
        start=1,
    ):

        print()
        print("=" * 80)

        print(
            f"[{index}/"
            f"{len(EVALUATION_STOCKS)}] "
            f"{symbol}"
        )

        print("=" * 80)

        df = prepare_stock(
            symbol
        )

        if df is None:

            print(
                "SKIPPED"
            )

            continue

        train_df, validation_df, test_df = (
            split_data(df)
        )

        print(
            f"Train rows      : "
            f"{len(train_df)}"
        )

        print(
            f"Validation rows : "
            f"{len(validation_df)}"
        )

        print(
            f"Test rows       : "
            f"{len(test_df)}"
        )

        # -------------------------------------------------------------
        # Show chronological boundaries.
        # -------------------------------------------------------------

        print(
            f"Train range     : "
            f"{train_df.index[0]} "
            f"→ "
            f"{train_df.index[-1]}"
        )

        print(
            f"Validation range: "
            f"{validation_df.index[0]} "
            f"→ "
            f"{validation_df.index[-1]}"
        )

        print(
            f"Test range      : "
            f"{test_df.index[0]} "
            f"→ "
            f"{test_df.index[-1]}"
        )

        try:

            result, trades = (
                evaluate_test_period(
                    model,
                    symbol,
                    test_df,
                )
            )

            results.append(
                result
            )

            all_trades.extend(
                trades
            )

            print()
            print(
                f"Strategy return : "
                f"{result['strategy_return'] * 100:.2f}%"
            )

            print(
                f"Buy & Hold      : "
                f"{result['buy_hold_return'] * 100:.2f}%"
            )

            print(
                f"Excess          : "
                f"{result['excess_vs_buy_hold'] * 100:.2f}%"
            )

            print(
                f"Max drawdown    : "
                f"{result['max_drawdown'] * 100:.2f}%"
            )

            print(
                f"Sharpe          : "
                f"{result['sharpe']:.3f}"
            )

            print(
                f"Sortino         : "
                f"{result['sortino']:.3f}"
            )

            print(
                f"Trades          : "
                f"{result['trade_count']}"
            )

            print(
                "Actions:"
            )

            for name, percentage in (
                result[
                    "action_distribution"
                ].items()
            ):

                print(
                    f"  {name:12s}"
                    f"{percentage:7.2f}%"
                )

        except Exception as exc:

            print(
                f"EVALUATION ERROR: "
                f"{exc}"
            )

    # =================================================================
    # FINAL SUMMARY
    # =================================================================

    if not results:

        raise RuntimeError(
            "No stocks were successfully evaluated."
        )

    results_df = pd.DataFrame(
        results
    )

    summary = {
        "model":
            "universal_ppo_v9",

        "stocks_requested":
            len(EVALUATION_STOCKS),

        "stocks_evaluated":
            len(results),

        "history_period":
            HISTORY_PERIOD,

        "train_ratio":
            TRAIN_RATIO,

        "validation_ratio":
            VALIDATION_RATIO,

        "test_ratio":
            TEST_RATIO,

        "average_strategy_return":
            float(
                results_df[
                    "strategy_return"
                ].mean()
            ),

        "median_strategy_return":
            float(
                results_df[
                    "strategy_return"
                ].median()
            ),

        "average_buy_hold_return":
            float(
                results_df[
                    "buy_hold_return"
                ].mean()
            ),

        "average_excess_vs_buy_hold":
            float(
                results_df[
                    "excess_vs_buy_hold"
                ].mean()
            ),

        "average_max_drawdown":
            float(
                results_df[
                    "max_drawdown"
                ].mean()
            ),

        "worst_max_drawdown":
            float(
                results_df[
                    "max_drawdown"
                ].min()
            ),

        "average_sharpe":
            float(
                results_df[
                    "sharpe"
                ].mean()
            ),

        "median_sharpe":
            float(
                results_df[
                    "sharpe"
                ].median()
            ),

        "average_sortino":
            float(
                results_df[
                    "sortino"
                ].mean()
            ),

        "total_trades":
            int(
                results_df[
                    "trade_count"
                ].sum()
            ),
    }

    # =================================================================
    # ACTION AGGREGATION
    # =================================================================

    action_totals = {
        ACTION_MAP[action]["name"]: 0.0
        for action in ACTION_MAP
    }

    total_actions = 0

    total_actions = 0

    for result in results:

        steps = result[
            "evaluation_steps"
        ]

        total_actions += steps

        for name, percentage in (
            result[
                "action_distribution"
            ].items()
        ):

            action_totals[
                name
            ] += (
                percentage
                / 100.0
                * steps
            )

    summary[
        "action_distribution"
    ] = {}

    for name, count in (
        action_totals.items()
    ):

        summary[
            "action_distribution"
        ][name] = float(
            count
            / total_actions
            * 100.0
            if total_actions > 0
            else 0.0
        )

    # =================================================================
    # SAVE
    # =================================================================

    results_path = os.path.join(
        OUTPUT_DIR,
        "walkforward_stock_results.csv",
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "walkforward_summary.json",
    )

    trades_path = os.path.join(
        OUTPUT_DIR,
        "walkforward_trades.csv",
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    pd.DataFrame(
        all_trades
    ).to_csv(
        trades_path,
        index=False,
    )

    # =================================================================
    # FINAL REPORT
    # =================================================================

    print()
    print("=" * 80)
    print("WALK-FORWARD EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"Stocks evaluated : "
        f"{len(results)}"
    )

    print(
        f"Average return   : "
        f"{summary['average_strategy_return'] * 100:.2f}%"
    )

    print(
        f"Median return    : "
        f"{summary['median_strategy_return'] * 100:.2f}%"
    )

    print(
        f"Average B&H      : "
        f"{summary['average_buy_hold_return'] * 100:.2f}%"
    )

    print(
        f"Average excess   : "
        f"{summary['average_excess_vs_buy_hold'] * 100:.2f}%"
    )

    print(
        f"Average max DD   : "
        f"{summary['average_max_drawdown'] * 100:.2f}%"
    )

    print(
        f"Worst max DD     : "
        f"{summary['worst_max_drawdown'] * 100:.2f}%"
    )

    print(
        f"Average Sharpe   : "
        f"{summary['average_sharpe']:.3f}"
    )

    print(
        f"Average Sortino  : "
        f"{summary['average_sortino']:.3f}"
    )

    print()
    print(
        "OVERALL ACTION DISTRIBUTION"
    )

    for name, percentage in (
        summary[
            "action_distribution"
        ].items()
    ):

        print(
            f"{name:12s}: "
            f"{percentage:7.2f}%"
        )

    print()
    print(
        "Saved:"
    )

    print(
        results_path
    )

    print(
        summary_path
    )

    print(
        trades_path
    )

    print("=" * 80)


if __name__ == "__main__":
    main()