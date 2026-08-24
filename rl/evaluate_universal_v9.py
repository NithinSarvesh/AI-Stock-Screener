"""
Universal PPO V9 Evaluator

Purpose:
- Load trained PPO V9
- Evaluate on unseen chronological data
- Compare agent performance with Buy & Hold
- Track actions
- Calculate risk/performance metrics
- Produce per-stock results
- Save evaluation artifacts

IMPORTANT:
Training data and evaluation data are separated chronologically.

The model is NOT retrained during evaluation.
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
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


# =====================================================================
# PROJECT IMPORTS
# =====================================================================

from indicators import IndicatorEngine

from rl.v6_inference import (
    PPOV6Inference,
)

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
)

# -------------------------------------------------------------
# Evaluation period
#
# We deliberately use recent data separately from training.
# -------------------------------------------------------------

EVALUATION_PERIOD = "2y"

EPISODE_LENGTH = 252

INITIAL_BALANCE = 100_000.0

TRANSACTION_COST = 0.0005

RISK_FREE_RATE = 0.0

# Keep deterministic inference.
DETERMINISTIC = True


# =====================================================================
# STOCKS
# =====================================================================

EVALUATION_STOCKS = [
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "ITC",
    "LT",
    "BHARTIARTL",
    "AXISBANK",
]


# =====================================================================
# HELPERS
# =====================================================================

def prepare_stock(
    symbol: str,
):
    """
    Download and prepare evaluation data.
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
            period=EVALUATION_PERIOD,
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
        # Normalize yfinance MultiIndex
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

        if len(df) < 300:

            print(
                f"  REJECT: only "
                f"{len(df)} rows"
            )

            return None

        # -------------------------------------------------------------
        # Same feature pipeline as training
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
# METRICS
# =====================================================================

def calculate_max_drawdown(
    equity_curve,
):
    """
    Maximum percentage drawdown.
    """

    equity = np.asarray(
        equity_curve,
        dtype=float,
    )

    if len(equity) == 0:
        return 0.0

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    return float(
        drawdown.min()
    )


def calculate_sharpe(
    returns,
):
    """
    Daily Sharpe ratio.
    """

    returns = np.asarray(
        returns,
        dtype=float,
    )

    if len(returns) < 2:
        return 0.0

    std = returns.std()

    if std <= 1e-12:
        return 0.0

    excess = (
        returns
        - RISK_FREE_RATE / 252.0
    )

    return float(
        np.sqrt(252.0)
        * excess.mean()
        / std
    )


def calculate_sortino(
    returns,
):
    """
    Daily Sortino ratio.
    """

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


def calculate_annualized_return(
    initial,
    final,
    days,
):
    """
    Annualized return using trading days.
    """

    if (
        initial <= 0
        or final <= 0
        or days <= 0
    ):
        return 0.0

    years = days / 252.0

    if years <= 0:
        return 0.0

    return float(
        (final / initial)
        ** (1.0 / years)
        - 1.0
    )


# =====================================================================
# SINGLE STOCK EVALUATION
# =====================================================================

def evaluate_stock(
    model,
    symbol,
    df,
):
    """
    Run the trained agent through the evaluation period.
    """

    # -------------------------------------------------------------
    # IMPORTANT:
    # No random starting point.
    #
    # We want the entire chronological evaluation period.
    # -------------------------------------------------------------

    env = StockTradingEnvV9(
        dataframe=df,
        initial_balance=INITIAL_BALANCE,
        transaction_cost=TRANSACTION_COST,
        episode_length=min(
            EPISODE_LENGTH,
            len(df) - 1,
        ),
        random_start=False,
    )

    obs, info = env.reset(
        seed=SEED
    )

    action_counts = {
        action: 0
        for action in ACTION_MAP
    }

    equity_curve = [
        INITIAL_BALANCE
    ]

    strategy_returns = []

    market_returns = []

    trades = []

    done = False

    previous_position = 0.0

    step_number = 0

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=DETERMINISTIC,
        )

        # SB3 returns a numpy scalar/array.
        action = int(
            np.asarray(action).reshape(-1)[0]
        )

        if action not in ACTION_MAP:

            raise RuntimeError(
                f"Invalid action: {action}"
            )

        action_counts[
            action
        ] += 1

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

        current_position = (
            ACTION_MAP[action][
                "position"
            ]
        )

        if (
            current_position
            != previous_position
        ):

            trades.append(
                {
                    "symbol": symbol,
                    "step": step_number,
                    "action": action,
                    "action_name":
                        ACTION_MAP[action][
                            "name"
                        ],
                    "position":
                        current_position,
                }
            )

        previous_position = (
            current_position
        )

        # ---------------------------------------------------------
        # Extract information generated by the environment.
        # ---------------------------------------------------------

        if isinstance(
            step_info,
            dict,
        ):

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

        else:

            market_return = 0.0

            strategy_return = float(
                reward
            )

            equity = (
                equity_curve[-1]
                * (
                    1.0
                    + strategy_return
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

    # -------------------------------------------------------------
    # Final metrics
    # -------------------------------------------------------------

    final_equity = float(
        equity_curve[-1]
    )

    total_return = (
        final_equity
        / INITIAL_BALANCE
        - 1.0
    )

    buy_hold_equity = (
        INITIAL_BALANCE
        * np.prod(
            1.0
            + np.asarray(
                market_returns,
                dtype=float,
            )
        )
    )

    buy_hold_return = (
        buy_hold_equity
        / INITIAL_BALANCE
        - 1.0
    )

    daily_returns = np.asarray(
        strategy_returns,
        dtype=float,
    )

    annualized_return = (
        calculate_annualized_return(
            INITIAL_BALANCE,
            final_equity,
            len(daily_returns),
        )
    )

    max_drawdown = (
        calculate_max_drawdown(
            equity_curve
        )
    )

    sharpe = calculate_sharpe(
        daily_returns
    )

    sortino = calculate_sortino(
        daily_returns
    )

    action_total = sum(
        action_counts.values()
    )

    action_distribution = {}

    for action, count in (
        action_counts.items()
    ):

        name = ACTION_MAP[
            action
        ]["name"]

        if action_total > 0:

            percentage = (
                count
                / action_total
                * 100.0
            )

        else:

            percentage = 0.0

        action_distribution[
            name
        ] = {
            "count": int(count),
            "percentage":
                float(percentage),
        }

    result = {
        "symbol": symbol,

        "evaluation_steps":
            int(len(daily_returns)),

        "initial_equity":
            float(INITIAL_BALANCE),

        "final_equity":
            final_equity,

        "total_return":
            float(total_return),

        "annualized_return":
            annualized_return,

        "buy_hold_return":
            float(buy_hold_return),

        "excess_vs_buy_hold":
            float(
                total_return
                - buy_hold_return
            ),

        "max_drawdown":
            float(max_drawdown),

        "sharpe":
            float(sharpe),

        "sortino":
            float(sortino),

        "number_of_trades":
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

    random.seed(SEED)

    np.random.seed(
        SEED
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("UNIVERSAL PPO V9 EVALUATION")
    print("=" * 80)

    print(
        f"Model:\n{MODEL_PATH}"
    )

    print(
        f"Evaluation period: "
        f"{EVALUATION_PERIOD}"
    )

    print(
        f"Stocks: "
        f"{len(EVALUATION_STOCKS)}"
    )

    # -------------------------------------------------------------
    # Verify model exists
    # -------------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )

    # -------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------

    print()
    print(
        "Loading V9 model..."
    )

    model = PPO.load(
        MODEL_PATH
    )

    print(
        "V9 model loaded."
    )

    print(
        "Observation:",
        model.observation_space,
    )

    print(
        "Actions:",
        model.action_space,
    )

    # -------------------------------------------------------------
    # Evaluate stocks
    # -------------------------------------------------------------

    results = []

    all_trades = []

    for index, symbol in enumerate(
        EVALUATION_STOCKS,
        start=1,
    ):

        print()
        print(
            "=" * 80
        )

        print(
            f"[{index}/"
            f"{len(EVALUATION_STOCKS)}] "
            f"EVALUATING {symbol}"
        )

        print(
            "=" * 80
        )

        df = prepare_stock(
            symbol
        )

        if (
            df is None
            or df.empty
        ):

            print(
                "SKIPPED"
            )

            continue

        try:

            result, trades = (
                evaluate_stock(
                    model,
                    symbol,
                    df,
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
                f"Return: "
                f"{result['total_return'] * 100:.2f}%"
            )

            print(
                f"Buy & Hold: "
                f"{result['buy_hold_return'] * 100:.2f}%"
            )

            print(
                f"Max DD: "
                f"{result['max_drawdown'] * 100:.2f}%"
            )

            print(
                f"Sharpe: "
                f"{result['sharpe']:.3f}"
            )

            print(
                f"Sortino: "
                f"{result['sortino']:.3f}"
            )

            print(
                f"Trades: "
                f"{result['number_of_trades']}"
            )

        except Exception as exc:

            print(
                f"EVALUATION ERROR: "
                f"{exc}"
            )

    if not results:

        raise RuntimeError(
            "No stocks were successfully evaluated."
        )

    # =================================================================
    # SUMMARY
    # =================================================================

    results_df = pd.DataFrame(
        results
    )

    summary = {
        "model":
            "universal_ppo_v9",

        "evaluation_period":
            EVALUATION_PERIOD,

        "stocks_requested":
            len(EVALUATION_STOCKS),

        "stocks_evaluated":
            len(results),

        "average_return":
            float(
                results_df[
                    "total_return"
                ].mean()
            ),

        "median_return":
            float(
                results_df[
                    "total_return"
                ].median()
            ),

        "average_buy_hold":
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
                    "number_of_trades"
                ].sum()
            ),
    }

    # =================================================================
    # ACTION SUMMARY
    # =================================================================

    action_summary = {}

    for action in ACTION_MAP:

        name = ACTION_MAP[
            action
        ]["name"]

        count = 0

        for result in results:

            count += result[
                "action_distribution"
            ].get(
                name,
                {}
            ).get(
                "count",
                0,
            )

        action_summary[
            name
        ] = count

    summary[
        "action_counts"
    ] = action_summary

    total_actions = sum(
        action_summary.values()
    )

    summary[
        "action_distribution"
    ] = {}

    for name, count in (
        action_summary.items()
    ):

        summary[
            "action_distribution"
        ][name] = (
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
        "v9_stock_results.csv",
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "v9_summary.json",
    )

    trades_path = os.path.join(
        OUTPUT_DIR,
        "v9_trades.csv",
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
    print("V9 EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"Stocks evaluated: "
        f"{len(results)}"
    )

    print(
        f"Average return: "
        f"{summary['average_return'] * 100:.2f}%"
    )

    print(
        f"Median return: "
        f"{summary['median_return'] * 100:.2f}%"
    )

    print(
        f"Average Buy & Hold: "
        f"{summary['average_buy_hold'] * 100:.2f}%"
    )

    print(
        f"Average excess vs B&H: "
        f"{summary['average_excess_vs_buy_hold'] * 100:.2f}%"
    )

    print(
        f"Average max drawdown: "
        f"{summary['average_max_drawdown'] * 100:.2f}%"
    )

    print(
        f"Worst max drawdown: "
        f"{summary['worst_max_drawdown'] * 100:.2f}%"
    )

    print(
        f"Average Sharpe: "
        f"{summary['average_sharpe']:.3f}"
    )

    print(
        f"Average Sortino: "
        f"{summary['average_sortino']:.3f}"
    )

    print()
    print(
        "ACTION DISTRIBUTION"
    )

    for name, percentage in (
        summary[
            "action_distribution"
        ].items()
    ):

        print(
            f"{name:12s}: "
            f"{percentage:6.2f}%"
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

    print()
    print("=" * 80)


if __name__ == "__main__":

    main()