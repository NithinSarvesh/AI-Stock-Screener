from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

from stable_baselines3 import PPO


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# IMPORTS
# ============================================================================

from indicators import IndicatorEngine
from rl.v6_inference import PPOV6Inference
from rl.trading_env_v9_1 import StockTradingEnvV91


# ============================================================================
# CONFIG
# ============================================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "universal_v9_1",
    "universal_ppo_v9_1.zip",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "universal_v9_1",
    "evaluation",
)

EVALUATION_PERIOD = "2y"

STOCKS = [
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

INITIAL_BALANCE = 100000.0

TRANSACTION_COST = 0.0005

EPISODE_LENGTH = 252


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_stock(symbol: str):

    ticker = symbol + ".NS"

    print(
        f"Downloading: {ticker}"
    )

    df = yf.download(
        ticker,
        period=EVALUATION_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:

        print(
            "  REJECT: empty data"
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

    if len(df) < 100:

        print(
            f"  REJECT: only {len(df)} rows"
        )

        return None

    df = (
        IndicatorEngine(
            df.copy()
        )
        .calculate_all()
    )

    df = (
        PPOV6Inference
        .add_context_features(
            df
        )
    )

    if df.empty:

        print(
            "  REJECT: no usable features"
        )

        return None

    print(
        f"  ACCEPT: {len(df)} rows"
    )

    return df


# ============================================================================
# METRICS
# ============================================================================

def calculate_max_drawdown(equity):

    equity = np.asarray(
        equity,
        dtype=float
    )

    peaks = np.maximum.accumulate(
        equity
    )

    drawdowns = (
        equity / peaks
    ) - 1.0

    return float(
        drawdowns.min()
    )


def calculate_sharpe(returns):

    returns = np.asarray(
        returns,
        dtype=float
    )

    if len(returns) < 2:
        return 0.0

    std = returns.std(
        ddof=1
    )

    if std == 0:
        return 0.0

    return float(
        returns.mean()
        / std
        * np.sqrt(252)
    )


def calculate_sortino(returns):

    returns = np.asarray(
        returns,
        dtype=float
    )

    if len(returns) < 2:
        return 0.0

    downside = returns[
        returns < 0
    ]

    if len(downside) == 0:
        return 0.0

    downside_std = (
        downside.std(
            ddof=1
        )
    )

    if downside_std == 0:
        return 0.0

    return float(
        returns.mean()
        / downside_std
        * np.sqrt(252)
    )


# ============================================================================
# EVALUATE ONE STOCK
# ============================================================================

def evaluate_stock(
    model,
    symbol,
    df,
):

    env = StockTradingEnvV91(
        df,
        initial_balance=INITIAL_BALANCE,
        transaction_cost=TRANSACTION_COST,
        episode_length=EPISODE_LENGTH,
        random_start=False,
    )

    obs, info = env.reset(
        seed=42
    )

    done = False

    equity_curve = [
        env.balance
    ]

    daily_returns = []

    actions = []

    trades = 0

    previous_position = 0.0

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action = int(action)

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

        actions.append(
            action
        )

        current_position = (
            step_info.get(
                "position",
                previous_position,
            )
        )

        if (
            current_position
            != previous_position
        ):
            trades += 1

        previous_position = (
            current_position
        )

        equity = (
            step_info.get(
                "portfolio_value",
                env.balance,
            )
        )

        equity_curve.append(
            float(equity)
        )

        if len(equity_curve) >= 2:

            previous_equity = (
                equity_curve[-2]
            )

            if previous_equity != 0:

                daily_returns.append(
                    (
                        equity
                        / previous_equity
                    )
                    - 1.0
                )

    strategy_return = (
        equity_curve[-1]
        / equity_curve[0]
    ) - 1.0

    prices = df[
        "Close"
    ].values.astype(float)

    if len(prices) >= 2:

        buy_hold = (
            prices[-1]
            / prices[0]
        ) - 1.0

    else:

        buy_hold = 0.0

    excess = (
        strategy_return
        - buy_hold
    )

    max_dd = calculate_max_drawdown(
        equity_curve
    )

    sharpe = calculate_sharpe(
        daily_returns
    )

    sortino = calculate_sortino(
        daily_returns
    )

    action_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }

    for action in actions:

        action_counts[
            action
        ] += 1

    total_actions = max(
        len(actions),
        1,
    )

    action_distribution = {
        0: action_counts[0]
        / total_actions,

        1: action_counts[1]
        / total_actions,

        2: action_counts[2]
        / total_actions,

        3: action_counts[3]
        / total_actions,

        4: action_counts[4]
        / total_actions,
    }

    return {
        "symbol": symbol,
        "return": strategy_return,
        "buy_hold": buy_hold,
        "excess": excess,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "trades": trades,
        "action_distribution":
            action_distribution,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("UNIVERSAL PPO V9.1 EVALUATION")
    print("=" * 80)

    print(
        f"Model:\n{MODEL_PATH}"
    )

    print(
        f"Evaluation period: "
        f"{EVALUATION_PERIOD}"
    )

    print(
        f"Stocks: {len(STOCKS)}"
    )

    print()
    print("Loading V9.1 model...")

    model = PPO.load(
        MODEL_PATH
    )

    print(
        "V9.1 model loaded."
    )

    print(
        "Observation:",
        model.observation_space,
    )

    print(
        "Actions:",
        model.action_space,
    )

    results = []

    action_totals = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }

    for index, symbol in enumerate(
        STOCKS,
        start=1,
    ):

        print()
        print("=" * 80)
        print(
            f"[{index}/{len(STOCKS)}] "
            f"EVALUATING {symbol}"
        )
        print("=" * 80)

        df = prepare_stock(
            symbol
        )

        if df is None:
            continue

        result = evaluate_stock(
            model,
            symbol,
            df,
        )

        results.append(
            result
        )

        for action, count in (
            result[
                "action_distribution"
            ].items()
        ):

            action_totals[
                action
            ] += count

        print()
        print(
            f"Return: "
            f"{result['return'] * 100:.2f}%"
        )

        print(
            f"Buy & Hold: "
            f"{result['buy_hold'] * 100:.2f}%"
        )

        print(
            f"Excess: "
            f"{result['excess'] * 100:.2f}%"
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
            f"{result['trades']}"
        )

    if not results:

        raise RuntimeError(
            "No stocks were successfully evaluated."
        )

    results_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 80)
    print("V9.1 EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"Stocks evaluated: "
        f"{len(results_df)}"
    )

    print(
        f"Average return: "
        f"{results_df['return'].mean() * 100:.2f}%"
    )

    print(
        f"Median return: "
        f"{results_df['return'].median() * 100:.2f}%"
    )

    print(
        f"Average Buy & Hold: "
        f"{results_df['buy_hold'].mean() * 100:.2f}%"
    )

    print(
        f"Average excess vs B&H: "
        f"{results_df['excess'].mean() * 100:.2f}%"
    )

    print(
        f"Average max drawdown: "
        f"{results_df['max_drawdown'].mean() * 100:.2f}%"
    )

    print(
        f"Worst max drawdown: "
        f"{results_df['max_drawdown'].min() * 100:.2f}%"
    )

    print(
        f"Average Sharpe: "
        f"{results_df['sharpe'].mean():.3f}"
    )

    print(
        f"Average Sortino: "
        f"{results_df['sortino'].mean():.3f}"
    )

    print()
    print("ACTION DISTRIBUTION")

    action_names = {
        0: "SHORT",
        1: "HALF_SHORT",
        2: "FLAT",
        3: "HALF_LONG",
        4: "LONG",
    }

    total_action_weight = sum(
        action_totals.values()
    )

    for action in range(5):

        percentage = (
            action_totals[action]
            / total_action_weight
            * 100
        )

        print(
            f"{action_names[action]:<12}: "
            f"{percentage:7.2f}%"
        )

    # ------------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------------

    stock_results_path = os.path.join(
        OUTPUT_DIR,
        "v9_1_stock_results.csv",
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "v9_1_summary.json",
    )

    results_df.drop(
        columns=[
            "action_distribution"
        ]
    ).to_csv(
        stock_results_path,
        index=False,
    )

    summary = {

        "version": "V9.1",

        "evaluation_period":
            EVALUATION_PERIOD,

        "stocks_evaluated":
            len(results_df),

        "average_return":
            float(
                results_df[
                    "return"
                ].mean()
            ),

        "median_return":
            float(
                results_df[
                    "return"
                ].median()
            ),

        "average_buy_hold":
            float(
                results_df[
                    "buy_hold"
                ].mean()
            ),

        "average_excess":
            float(
                results_df[
                    "excess"
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

        "average_sortino":
            float(
                results_df[
                    "sortino"
                ].mean()
            ),

        "action_distribution": {
            action_names[action]:
                float(
                    action_totals[action]
                    / total_action_weight
                )
            for action in range(5)
        },
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("Saved:")
    print(
        stock_results_path
    )
    print(
        summary_path
    )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()