from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from indicators import IndicatorEngine
from rl.v6_inference import PPOV6Inference
from rl.trading_env_v9_2 import StockTradingEnvV92


# =============================================================================
# CONFIG
# =============================================================================

MODEL_PATH = Path(
    "models/universal_v9_2/universal_ppo_v9_2.zip"
)

OUTPUT_DIR = Path(
    "models/universal_v9_2/evaluation"
)

EVALUATION_PERIOD = "2y"

STOCKS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "AXISBANK": "AXISBANK.NS",
}

INITIAL_BALANCE = 100000.0

TRANSACTION_COST = 0.0005

EPISODE_LENGTH = 252


# =============================================================================
# DATA
# =============================================================================

def download_stock(symbol: str) -> pd.DataFrame | None:
    ticker = STOCKS[symbol]

    print(f"Downloading: {ticker}")

    try:
        df = yf.download(
            ticker,
            period=EVALUATION_PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        print(f"  REJECT: download failed: {exc}")
        return None

    if df is None or df.empty:
        print("  REJECT: empty data")
        return None

    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        print(f"  REJECT: missing columns: {missing}")
        return None

    try:
        df = IndicatorEngine(df).calculate_all()
        df = PPOV6Inference.add_context_features(df)
    except Exception as exc:
        print(f"  REJECT: indicator preparation failed: {exc}")
        return None

    df = df.dropna().copy()

    if len(df) < 100:
        print(f"  REJECT: insufficient rows: {len(df)}")
        return None

    print(f"  ACCEPT: {len(df)} rows")

    return df


# =============================================================================
# METRICS
# =============================================================================

def calculate_max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0

    running_max = np.maximum.accumulate(equity)

    drawdown = (
        equity / np.maximum(running_max, 1e-12)
    ) - 1.0

    return float(np.min(drawdown))


def calculate_sharpe(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=np.float64)

    if len(returns) < 2:
        return 0.0

    std = np.std(returns, ddof=1)

    if std <= 1e-12:
        return 0.0

    return float(
        np.mean(returns)
        / std
        * np.sqrt(252)
    )


def calculate_sortino(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=np.float64)

    if len(returns) < 2:
        return 0.0

    downside = returns[returns < 0]

    if len(downside) == 0:
        return 0.0

    downside_std = np.std(
        downside,
        ddof=1,
    )

    if downside_std <= 1e-12:
        return 0.0

    return float(
        np.mean(returns)
        / downside_std
        * np.sqrt(252)
    )


# =============================================================================
# SINGLE STOCK EVALUATION
# =============================================================================

def evaluate_stock(
    model: PPO,
    symbol: str,
    df: pd.DataFrame,
):
    print()
    print("=" * 80)
    print(f"EVALUATING {symbol}")
    print("=" * 80)

    env = StockTradingEnvV92(
        df,
        initial_balance=INITIAL_BALANCE,
        transaction_cost=TRANSACTION_COST,
        episode_length=min(
            EPISODE_LENGTH,
            len(df) - 1,
        ),
        random_start=False,
    )

    observation, info = env.reset(seed=42)

    equity = [INITIAL_BALANCE]
    daily_returns = []

    actions = []
    trades = 0

    previous_position = 0.0

    prices = []

    terminated = False
    truncated = False

    while not (terminated or truncated):

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        action = int(action)

        (
            observation,
            reward,
            terminated,
            truncated,
            step_info,
        ) = env.step(action)

        actions.append(action)

        position = float(
            step_info.get(
                "position",
                previous_position,
            )
        )

        if abs(position - previous_position) > 1e-9:
            trades += 1

        previous_position = position

        portfolio_value = float(
            step_info.get(
                "portfolio_value",
                INITIAL_BALANCE,
            )
        )

        equity.append(portfolio_value)

        market_return = float(
            step_info.get(
                "market_return",
                0.0,
            )
        )

        daily_returns.append(
            float(
                step_info.get(
                    "strategy_return",
                    position * market_return,
                )
            )
        )

        try:
            price = float(
                step_info.get(
                    "price",
                    np.nan,
                )
            )
        except Exception:
            price = np.nan

        prices.append(price)

    equity = np.asarray(
        equity,
        dtype=np.float64,
    )

    daily_returns = np.asarray(
        daily_returns,
        dtype=np.float64,
    )

    strategy_return = (
        equity[-1]
        / INITIAL_BALANCE
        - 1.0
    )

    # Buy & hold based on actual close prices.
    close_prices = df["Close"].astype(float).values

    if len(close_prices) >= 2:
        buy_hold_return = (
            close_prices[-1]
            / close_prices[0]
            - 1.0
        )
    else:
        buy_hold_return = 0.0

    max_drawdown = calculate_max_drawdown(
        equity
    )

    sharpe = calculate_sharpe(
        daily_returns
    )

    sortino = calculate_sortino(
        daily_returns
    )

    excess = (
        strategy_return
        - buy_hold_return
    )

    action_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }

    for action in actions:
        if action in action_counts:
            action_counts[action] += 1

    total_actions = max(
        len(actions),
        1,
    )

    action_distribution = {
        action: count / total_actions
        for action, count in action_counts.items()
    }

    print()
    print(
        f"Return: {strategy_return * 100:.2f}%"
    )

    print(
        f"Buy & Hold: {buy_hold_return * 100:.2f}%"
    )

    print(
        f"Excess: {excess * 100:.2f}%"
    )

    print(
        f"Max DD: {max_drawdown * 100:.2f}%"
    )

    print(
        f"Sharpe: {sharpe:.3f}"
    )

    print(
        f"Sortino: {sortino:.3f}"
    )

    print(
        f"Trades: {trades}"
    )

    return {
        "symbol": symbol,
        "strategy_return": strategy_return,
        "buy_hold_return": buy_hold_return,
        "excess_vs_bh": excess,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "trades": trades,
        "actions": action_distribution,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)
    print("UNIVERSAL PPO V9.2 EVALUATION")
    print("=" * 80)

    print(f"Model:")
    print(
        Path(MODEL_PATH).resolve()
    )

    print(
        f"Evaluation period: {EVALUATION_PERIOD}"
    )

    print(
        f"Stocks: {len(STOCKS)}"
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Loading V9.2 model...")

    model = PPO.load(
        str(MODEL_PATH)
    )

    print("V9.2 model loaded.")

    print(
        "Observation:",
        model.observation_space,
    )

    print(
        "Actions:",
        model.action_space,
    )

    results = []

    total_action_counts = {
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
        print(
            f"[{index}/{len(STOCKS)}] "
            f"EVALUATING {symbol}"
        )

        df = download_stock(symbol)

        if df is None:
            continue

        try:

            result = evaluate_stock(
                model,
                symbol,
                df,
            )

            results.append(result)

            for action, fraction in result[
                "actions"
            ].items():

                total_action_counts[
                    action
                ] += fraction

        except Exception as exc:

            print(
                f"  ERROR evaluating "
                f"{symbol}: {exc}"
            )

    if not results:
        raise RuntimeError(
            "No stocks were successfully evaluated."
        )

    results_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key != "actions"
            }
            for result in results
        ]
    )

    # Normalize aggregate action distribution.
    action_distribution = (
        np.asarray(
            [
                total_action_counts[i]
                for i in range(5)
            ],
            dtype=np.float64,
        )
        / len(results)
    )

    # =============================================================================
    # SUMMARY
    # =============================================================================

    print()
    print("=" * 80)
    print("V9.2 EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"Stocks evaluated: {len(results)}"
    )

    average_return = float(
        results_df[
            "strategy_return"
        ].mean()
    )

    median_return = float(
        results_df[
            "strategy_return"
        ].median()
    )

    average_bh = float(
        results_df[
            "buy_hold_return"
        ].mean()
    )

    average_excess = float(
        results_df[
            "excess_vs_bh"
        ].mean()
    )

    average_dd = float(
        results_df[
            "max_drawdown"
        ].mean()
    )

    worst_dd = float(
        results_df[
            "max_drawdown"
        ].min()
    )

    average_sharpe = float(
        results_df[
            "sharpe"
        ].mean()
    )

    average_sortino = float(
        results_df[
            "sortino"
        ].mean()
    )

    print(
        f"Average return: "
        f"{average_return * 100:.2f}%"
    )

    print(
        f"Median return: "
        f"{median_return * 100:.2f}%"
    )

    print(
        f"Average Buy & Hold: "
        f"{average_bh * 100:.2f}%"
    )

    print(
        f"Average excess vs B&H: "
        f"{average_excess * 100:.2f}%"
    )

    print(
        f"Average max drawdown: "
        f"{average_dd * 100:.2f}%"
    )

    print(
        f"Worst max drawdown: "
        f"{worst_dd * 100:.2f}%"
    )

    print(
        f"Average Sharpe: "
        f"{average_sharpe:.3f}"
    )

    print(
        f"Average Sortino: "
        f"{average_sortino:.3f}"
    )

    # =============================================================================
    # ACTION DISTRIBUTION
    # =============================================================================

    action_names = {
        0: "SHORT",
        1: "HALF_SHORT",
        2: "FLAT",
        3: "HALF_LONG",
        4: "LONG",
    }

    print()
    print("ACTION DISTRIBUTION")

    for action in range(5):

        print(
            f"{action_names[action]:12s}: "
            f"{action_distribution[action] * 100:7.2f}%"
        )

    # =============================================================================
    # SAVE RESULTS
    # =============================================================================

    results_path = (
        OUTPUT_DIR
        / "v9_2_stock_results.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "v9_2_summary.json"
    )

    actions_path = (
        OUTPUT_DIR
        / "v9_2_action_distribution.json"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    summary = {
        "model": str(
            MODEL_PATH.resolve()
        ),
        "evaluation_period": EVALUATION_PERIOD,
        "stocks_evaluated": len(results),
        "average_return": average_return,
        "median_return": median_return,
        "average_buy_hold": average_bh,
        "average_excess_vs_bh": average_excess,
        "average_max_drawdown": average_dd,
        "worst_max_drawdown": worst_dd,
        "average_sharpe": average_sharpe,
        "average_sortino": average_sortino,
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    action_summary = {
        action_names[i]: float(
            action_distribution[i]
        )
        for i in range(5)
    }

    with open(
        actions_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            action_summary,
            file,
            indent=2,
        )

    print()
    print("Saved:")
    print(
        results_path.resolve()
    )

    print(
        summary_path.resolve()
    )

    print(
        actions_path.resolve()
    )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()