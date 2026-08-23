"""
Hybrid PPO V6 backtest
======================

Run from the Stock Assistant project root:

    python rl/backtest_hybrid_v6.py

What it compares:
    1. Buy & Hold
    2. Existing technical signal
    3. Hybrid: Quant + Technical + PPO V6

Important:
    - A decision at bar t only uses history through bar t.
    - The resulting position is applied to t -> t+1.
    - Transaction cost defaults to 0.05% per unit of position turnover.
    - PPO receives the actual current position, not a forced 0 every day.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from stock_fetcher import StockFetcher
from indicators import IndicatorEngine
from support_resistance import SupportResistance
from candlestick import CandlePattern
from core.signal_engine import SignalEngine
from rl.v6_inference import get_v6_signal
from rl.decision_engine import HybridDecisionEngine


# ============================================================
# CONFIG
# ============================================================

TICKERS = [
    "RELIANCE",
    "TCS",
    "INFY",
    "ICICIBANK",
    "SBIN",
    "LT",
    "ITC",
]

INITIAL_CAPITAL = 100_000.0
TRANSACTION_COST = 0.0005       # 0.05%
MIN_HISTORY = 200               # enough for EMA200/context features
ANNUALIZATION = 252


# ============================================================
# HELPERS
# ============================================================

def signal_to_position(signal: str) -> int:
    """Convert the Stock Assistant signal vocabulary to target position."""
    value = str(signal or "").strip().upper()

    if value in {"STRONG BUY", "BUY"}:
        return 1

    if value in {"STRONG SELL", "SELL"}:
        return -1

    return 0


def safe_float(value, default=0.0) -> float:
    try:
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def sharpe_ratio(returns: pd.Series) -> float:
    returns = returns.dropna().astype(float)

    if len(returns) < 2:
        return 0.0

    std = returns.std(ddof=1)

    if std <= 1e-12:
        return 0.0

    return float(
        returns.mean() / std * math.sqrt(ANNUALIZATION)
    )


def profit_factor(returns: pd.Series) -> float:
    returns = returns.astype(float)

    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()

    if losses <= 1e-12:
        return float("inf") if gains > 0 else 0.0

    return float(gains / losses)


# ============================================================
# DATA
# ============================================================

def load_history(ticker: str) -> tuple[pd.DataFrame, str]:
    stock = StockFetcher(ticker)

    data = stock.fetch_all()

    history = data["history"].copy()

    history = IndicatorEngine(
        history
    ).calculate_all()

    history = (
        history
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["Close"])
        .sort_index()
    )

    if len(history) < MIN_HISTORY:
        raise RuntimeError(
            f"{ticker}: only {len(history)} rows available; "
            f"need at least {MIN_HISTORY}."
        )

    return history, stock.symbol


# ============================================================
# ONE STRATEGY BACKTEST
# ============================================================

def run_backtest(
    ticker: str,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Walk forward through history.

    At t:
        - build every signal using history[:t+1]
        - choose target positions
        - apply position from t -> t+1

    This prevents tomorrow's close from influencing today's decision.
    """

    rows: List[dict] = []

    technical_position = 0
    hybrid_position = 0

    # Skip early rows until the indicators/context are sufficiently mature.
    start = MIN_HISTORY - 1

    for i in range(start, len(history) - 1):
        prefix = history.iloc[: i + 1].copy()

        latest = prefix.iloc[-1]

        # --------------------------------------------------------
        # Existing technical engine
        # --------------------------------------------------------

        levels = SupportResistance(prefix).calculate()

        pattern = CandlePattern(prefix).detect()

        analysis = SignalEngine(
            latest=latest,
            levels=levels,
            pattern=pattern,
        ).analyze()

        technical_signal = str(
            analysis.get("signal", "WAIT")
        )

        quant_score = safe_float(
            analysis.get("score", 0.0)
        )

        technical_target = signal_to_position(
            technical_signal
        )

        # --------------------------------------------------------
        # PPO V6
        # --------------------------------------------------------

        rl_result = get_v6_signal(
            prefix,
            current_position=float(hybrid_position),
        )

        # --------------------------------------------------------
        # Hybrid decision
        # --------------------------------------------------------

        hybrid_result = HybridDecisionEngine().decide(
            quantitative_score=quant_score,
            technical_signal=technical_signal,
            rl_result=rl_result,
        )

        hybrid_signal = str(
            hybrid_result.final_signal
        )

        hybrid_target = signal_to_position(
            hybrid_signal
        )

        # --------------------------------------------------------
        # Next-bar market return
        # --------------------------------------------------------

        current_close = safe_float(
            history.iloc[i]["Close"]
        )

        next_close = safe_float(
            history.iloc[i + 1]["Close"]
        )

        if current_close <= 0 or next_close <= 0:
            continue

        market_return = (
            next_close / current_close
        ) - 1.0

        # --------------------------------------------------------
        # Turnover / costs
        # --------------------------------------------------------

        technical_turnover = abs(
            technical_target - technical_position
        )

        hybrid_turnover = abs(
            hybrid_target - hybrid_position
        )

        technical_cost = (
            technical_turnover * TRANSACTION_COST
        )

        hybrid_cost = (
            hybrid_turnover * TRANSACTION_COST
        )

        # Position is chosen at t and held over t -> t+1.
        technical_return = (
            technical_target * market_return
            - technical_cost
        )

        hybrid_return = (
            hybrid_target * market_return
            - hybrid_cost
        )

        # Update state for next decision.
        technical_position = technical_target
        hybrid_position = hybrid_target

        rows.append(
            {
                "date": history.index[i],
                "next_date": history.index[i + 1],
                "close": current_close,
                "next_close": next_close,
                "market_return": market_return,

                "quant_score": quant_score,
                "technical_signal": technical_signal,
                "technical_position": technical_target,

                "ppo_action": (
                    rl_result.get("name", "UNKNOWN")
                    if isinstance(rl_result, dict)
                    else "UNKNOWN"
                ),
                "ppo_position": (
                    safe_float(
                        rl_result.get("position", 0.0)
                    )
                    if isinstance(rl_result, dict)
                    else 0.0
                ),

                "hybrid_signal": hybrid_signal,
                "hybrid_score": safe_float(
                    hybrid_result.final_score
                ),
                "hybrid_position": hybrid_target,

                "technical_turnover": technical_turnover,
                "hybrid_turnover": hybrid_turnover,

                "technical_return": technical_return,
                "hybrid_return": hybrid_return,

                "technical_cost": technical_cost,
                "hybrid_cost": hybrid_cost,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# METRICS
# ============================================================

def strategy_metrics(
    returns: pd.Series,
    initial_capital: float,
) -> Dict[str, float]:

    returns = (
        returns
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )

    equity = (
        initial_capital
        * (1.0 + returns).cumprod()
    )

    total_return = (
        equity.iloc[-1] / initial_capital
    ) - 1.0

    years = max(
        len(returns) / ANNUALIZATION,
        1.0 / ANNUALIZATION,
    )

    cagr = (
        (equity.iloc[-1] / initial_capital)
        ** (1.0 / years)
        - 1.0
    )

    winning = returns[returns > 0]

    return {
        "final_equity": float(equity.iloc[-1]),
        "return": float(total_return),
        "cagr": float(cagr),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "win_rate": (
            float(len(winning) / len(returns))
            if len(returns)
            else 0.0
        ),
        "profit_factor": profit_factor(returns),
        "observations": int(len(returns)),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("HYBRID PPO V6 BACKTEST")
    print("=" * 100)
    print()
    print(f"Initial capital : ₹{INITIAL_CAPITAL:,.2f}")
    print(f"Transaction cost: {TRANSACTION_COST * 100:.3f}%")
    print()
    print("Strategies:")
    print("  A = Buy & Hold")
    print("  B = Existing Technical Signal")
    print("  C = Quant + Technical + PPO V6")
    print()
    print("IMPORTANT: decisions use data available only through each decision bar.")
    print()

    all_results = []
    all_traces = []

    output_dir = Path("models") / "hybrid_v6_backtest"
    output_dir.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        print("-" * 100)
        print(f"BACKTESTING {ticker}")

        try:
            history, resolved_symbol = load_history(ticker)

            print(
                f"Resolved: {resolved_symbol} | "
                f"Rows: {len(history):,}"
            )

            trace = run_backtest(
                ticker,
                history,
            )

            if trace.empty:
                raise RuntimeError(
                    "No usable backtest rows."
                )

            # Buy & hold: simply hold +1 throughout.
            buy_hold_returns = trace[
                "market_return"
            ]

            technical_returns = trace[
                "technical_return"
            ]

            hybrid_returns = trace[
                "hybrid_return"
            ]

            bh = strategy_metrics(
                buy_hold_returns,
                INITIAL_CAPITAL,
            )

            technical = strategy_metrics(
                technical_returns,
                INITIAL_CAPITAL,
            )

            hybrid = strategy_metrics(
                hybrid_returns,
                INITIAL_CAPITAL,
            )

            technical_trades = int(
                trace["technical_turnover"].gt(0).sum()
            )

            hybrid_trades = int(
                trace["hybrid_turnover"].gt(0).sum()
            )

            result_rows = [
                {
                    "stock": ticker,
                    "strategy": "BUY_HOLD",
                    **bh,
                    "trades": 0,
                    "long_exposure": 1.0,
                    "short_exposure": 0.0,
                },
                {
                    "stock": ticker,
                    "strategy": "TECHNICAL",
                    **technical,
                    "trades": technical_trades,
                    "long_exposure": float(
                        (trace["technical_position"] == 1).mean()
                    ),
                    "short_exposure": float(
                        (trace["technical_position"] == -1).mean()
                    ),
                },
                {
                    "stock": ticker,
                    "strategy": "HYBRID_V6",
                    **hybrid,
                    "trades": hybrid_trades,
                    "long_exposure": float(
                        (trace["hybrid_position"] == 1).mean()
                    ),
                    "short_exposure": float(
                        (trace["hybrid_position"] == -1).mean()
                    ),
                },
            ]

            all_results.extend(result_rows)

            trace = trace.copy()
            trace.insert(0, "ticker", ticker)

            trace_path = (
                output_dir
                / f"{ticker.lower()}_trace.csv"
            )

            trace.to_csv(
                trace_path,
                index=False,
            )

            all_traces.append(trace)

            print()
            print(
                f"{'Strategy':<15}"
                f"{'Return':>12}"
                f"{'Sharpe':>10}"
                f"{'Max DD':>12}"
                f"{'Trades':>10}"
            )

            for name, metrics, trades in [
                ("BUY_HOLD", bh, 0),
                ("TECHNICAL", technical, technical_trades),
                ("HYBRID_V6", hybrid, hybrid_trades),
            ]:
                print(
                    f"{name:<15}"
                    f"{metrics['return'] * 100:>11.2f}%"
                    f"{metrics['sharpe']:>10.3f}"
                    f"{metrics['max_drawdown'] * 100:>11.2f}%"
                    f"{trades:>10}"
                )

            print()

        except Exception as exc:
            print(
                f"ERROR {ticker}: {exc}"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    if not all_results:
        raise RuntimeError(
            "No stocks completed successfully."
        )

    results = pd.DataFrame(all_results)

    results_path = (
        output_dir
        / "hybrid_v6_results.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    # --------------------------------------------------------
    # Aggregate comparison
    # --------------------------------------------------------

    aggregate = (
        results
        .groupby("strategy")
        .agg(
            stocks=("stock", "nunique"),
            mean_return=("return", "mean"),
            median_return=("return", "median"),
            mean_sharpe=("sharpe", "mean"),
            mean_max_drawdown=("max_drawdown", "mean"),
            mean_trades=("trades", "mean"),
        )
        .reset_index()
    )

    aggregate_path = (
        output_dir
        / "hybrid_v6_aggregate.csv"
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    # --------------------------------------------------------
    # Excess return table
    # --------------------------------------------------------

    pivot = (
        results
        .pivot(
            index="stock",
            columns="strategy",
            values="return",
        )
    )

    if (
        "BUY_HOLD" in pivot.columns
        and "TECHNICAL" in pivot.columns
        and "HYBRID_V6" in pivot.columns
    ):
        comparison = pd.DataFrame(
            {
                "technical_excess_vs_bh":
                    pivot["TECHNICAL"]
                    - pivot["BUY_HOLD"],

                "hybrid_excess_vs_bh":
                    pivot["HYBRID_V6"]
                    - pivot["BUY_HOLD"],

                "hybrid_minus_technical":
                    pivot["HYBRID_V6"]
                    - pivot["TECHNICAL"],
            }
        )

        comparison_path = (
            output_dir
            / "hybrid_v6_comparison.csv"
        )

        comparison.to_csv(
            comparison_path
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("AGGREGATE RESULTS")
    print("=" * 100)

    print(
        aggregate.to_string(
            index=False,
            formatters={
                "mean_return":
                    lambda x: f"{x * 100:.2f}%",
                "median_return":
                    lambda x: f"{x * 100:.2f}%",
                "mean_sharpe":
                    lambda x: f"{x:.3f}",
                "mean_max_drawdown":
                    lambda x: f"{x * 100:.2f}%",
                "mean_trades":
                    lambda x: f"{x:.1f}",
            },
        )
    )

    print()
    print("=" * 100)
    print("FILES")
    print("=" * 100)
    print(f"Results    : {results_path}")
    print(f"Aggregate  : {aggregate_path}")
    print(f"Traces dir : {output_dir}")
    print()
    print("Next decision should be based on HYBRID_V6 vs TECHNICAL,")
    print("not on whether PPO looks impressive by itself.")
    print("=" * 100)


if __name__ == "__main__":
    main()
