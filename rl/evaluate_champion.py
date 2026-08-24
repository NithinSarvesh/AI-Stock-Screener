from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import yfinance as yf
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from champion_features import build_features
from champion_env import ACTION_MAP


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

MODEL = PROJECT_ROOT / "models" / "champion_agent" / "best" / "best_model.zip"
OUTPUT = PROJECT_ROOT / "models" / "champion_agent" / "evaluation"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_stock(ticker):
    df = yf.download(
        ticker,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return None

    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    return build_features(df)


def evaluate(df, model):
    n = len(df)
    start = int(n * 0.80)
    test = df.iloc[start:].reset_index(drop=True)

    equity = 1.0
    peak = 1.0
    position = 0.0
    trades = 0
    returns = []
    actions = []

    for i in range(len(test) - 1):
        row = test.iloc[i]

        obs = np.asarray(
            [float(row.get(c, 0.0)) for c in __import__("champion_features").MARKET_FEATURES]
            + [position],
            dtype=np.float32,
        )
        obs = np.clip(np.nan_to_num(obs), -10, 10)

        action, _ = model.predict(obs, deterministic=True)
        action = int(np.asarray(action).item())
        target = ACTION_MAP[action]["position"]

        r = float(test.iloc[i + 1]["Close"]) / max(float(row["Close"]), 1e-8) - 1.0
        turnover = abs(target - position)
        cost = turnover * 0.0005

        portfolio_r = target * r - cost
        equity *= max(1e-8, 1.0 + portfolio_r)

        peak = max(peak, equity)
        returns.append(portfolio_r)
        actions.append(action)

        if turnover > 1e-12:
            trades += 1

        position = target

    if not returns:
        return {}

    arr = np.asarray(returns)
    sharpe = np.mean(arr) / (np.std(arr) + 1e-12) * np.sqrt(252)
    downside = arr[arr < 0]
    sortino = (
        np.mean(arr) / (np.std(downside) + 1e-12) * np.sqrt(252)
        if len(downside)
        else 0.0
    )

    bh = float(test["Close"].iloc[-1]) / float(test["Close"].iloc[0]) - 1.0

    return {
        "strategy_return": equity - 1.0,
        "buy_hold": bh,
        "excess": (equity - 1.0) - bh,
        "max_drawdown": min(
            np.cumprod(1.0 + arr) / np.maximum.accumulate(np.cumprod(1.0 + arr)) - 1.0
        ),
        "sharpe": sharpe,
        "sortino": sortino,
        "trades": trades,
        "action_concentration": max(
            np.mean(np.asarray(actions) == a) for a in range(5)
        ),
    }


def main():
    if not MODEL.exists():
        raise FileNotFoundError(
            f"Champion model not found:\n{MODEL}\n"
            "Train first with train_champion.py"
        )

    print("=" * 80)
    print("CHAMPION UNSEEN TEST")
    print("=" * 80)
    print("Model:", MODEL)
    print("Test: final 20% of each stock's chronological history")
    print()

    model = PPO.load(MODEL)

    rows = []

    for i, (symbol, ticker) in enumerate(STOCKS.items(), 1):
        print(f"[{i}/{len(STOCKS)}] {symbol}")
        try:
            df = load_stock(ticker)
            if df is None:
                print("  REJECT: empty")
                continue

            result = evaluate(df, model)
            result["symbol"] = symbol
            rows.append(result)

            print(f"  Strategy : {result['strategy_return']:+.2%}")
            print(f"  B&H      : {result['buy_hold']:+.2%}")
            print(f"  Excess   : {result['excess']:+.2%}")
            print(f"  Max DD   : {result['max_drawdown']:+.2%}")
            print(f"  Sharpe   : {result['sharpe']:+.2f}")
            print(f"  Trades   : {result['trades']}")
            print(f"  Concentr.: {result['action_concentration']:.1%}")

        except Exception as exc:
            print("  ERROR:", exc)

    if not rows:
        raise RuntimeError("No stocks evaluated.")

    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    summary = {
        "stocks_evaluated": len(rows),
        "average_return": avg("strategy_return"),
        "average_buy_hold": avg("buy_hold"),
        "average_excess": avg("excess"),
        "average_max_drawdown": avg("max_drawdown"),
        "worst_max_drawdown": float(min(r["max_drawdown"] for r in rows)),
        "average_sharpe": avg("sharpe"),
        "average_sortino": avg("sortino"),
        "average_trades": avg("trades"),
        "max_action_concentration": max(r["action_concentration"] for r in rows),
    }

    with open(OUTPUT / "champion_test_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with open(OUTPUT / "champion_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 80)
    print("CHAMPION TEST SUMMARY")
    print("=" * 80)

    for k, v in summary.items():
        if isinstance(v, float):
            if "sharpe" in k or "sortino" in k:
                print(f"{k:28}: {v:.3f}")
            elif "concentration" in k:
                print(f"{k:28}: {v:.2%}")
            else:
                print(f"{k:28}: {v:.2%}")
        else:
            print(f"{k:28}: {v}")

    # Hard gate. A model that fails is NOT called champion.
    passed = (
        summary["average_excess"] > 0.0
        and summary["average_sharpe"] > 0.5
        and summary["worst_max_drawdown"] > -0.30
        and summary["max_action_concentration"] < 0.75
    )

    print()
    if passed:
        print("STATUS: CHAMPION CANDIDATE PASSED")
    else:
        print("STATUS: FAILED CHAMPION GATE")
        print("Do NOT deploy this model to OCI.")

    summary["champion_gate_passed"] = passed

    with open(OUTPUT / "champion_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
