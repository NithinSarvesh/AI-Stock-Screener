from pathlib import Path
import sys

# Ensure the Stock Assistant project root is importable when this
# script is executed directly with: python .\rl\script.py
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from __future__ import annotations


# Project-root import path for direct execution from rl\
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO

from indicators import IndicatorEngine
from rl.champion_v10_features import prepare_features
from rl.champion_v10_env import ChampionV10Env, ACTION_TO_POSITION

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "champion_v10" / "champion_v10.zip"

TICKERS = {
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

def metrics(equity, daily_returns, bh_returns, trades, actions):
    eq = np.asarray(equity, dtype=float)
    r = np.asarray(daily_returns, dtype=float)
    bh = np.asarray(bh_returns, dtype=float)

    total = eq[-1] / eq[0] - 1
    bh_total = np.prod(1 + bh) - 1
    excess = total - bh_total

    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    maxdd = float(dd.min())

    sharpe = 0.0
    sortino = 0.0
    if len(r) > 1 and r.std() > 1e-12:
        sharpe = float(np.sqrt(252) * r.mean() / r.std())
    downside = r[r < 0]
    if len(downside) > 1 and downside.std() > 1e-12:
        sortino = float(np.sqrt(252) * r.mean() / downside.std())

    concentration = max(np.bincount(actions, minlength=5)) / len(actions) if actions else 1.0

    return {
        "strategy_return": total,
        "buy_hold": bh_total,
        "excess": excess,
        "max_drawdown": maxdd,
        "sharpe": sharpe,
        "sortino": sortino,
        "trades": trades,
        "concentration": concentration,
    }

def main():
    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")

    model = PPO.load(MODEL)
    results = []

    print("=" * 80)
    print("CHAMPION V10 UNSEEN TEST")
    print("=" * 80)

    for symbol, ticker in TICKERS.items():
        print(f"\n[{symbol}]")
        df = yf.download(
            ticker, period="5y", interval="1d",
            auto_adjust=False, progress=False, threads=False
        )
        if df.empty:
            print("REJECT: no data")
            continue
        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)

        raw = df.copy()
        features = prepare_features(IndicatorEngine(df).calculate_all())

        # Align raw prices to prepared feature index.
        raw = raw.loc[features.index]

        split = int(len(features) * 0.8)
        test = features.iloc[split:].copy()
        raw_test = raw.iloc[split:].copy()

        if len(test) < 50:
            print("REJECT: insufficient test rows")
            continue

        env = ChampionV10Env(
            test,
            random_start=False,
            episode_length=len(test) - 1,
            min_hold_days=3,
            cooldown_days=2,
            transaction_cost=0.0005,
            turnover_penalty=0.001,
            drawdown_penalty=0.01,
            downside_penalty=0.01,
            signal_threshold=0.08,
        )

        obs, _ = env.reset(seed=123)
        equity = [env.equity]
        daily = []
        bh = []
        actions = []
        trades = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            actions.append(action)

            old_trades = env.trade_count
            obs, reward, term, trunc, info = env.step(action)
            if env.trade_count > old_trades:
                trades += 1

            equity.append(env.equity)
            daily.append(float(info["strategy_return"]))
            bh.append(float(info["market_return"]))
            done = term or trunc

        m = metrics(equity, daily, bh, trades, actions)
        m["symbol"] = symbol
        results.append(m)

        print(f"  Strategy : {m['strategy_return']:+.2%}")
        print(f"  B&H      : {m['buy_hold']:+.2%}")
        print(f"  Excess   : {m['excess']:+.2%}")
        print(f"  Max DD   : {m['max_drawdown']:+.2%}")
        print(f"  Sharpe   : {m['sharpe']:+.2f}")
        print(f"  Sortino  : {m['sortino']:+.2f}")
        print(f"  Trades   : {m['trades']}")
        print(f"  Concentr.: {m['concentration']:.1%}")

    out = pd.DataFrame(results)
    out_dir = MODEL.parent / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "v10_unseen_results.csv", index=False)

    if out.empty:
        raise RuntimeError("No evaluation results")

    avg_excess = out["excess"].mean()
    median_excess = out["excess"].median()
    avg_sharpe = out["sharpe"].mean()
    avg_dd = out["max_drawdown"].mean()
    worst_dd = out["max_drawdown"].min()
    median_trades = out["trades"].median()
    worst_excess = out["excess"].min()
    avg_return = out["strategy_return"].mean()
    avg_bh = out["buy_hold"].mean()

    # Harder gate than V9.x. The model must beat B&H rather than merely make money.
    passed = (
        avg_excess > 0.05
        and median_excess > 0.0
        and avg_sharpe > 0.8
        and avg_dd > -0.25
        and worst_dd > -0.40
        and worst_excess > -0.15
        and median_trades < 100
        and out["concentration"].max() < 0.75
    )

    summary = {
        "stocks": len(out),
        "average_return": avg_return,
        "average_buy_hold": avg_bh,
        "average_excess": avg_excess,
        "median_excess": median_excess,
        "worst_excess": worst_excess,
        "average_sharpe": avg_sharpe,
        "average_max_drawdown": avg_dd,
        "worst_max_drawdown": worst_dd,
        "median_trades": float(median_trades),
        "max_concentration": float(out["concentration"].max()),
        "status": "PASSED" if passed else "FAILED",
    }

    (out_dir / "v10_unseen_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("V10 CHAMPION GATE")
    print("=" * 80)
    for k, v in summary.items():
        if isinstance(v, float):
            if "return" in k or "excess" in k or "drawdown" in k:
                print(f"{k:24s}: {v:+.2%}")
            else:
                print(f"{k:24s}: {v:.3f}")
        else:
            print(f"{k:24s}: {v}")

    print("\nSTATUS:", "CHAMPION PASSED" if passed else "FAILED — DO NOT DEPLOY")

if __name__ == "__main__":
    main()
