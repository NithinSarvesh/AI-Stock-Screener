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

import yfinance as yf
from stable_baselines3 import PPO

from indicators import IndicatorEngine
from rl.champion_v10_features import prepare_features
from rl.multi_stock_champion_v10 import MultiStockChampionV10

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "champion_v10"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

TOTAL_TIMESTEPS = 500_000

def download_features(symbol, ticker):
    print(f"Downloading {ticker}")
    df = yf.download(
        ticker, period="max", interval="1d",
        auto_adjust=False, progress=False, threads=False
    )
    if df.empty:
        raise ValueError("empty data")
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    df = IndicatorEngine(df).calculate_all()
    return prepare_features(df)

def main():
    data = {}
    metadata = {}

    print("=" * 80)
    print("CHAMPION V10 TRAINING")
    print("=" * 80)

    for symbol, ticker in TICKERS.items():
        try:
            f = download_features(symbol, ticker)
            if len(f) < 700:
                print(f"{symbol}: REJECT {len(f)} rows")
                continue

            # Preserve chronological data. The environment randomizes starts
            # inside this history; no future test data is used during training.
            data[symbol] = f
            metadata[symbol] = {
                "rows": len(f),
                "start": str(f.index[0]),
                "end": str(f.index[-1]),
            }
            print(f"{symbol}: ACCEPT {len(f)} rows")
        except Exception as exc:
            print(f"{symbol}: REJECT {exc}")

    if len(data) < 5:
        raise RuntimeError("Fewer than 5 usable stocks")

    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    env = MultiStockChampionV10(
        data,
        initial_balance=100000,
        transaction_cost=0.0005,
        episode_length=252,
        random_start=True,
        min_hold_days=3,
        cooldown_days=2,
        turnover_penalty=0.001,
        drawdown_penalty=0.01,
        downside_penalty=0.01,
        signal_threshold=0.08,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        verbose=1,
        device="auto",
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    model.save(MODEL_DIR / "champion_v10")
    print(f"\nSaved: {MODEL_DIR / 'champion_v10.zip'}")

if __name__ == "__main__":
    main()
