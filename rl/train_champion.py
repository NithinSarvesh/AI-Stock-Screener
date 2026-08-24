from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from champion_features import build_features
from multi_stock_champion_env import MultiStockChampionEnv


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

TOTAL_TIMESTEPS = 500_000
EPISODE_LENGTH = 252
OUTPUT = PROJECT_ROOT / "models" / "champion_agent"
OUTPUT.mkdir(parents=True, exist_ok=True)


def download(symbol):
    ticker = STOCKS[symbol]
    print(f"Downloading {ticker}")
    df = yf.download(
        ticker,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        print("  REJECT: empty data")
        return None

    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    try:
        features = build_features(df)
    except Exception as exc:
        print(f"  REJECT: {exc}")
        return None

    print(f"  ACCEPT: {len(features)} usable rows")
    return features


def split_data(df):
    """60/20/20 chronological split. No future information enters training."""
    n = len(df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


def main():
    print("=" * 80)
    print("CHAMPION PPO TRAINING")
    print("=" * 80)
    print(f"Timesteps: {TOTAL_TIMESTEPS:,}")
    print("Split: 60% train / 20% validation / 20% unseen test")
    print("Reward: portfolio log-return + small risk/turnover stabilizers")
    print()

    train_data = {}
    validation_data = {}
    test_data = {}
    metadata = {}

    for i, symbol in enumerate(STOCKS, 1):
        print(f"[{i}/{len(STOCKS)}] {symbol}")
        df = download(symbol)
        if df is None:
            continue

        train, val, test = split_data(df)

        if len(train) < 500 or len(val) < 100 or len(test) < 100:
            print("  REJECT: insufficient split size")
            continue

        train_data[symbol] = train
        validation_data[symbol] = val
        test_data[symbol] = test

        metadata[symbol] = {
            "rows": len(df),
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
        }

    if len(train_data) < 5:
        raise RuntimeError("Fewer than 5 stocks survived preprocessing.")

    with open(OUTPUT / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("USABLE STOCKS:", list(train_data))
    print()

    train_env = DummyVecEnv([
        lambda: MultiStockChampionEnv(
            train_data,
            episode_length=EPISODE_LENGTH,
            random_stock=True,
            random_start=True,
        )
    ])

    validation_env = DummyVecEnv([
        lambda: MultiStockChampionEnv(
            validation_data,
            episode_length=min(EPISODE_LENGTH, min(len(x) - 1 for x in validation_data.values())),
            random_stock=True,
            random_start=True,
        )
    ])

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        ),
        verbose=1,
        device="auto",
        seed=42,
    )

    callback = EvalCallback(
        validation_env,
        best_model_save_path=str(OUTPUT / "best"),
        log_path=str(OUTPUT / "eval_logs"),
        eval_freq=25_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    print()
    print("STARTING TRAINING")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=True,
    )

    model.save(OUTPUT / "champion_final")

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print("Final model :", OUTPUT / "champion_final.zip")
    print("Best model  :", OUTPUT / "best" / "best_model.zip")
    print()
    print("NEXT STEP:")
    print("python .\\rl\\evaluate_champion.py")


if __name__ == "__main__":
    main()
