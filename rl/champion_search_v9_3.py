from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from indicators import IndicatorEngine

from rl.v6_inference import (
    PPOV6Inference
)

from rl.trading_env_v9_2 import (
    StockTradingEnvV92
)

from rl.multi_stock_env_v9_2 import (
    MultiStockEnvV92
)


# ============================================================
# CONFIGURATION
# ============================================================

TIMESTEPS = 100_000

STOCKS = [
    "RELIANCE",
    "TCS",
    "INFY",
    "SBIN",
    "HDFCBANK",
]

DATA_PERIOD = "5y"

EPISODE_LENGTH = 252

INITIAL_BALANCE = 100000.0

TRANSACTION_COST = 0.0005


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "champion_search_v9_3"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CANDIDATES
# ============================================================

CANDIDATES = {

    "A_baseline": {

        "drawdown_penalty": 0.05,

        "downside_penalty": 0.02,

        "directional_weight": 0.003,

        "turnover_penalty": 0.005,
    },

    "B_risk_control": {

        "drawdown_penalty": 0.10,

        "downside_penalty": 0.05,

        "directional_weight": 0.003,

        "turnover_penalty": 0.005,
    },

    "C_low_directional": {

        "drawdown_penalty": 0.05,

        "downside_penalty": 0.02,

        "directional_weight": 0.001,

        "turnover_penalty": 0.005,
    },

    "D_high_directional": {

        "drawdown_penalty": 0.05,

        "downside_penalty": 0.02,

        "directional_weight": 0.005,

        "turnover_penalty": 0.005,
    },

    "E_low_turnover": {

        "drawdown_penalty": 0.05,

        "downside_penalty": 0.02,

        "directional_weight": 0.003,

        "turnover_penalty": 0.001,
    },
}


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_stock(
    symbol: str
):

    ticker = (
        f"{symbol}.NS"
    )

    print(
        f"Downloading {ticker}"
    )

    df = yf.download(
        ticker,
        period=DATA_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:

        raise ValueError(
            f"No data returned for {ticker}"
        )

    if (
        hasattr(
            df.columns,
            "nlevels"
        )
        and df.columns.nlevels > 1
    ):

        df.columns = (
            df.columns
            .get_level_values(0)
        )

    # Indicators
    df = (
        IndicatorEngine(df)
        .calculate_all()
    )

    # Context features used by V9.x
    df = (
        PPOV6Inference
        .add_context_features(df)
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna().copy()

    if len(df) < 500:

        raise ValueError(
            f"{symbol}: only "
            f"{len(df)} usable rows"
        )

    print(
        f"  ACCEPT: {len(df)} rows"
    )

    return df


# ============================================================
# BUILD TRAINING UNIVERSE
# ============================================================

def build_universe():

    universe = {}

    print()
    print("=" * 80)
    print("BUILDING V9.3 CHAMPION SEARCH UNIVERSE")
    print("=" * 80)

    for index, symbol in enumerate(
        STOCKS,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(STOCKS)}] "
            f"{symbol}"
        )

        try:

            universe[symbol] = (
                download_stock(symbol)
            )

        except Exception as exc:

            print(
                f"  REJECT: {exc}"
            )

    if not universe:

        raise RuntimeError(
            "No usable stocks."
        )

    print()
    print("=" * 80)
    print(
        f"USABLE STOCKS: "
        f"{len(universe)}"
    )
    print("=" * 80)

    for symbol, df in (
        universe.items()
    ):

        print(
            f"{symbol:12s}"
            f"{len(df):6d} rows"
        )

    return universe


# ============================================================
# ENV FACTORY
# ============================================================

def make_env(
    universe,
    params
):

    def factory():

        return MultiStockEnvV92(

            universe,

            initial_balance=(
                INITIAL_BALANCE
            ),

            transaction_cost=(
                TRANSACTION_COST
            ),

            episode_length=(
                EPISODE_LENGTH
            ),

            random_start=True,

            drawdown_penalty=(
                params[
                    "drawdown_penalty"
                ]
            ),

            downside_penalty=(
                params[
                    "downside_penalty"
                ]
            ),

            directional_weight=(
                params[
                    "directional_weight"
                ]
            ),

            turnover_penalty=(
                params[
                    "turnover_penalty"
                ]
            ),
        )

    return factory


# ============================================================
# TRAIN CANDIDATE
# ============================================================

def train_candidate(
    name,
    universe,
    params
):

    print()
    print("=" * 80)
    print(
        f"TRAINING CANDIDATE: {name}"
    )
    print("=" * 80)

    print()

    print(
        json.dumps(
            params,
            indent=2
        )
    )

    env = DummyVecEnv([
        make_env(
            universe,
            params
        )
    ])

    print()
    print(
        "Observation:",
        env.observation_space
    )

    print(
        "Actions:",
        env.action_space
    )

    model = PPO(

        "MlpPolicy",

        env,

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

        verbose=1,

        device="cpu",
    )

    start_time = time.time()

    print()
    print(
        f"Starting "
        f"{TIMESTEPS:,} timesteps..."
    )

    model.learn(
        total_timesteps=TIMESTEPS,
        progress_bar=True,
    )

    elapsed = (
        time.time()
        - start_time
    )

    model_path = (
        MODEL_DIR
        / f"{name}.zip"
    )

    model.save(
        model_path
    )

    env.close()

    print()
    print(
        f"{name} TRAINING COMPLETE"
    )

    print(
        f"Time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"Saved: {model_path}"
    )

    return (
        model_path,
        elapsed
    )


# ============================================================
# SINGLE-STOCK EVALUATION
# ============================================================

def evaluate_candidate(
    model_path,
    universe,
    episodes_per_stock=3
):

    model = PPO.load(
        model_path,
        device="cpu"
    )

    results = []

    for symbol, df in (
        universe.items()
    ):

        env = StockTradingEnvV92(

            df,

            initial_balance=(
                INITIAL_BALANCE
            ),

            transaction_cost=(
                TRANSACTION_COST
            ),

            episode_length=(
                EPISODE_LENGTH
            ),

            random_start=False,

        )

        for episode in range(
            episodes_per_stock
        ):

            obs, reset_info = (
                env.reset(
                    seed=1000 + episode
                )
            )

            starting_equity = float(
                reset_info.get(
                    "equity",
                    INITIAL_BALANCE
                )
            )

            actions = []

            equity_curve = [
                starting_equity
            ]

            terminated = False
            truncated = False

            while not (
                terminated
                or truncated
            ):

                action, _ = (
                    model.predict(
                        obs,
                        deterministic=True
                    )
                )

                action = int(
                    action
                )

                actions.append(
                    action
                )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    action
                )

                current_equity = float(
                    info.get(
                        "equity",
                        equity_curve[-1]
                    )
                )

                equity_curve.append(
                    current_equity
                )

            ending_equity = (
                equity_curve[-1]
            )

            total_return = (
                ending_equity
                / starting_equity
                - 1.0
            )

            curve = np.asarray(
                equity_curve,
                dtype=float
            )

            running_max = (
                np.maximum.accumulate(
                    curve
                )
            )

            drawdowns = (
                curve
                / running_max
                - 1.0
            )

            max_drawdown = float(
                drawdowns.min()
            )

            # Daily equity returns
            if len(curve) > 1:

                daily_returns = (
                    np.diff(curve)
                    / curve[:-1]
                )

                daily_returns = (
                    daily_returns[
                        np.isfinite(
                            daily_returns
                        )
                    ]
                )

            else:

                daily_returns = (
                    np.array([])
                )

            if (
                len(daily_returns) > 1
                and daily_returns.std() > 0
            ):

                sharpe = (
                    daily_returns.mean()
                    / daily_returns.std()
                    * np.sqrt(252)
                )

            else:

                sharpe = 0.0

            trade_count = int(
                info.get(
                    "trade_count",
                    0
                )
            )

            action_counts = (
                np.bincount(
                    actions,
                    minlength=5
                )
            )

            results.append({

                "symbol": symbol,

                "episode": episode,

                "return": total_return,

                "max_drawdown": max_drawdown,

                "sharpe": sharpe,

                "trades": trade_count,

                "short": int(
                    action_counts[0]
                ),

                "half_short": int(
                    action_counts[1]
                ),

                "flat": int(
                    action_counts[2]
                ),

                "half_long": int(
                    action_counts[3]
                ),

                "long": int(
                    action_counts[4]
                ),
            })

        env.close()

    return pd.DataFrame(
        results
    )


# ============================================================
# SCORE CANDIDATE
# ============================================================

def score_candidate(
    result
):

    avg_return = float(
        result[
            "return"
        ].mean()
    )

    median_return = float(
        result[
            "return"
        ].median()
    )

    avg_drawdown = float(
        result[
            "max_drawdown"
        ].mean()
    )

    worst_drawdown = float(
        result[
            "max_drawdown"
        ].min()
    )

    avg_sharpe = float(
        result[
            "sharpe"
        ].mean()
    )

    avg_trades = float(
        result[
            "trades"
        ].mean()
    )

    action_columns = [
        "short",
        "half_short",
        "flat",
        "half_long",
        "long",
    ]

    action_totals = (
        result[
            action_columns
        ].sum()
    )

    total_actions = float(
        action_totals.sum()
    )

    if total_actions > 0:

        action_pct = (
            action_totals
            / total_actions
        )

    else:

        action_pct = pd.Series(
            0.0,
            index=action_columns
        )

    max_action_pct = float(
        action_pct.max()
    )

    # --------------------------------------------------------
    # Collapse penalty
    # --------------------------------------------------------

    collapse_penalty = max(
        0.0,
        max_action_pct - 0.70
    )

    # --------------------------------------------------------
    # Champion score
    #
    # Return       -> primary
    # Median       -> robustness
    # Sharpe       -> risk-adjusted quality
    # Drawdown     -> risk penalty
    # Collapse     -> punish single-action policies
    # --------------------------------------------------------

    score = (

        avg_return * 100.0

        + median_return * 50.0

        + avg_sharpe * 2.0

        + avg_drawdown * 25.0

        - collapse_penalty * 20.0

        + min(
            avg_trades / 100.0,
            1.0
        )
    )

    return {

        "score": score,

        "avg_return": avg_return,

        "median_return": median_return,

        "avg_drawdown": avg_drawdown,

        "worst_drawdown": worst_drawdown,

        "avg_sharpe": avg_sharpe,

        "avg_trades": avg_trades,

        "max_action_pct": max_action_pct,

        "short_pct": float(
            action_pct[
                "short"
            ]
        ),

        "half_short_pct": float(
            action_pct[
                "half_short"
            ]
        ),

        "flat_pct": float(
            action_pct[
                "flat"
            ]
        ),

        "half_long_pct": float(
            action_pct[
                "half_long"
            ]
        ),

        "long_pct": float(
            action_pct[
                "long"
            ]
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("V9.3 CHAMPION SEARCH")
    print("=" * 80)

    print(
        f"Candidates: "
        f"{len(CANDIDATES)}"
    )

    print(
        f"Timesteps/candidate: "
        f"{TIMESTEPS:,}"
    )

    print(
        f"Total planned timesteps: "
        f"{TIMESTEPS * len(CANDIDATES):,}"
    )

    print(
        f"Stocks requested: "
        f"{len(STOCKS)}"
    )

    universe = (
        build_universe()
    )

    metadata = {

        "timesteps_per_candidate":
            TIMESTEPS,

        "total_timesteps":
            TIMESTEPS
            * len(CANDIDATES),

        "stocks":
            list(
                universe.keys()
            ),

        "candidates":
            CANDIDATES,
    }

    with open(
        MODEL_DIR
        / "search_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    rankings = []

    # ========================================================
    # CANDIDATE LOOP
    # ========================================================

    for name, params in (
        CANDIDATES.items()
    ):

        try:

            model_path, elapsed = (
                train_candidate(
                    name,
                    universe,
                    params
                )
            )

            print()
            print(
                "=" * 80
            )

            print(
                f"EVALUATING {name}"
            )

            print(
                "=" * 80
            )

            result = (
                evaluate_candidate(
                    model_path,
                    universe
                )
            )

            result_path = (
                MODEL_DIR
                / f"{name}_episodes.csv"
            )

            result.to_csv(
                result_path,
                index=False
            )

            metrics = (
                score_candidate(
                    result
                )
            )

            metrics[
                "candidate"
            ] = name

            metrics[
                "training_minutes"
            ] = (
                elapsed / 60.0
            )

            rankings.append(
                metrics
            )

            print()
            print(
                f"{name} RESULT"
            )

            print(
                f"Score: "
                f"{metrics['score']:.4f}"
            )

            print(
                f"Average return: "
                f"{metrics['avg_return'] * 100:.2f}%"
            )

            print(
                f"Median return: "
                f"{metrics['median_return'] * 100:.2f}%"
            )

            print(
                f"Average Sharpe: "
                f"{metrics['avg_sharpe']:.3f}"
            )

            print(
                f"Average max DD: "
                f"{metrics['avg_drawdown'] * 100:.2f}%"
            )

            print(
                f"Worst max DD: "
                f"{metrics['worst_drawdown'] * 100:.2f}%"
            )

            print(
                f"Average trades: "
                f"{metrics['avg_trades']:.1f}"
            )

            print(
                f"Max action concentration: "
                f"{metrics['max_action_pct'] * 100:.1f}%"
            )

        except Exception as exc:

            print()
            print(
                "=" * 80
            )

            print(
                f"CANDIDATE FAILED: {name}"
            )

            print(
                repr(exc)
            )

            print(
                "=" * 80
            )

            rankings.append({

                "candidate":
                    name,

                "score":
                    -999999.0,

                "error":
                    repr(exc),
            })

    # ========================================================
    # RANKING
    # ========================================================

    ranking_df = pd.DataFrame(
        rankings
    )

    ranking_df = (
        ranking_df
        .sort_values(
            "score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    ranking_path = (
        MODEL_DIR
        / "champion_ranking.csv"
    )

    ranking_df.to_csv(
        ranking_path,
        index=False
    )

    json_path = (
        MODEL_DIR
        / "champion_ranking.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            ranking_df.to_dict(
                orient="records"
            ),
            file,
            indent=2
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 80)
    print("V9.3 CHAMPION SEARCH COMPLETE")
    print("=" * 80)

    print()

    display_columns = [
        "candidate",
        "score",
        "avg_return",
        "median_return",
        "avg_sharpe",
        "avg_drawdown",
        "worst_drawdown",
        "avg_trades",
        "max_action_pct",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in ranking_df.columns
    ]

    print(
        ranking_df[
            available_columns
        ].to_string(
            index=False
        )
    )

    print()

    if not ranking_df.empty:

        winner = (
            ranking_df.iloc[0]
        )

        print(
            "=" * 80
        )

        print(
            "CURRENT CHAMPION"
        )

        print(
            "=" * 80
        )

        print(
            f"Candidate: "
            f"{winner['candidate']}"
        )

        print(
            f"Score: "
            f"{winner['score']:.4f}"
        )

        if (
            "avg_return"
            in winner
        ):

            print(
                f"Average return: "
                f"{winner['avg_return'] * 100:.2f}%"
            )

            print(
                f"Median return: "
                f"{winner['median_return'] * 100:.2f}%"
            )

            print(
                f"Average Sharpe: "
                f"{winner['avg_sharpe']:.3f}"
            )

            print(
                f"Average max DD: "
                f"{winner['avg_drawdown'] * 100:.2f}%"
            )

            print(
                f"Worst max DD: "
                f"{winner['worst_drawdown'] * 100:.2f}%"
            )

            print(
                f"Average trades: "
                f"{winner['avg_trades']:.1f}"
            )

            print(
                f"Max action concentration: "
                f"{winner['max_action_pct'] * 100:.1f}%"
            )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "This is only a SHORTLIST champion."
        )

        print(
            "It must pass walk-forward "
            "and unseen-data testing before "
            "we deploy anything to OCI."
        )

    print()
    print(
        f"Ranking saved to:"
    )

    print(
        ranking_path
    )

    print(
        json_path
    )


if __name__ == "__main__":

    main()