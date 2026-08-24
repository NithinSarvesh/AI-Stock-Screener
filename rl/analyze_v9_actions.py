"""
Universal PPO V9 Action Intelligence Analyzer

Purpose
-------
Analyze whether PPO V9 actions contain useful forward-looking information.

For every model decision we record:

    action
    position
    price
    RSI
    momentum
    trend/context features
    future 1D return
    future 3D return
    future 5D return
    future 10D return

Then aggregate the results by action.

This is a DIAGNOSTIC tool.

It does NOT train the model.
It does NOT modify the model.
It does NOT promote a model.
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
# IMPORTS
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
    "action_analysis",
)

# -------------------------------------------------------------
# Same 5 stocks used for the current V9 walk-forward test.
# -------------------------------------------------------------

STOCKS = [
    "RELIANCE",
    "TCS",
    "INFY",
    "SBIN",
    "HDFCBANK",
]

HISTORY_PERIOD = "5y"

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20

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
    Download and prepare stock data using the same feature pipeline
    used by V9.
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
        # Indicator pipeline.
        # -------------------------------------------------------------

        features = (
            IndicatorEngine(
                df.copy()
            ).calculate_all()
        )

        # -------------------------------------------------------------
        # V6/V9 context features.
        # -------------------------------------------------------------

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
# TEST SPLIT
# =====================================================================

def get_test_data(
    df: pd.DataFrame,
):
    """
    Use the exact same chronological split as walk-forward V9.

    60% train
    20% validation
    20% test
    """

    n = len(df)

    validation_end = int(
        n
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    test_df = (
        df.iloc[
            validation_end:
        ]
        .copy()
    )

    return test_df


# =====================================================================
# FEATURE DISCOVERY
# =====================================================================

def find_column(
    df: pd.DataFrame,
    candidates,
):
    """
    Find the first matching column.

    Column names can vary slightly depending on the indicator pipeline.
    """

    normalized = {
        str(column).lower().replace(
            " ",
            "_",
        ): column
        for column in df.columns
    }

    for candidate in candidates:

        key = (
            candidate
            .lower()
            .replace(
                " ",
                "_",
            )
        )

        if key in normalized:

            return normalized[key]

    return None


# =====================================================================
# ACTION ANALYSIS
# =====================================================================

def analyze_stock(
    model,
    symbol: str,
    test_df: pd.DataFrame,
):
    """
    Run PPO over the test period and record future returns.
    """

    if len(test_df) < 30:

        raise ValueError(
            "Test period too short."
        )

    # -------------------------------------------------------------
    # Locate price column.
    # -------------------------------------------------------------

    close_column = find_column(
        test_df,
        [
            "Close",
            "close",
            "Adj Close",
            "adj_close",
        ],
    )

    if close_column is None:

        raise RuntimeError(
            "Could not find Close column."
        )

    prices = (
        pd.to_numeric(
            test_df[
                close_column
            ],
            errors="coerce",
        )
        .astype(float)
    )

    # -------------------------------------------------------------
    # Create environment.
    #
    # We use the same test period as the walk-forward evaluator.
    # -------------------------------------------------------------

    env = StockTradingEnvV9(
        dataframe=test_df,
        initial_balance=100_000.0,
        transaction_cost=0.0005,
        episode_length=len(test_df) - 1,
        random_start=False,
    )

    obs, info = env.reset(
        seed=SEED
    )

    rows = []

    done = False

    step = 0

    while not done:

        # ---------------------------------------------------------
        # PPO decision.
        # ---------------------------------------------------------

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

        action_info = ACTION_MAP[
            action
        ]

        # ---------------------------------------------------------
        # Current date/price.
        # ---------------------------------------------------------

        if step >= len(test_df):

            break

        date = test_df.index[
            step
        ]

        current_price = float(
            prices.iloc[
                step
            ]
        )

        # ---------------------------------------------------------
        # Future returns.
        #
        # IMPORTANT:
        # These are calculated AFTER the model decision.
        # Therefore they are diagnostics only and are not
        # fed into the model.
        # ---------------------------------------------------------

        future_returns = {}

        for horizon in [
            1,
            3,
            5,
            10,
        ]:

            future_index = (
                step
                + horizon
            )

            if (
                future_index
                < len(prices)
            ):

                future_price = float(
                    prices.iloc[
                        future_index
                    ]
                )

                if current_price != 0:

                    future_return = (
                        future_price
                        / current_price
                        - 1.0
                    )

                else:

                    future_return = np.nan

            else:

                future_return = np.nan

            future_returns[
                horizon
            ] = future_return

        # ---------------------------------------------------------
        # Current feature values.
        #
        # We record useful features if they exist.
        # ---------------------------------------------------------

        feature_row = (
            test_df.iloc[
                step
            ]
        )

        rsi_column = find_column(
            test_df,
            [
                "RSI",
                "rsi",
                "RSI_14",
                "rsi_14",
            ],
        )

        momentum_column = find_column(
            test_df,
            [
                "momentum",
                "Momentum",
                "momentum_10",
                "momentum_20",
            ],
        )

        atr_column = find_column(
            test_df,
            [
                "ATR",
                "atr",
                "ATR_14",
                "atr_14",
            ],
        )

        volume_column = find_column(
            test_df,
            [
                "Volume",
                "volume",
            ],
        )

        record = {
            "symbol":
                symbol,

            "date":
                str(date),

            "step":
                step,

            "action":
                action,

            "action_name":
                action_info[
                    "name"
                ],

            "position":
                action_info[
                    "position"
                ],

            "price":
                current_price,

            "future_1d_return":
                future_returns[1],

            "future_3d_return":
                future_returns[3],

            "future_5d_return":
                future_returns[5],

            "future_10d_return":
                future_returns[10],
        }

        # ---------------------------------------------------------
        # Optional diagnostic features.
        # ---------------------------------------------------------

        if rsi_column is not None:

            record["rsi"] = float(
                feature_row[
                    rsi_column
                ]
            )

        else:

            record["rsi"] = np.nan

        if momentum_column is not None:

            record["momentum"] = float(
                feature_row[
                    momentum_column
                ]
            )

        else:

            record["momentum"] = np.nan

        if atr_column is not None:

            record["atr"] = float(
                feature_row[
                    atr_column
                ]
            )

        else:

            record["atr"] = np.nan

        if volume_column is not None:

            record["volume"] = float(
                feature_row[
                    volume_column
                ]
            )

        else:

            record["volume"] = np.nan

        rows.append(
            record
        )

        # ---------------------------------------------------------
        # Environment transition.
        # ---------------------------------------------------------

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        done = (
            terminated
            or truncated
        )

        step += 1

    return pd.DataFrame(
        rows
    )


# =====================================================================
# ACTION SUMMARY
# =====================================================================

def summarize_actions(
    decisions: pd.DataFrame,
):
    """
    Aggregate future returns by PPO action.
    """

    results = []

    for action in sorted(
        decisions[
            "action"
        ].dropna()
        .unique()
    ):

        subset = decisions[
            decisions[
                "action"
            ]
            == action
        ].copy()

        if subset.empty:

            continue

        action_name = ACTION_MAP[
            int(action)
        ]["name"]

        row = {
            "action":
                int(action),

            "action_name":
                action_name,

            "samples":
                int(len(subset)),
        }

        # ---------------------------------------------------------
        # Future return statistics.
        # ---------------------------------------------------------

        for horizon in [
            1,
            3,
            5,
            10,
        ]:

            column = (
                f"future_{horizon}d_return"
            )

            values = (
                pd.to_numeric(
                    subset[
                        column
                    ],
                    errors="coerce",
                )
                .dropna()
            )

            if len(values) > 0:

                row[
                    f"avg_{horizon}d_return"
                ] = float(
                    values.mean()
                )

                row[
                    f"median_{horizon}d_return"
                ] = float(
                    values.median()
                )

                row[
                    f"win_rate_{horizon}d"
                ] = float(
                    (
                        values > 0
                    ).mean()
                )

            else:

                row[
                    f"avg_{horizon}d_return"
                ] = np.nan

                row[
                    f"median_{horizon}d_return"
                ] = np.nan

                row[
                    f"win_rate_{horizon}d"
                ] = np.nan

        # ---------------------------------------------------------
        # Feature averages.
        # ---------------------------------------------------------

        for feature in [
            "rsi",
            "momentum",
            "atr",
        ]:

            if feature in subset:

                values = (
                    pd.to_numeric(
                        subset[
                            feature
                        ],
                        errors="coerce",
                    )
                    .dropna()
                )

                row[
                    f"avg_{feature}"
                ] = (
                    float(
                        values.mean()
                    )
                    if len(values) > 0
                    else np.nan
                )

        results.append(
            row
        )

    return pd.DataFrame(
        results
    )


# =====================================================================
# OVERALL FUTURE-RETURN BASELINE
# =====================================================================

def calculate_baseline(
    decisions: pd.DataFrame,
):
    """
    Calculate future returns across ALL model decisions.

    This gives us a baseline.

    Example:

        All observations:
            +0.3% average 5D return

        LONG:
            +0.8%

    That suggests LONG decisions are associated with
    better-than-average future returns.
    """

    baseline = {}

    for horizon in [
        1,
        3,
        5,
        10,
    ]:

        column = (
            f"future_{horizon}d_return"
        )

        values = (
            pd.to_numeric(
                decisions[
                    column
                ],
                errors="coerce",
            )
            .dropna()
        )

        if len(values) == 0:

            baseline[
                f"avg_{horizon}d_return"
            ] = np.nan

            baseline[
                f"win_rate_{horizon}d"
            ] = np.nan

        else:

            baseline[
                f"avg_{horizon}d_return"
            ] = float(
                values.mean()
            )

            baseline[
                f"win_rate_{horizon}d"
            ] = float(
                (
                    values > 0
                ).mean()
            )

    return baseline


# =====================================================================
# MAIN
# =====================================================================

def main():

    print()
    print("=" * 80)
    print("V9 ACTION INTELLIGENCE ANALYSIS")
    print("=" * 80)

    print(
        f"Model:\n{MODEL_PATH}"
    )

    print(
        f"Stocks: {len(STOCKS)}"
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

    all_decisions = []

    # =================================================================
    # STOCK LOOP
    # =================================================================

    for index, symbol in enumerate(
        STOCKS,
        start=1,
    ):

        print()
        print("=" * 80)

        print(
            f"[{index}/{len(STOCKS)}] "
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

        test_df = get_test_data(
            df
        )

        print(
            f"Test rows: "
            f"{len(test_df)}"
        )

        print(
            f"Test range: "
            f"{test_df.index[0]} "
            f"→ "
            f"{test_df.index[-1]}"
        )

        try:

            decisions = analyze_stock(
                model,
                symbol,
                test_df,
            )

            print(
                f"Decisions recorded: "
                f"{len(decisions)}"
            )

            all_decisions.append(
                decisions
            )

        except Exception as exc:

            print(
                f"ANALYSIS ERROR: "
                f"{exc}"
            )

    # =================================================================
    # COMBINE
    # =================================================================

    if not all_decisions:

        raise RuntimeError(
            "No action data collected."
        )

    decisions_df = pd.concat(
        all_decisions,
        ignore_index=True,
    )

    # =================================================================
    # ACTION SUMMARY
    # =================================================================

    action_summary = summarize_actions(
        decisions_df
    )

    baseline = calculate_baseline(
        decisions_df
    )

    # =================================================================
    # PRINT RESULTS
    # =================================================================

    print()
    print("=" * 80)
    print("ACTION → FUTURE RETURN ANALYSIS")
    print("=" * 80)

    print()

    for _, row in (
        action_summary.iterrows()
    ):

        print(
            f"{row['action_name']}"
        )

        print(
            f"  Samples: "
            f"{int(row['samples'])}"
        )

        for horizon in [
            1,
            3,
            5,
            10,
        ]:

            avg = row[
                f"avg_{horizon}d_return"
            ]

            win = row[
                f"win_rate_{horizon}d"
            ]

            if pd.notna(avg):

                print(
                    f"  {horizon:2d}D: "
                    f"avg={avg * 100:+.3f}% "
                    f"win={win * 100:.1f}%"
                )

        print()

    # =================================================================
    # BASELINE
    # =================================================================

    print("=" * 80)
    print("MARKET BASELINE")
    print("=" * 80)

    for horizon in [
        1,
        3,
        5,
        10,
    ]:

        avg = baseline[
            f"avg_{horizon}d_return"
        ]

        win = baseline[
            f"win_rate_{horizon}d"
        ]

        print(
            f"{horizon:2d}D: "
            f"avg={avg * 100:+.3f}% "
            f"win={win * 100:.1f}%"
        )

    # =================================================================
    # ACTION VS BASELINE
    # =================================================================

    comparison_rows = []

    for _, row in (
        action_summary.iterrows()
    ):

        comparison = {
            "action":
                row[
                    "action"
                ],

            "action_name":
                row[
                    "action_name"
                ],

            "samples":
                row[
                    "samples"
                ],
        }

        for horizon in [
            1,
            3,
            5,
            10,
        ]:

            model_avg = row[
                f"avg_{horizon}d_return"
            ]

            baseline_avg = baseline[
                f"avg_{horizon}d_return"
            ]

            comparison[
                f"edge_{horizon}d"
            ] = (
                model_avg
                - baseline_avg
                if (
                    pd.notna(model_avg)
                    and pd.notna(
                        baseline_avg
                    )
                )
                else np.nan
            )

        comparison_rows.append(
            comparison
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    print()
    print("=" * 80)
    print("ACTION EDGE VS MARKET BASELINE")
    print("=" * 80)

    for _, row in (
        comparison_df.iterrows()
    ):

        print(
            f"{row['action_name']}"
        )

        for horizon in [
            1,
            3,
            5,
            10,
        ]:

            edge = row[
                f"edge_{horizon}d"
            ]

            if pd.notna(edge):

                print(
                    f"  {horizon:2d}D edge: "
                    f"{edge * 100:+.3f}%"
                )

        print()

    # =================================================================
    # SAVE RAW DECISIONS
    # =================================================================

    decisions_path = os.path.join(
        OUTPUT_DIR,
        "v9_action_decisions.csv",
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "v9_action_summary.csv",
    )

    comparison_path = os.path.join(
        OUTPUT_DIR,
        "v9_action_edge_vs_baseline.csv",
    )

    baseline_path = os.path.join(
        OUTPUT_DIR,
        "v9_market_baseline.json",
    )

    decisions_df.to_csv(
        decisions_path,
        index=False,
    )

    action_summary.to_csv(
        summary_path,
        index=False,
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    with open(
        baseline_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            baseline,
            file,
            indent=4,
        )

    # =================================================================
    # COMPLETE
    # =================================================================

    print()
    print("=" * 80)
    print("V9 ACTION ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        "Saved:"
    )

    print(
        decisions_path
    )

    print(
        summary_path
    )

    print(
        comparison_path
    )

    print(
        baseline_path
    )

    print("=" * 80)


if __name__ == "__main__":

    main()