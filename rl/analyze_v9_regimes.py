"""
V9 Market Regime Analyzer

Analyzes what market conditions exist when PPO V9 chooses:

    SHORT
    HALF_SHORT
    FLAT
    HALF_LONG
    LONG

The goal is to determine whether V9 actions are associated with
specific RSI, momentum, volatility and trend regimes.

This is diagnostic only.
It does NOT train or modify the model.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd


# =====================================================================
# PROJECT ROOT
# =====================================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =====================================================================
# CONFIG
# =====================================================================

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "universal_v9",
    "evaluation",
    "action_analysis",
    "v9_action_decisions.csv",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "universal_v9",
    "evaluation",
    "action_analysis",
    "regimes",
)


# =====================================================================
# HELPERS
# =====================================================================

def classify_rsi(value):
    """
    RSI regime.
    """

    if pd.isna(value):
        return "UNKNOWN"

    if value < 30:
        return "OVERSOLD"

    if value < 45:
        return "BEARISH"

    if value < 55:
        return "NEUTRAL"

    if value < 70:
        return "BULLISH"

    return "OVERBOUGHT"


def classify_momentum(value):
    """
    Momentum regime.

    The exact numerical scale depends on the existing indicator
    pipeline, so we classify using sign and relative magnitude.
    """

    if pd.isna(value):
        return "UNKNOWN"

    if value < -0.05:
        return "STRONG_NEGATIVE"

    if value < 0:
        return "NEGATIVE"

    if value < 0.05:
        return "POSITIVE"

    return "STRONG_POSITIVE"


def classify_atr(value, median_atr):
    """
    Relative volatility regime.

    ATR is compared against the median ATR of the analyzed
    dataset rather than using an arbitrary absolute threshold.
    """

    if pd.isna(value):
        return "UNKNOWN"

    if pd.isna(median_atr):
        return "UNKNOWN"

    if value < median_atr * 0.75:
        return "LOW_VOLATILITY"

    if value > median_atr * 1.50:
        return "HIGH_VOLATILITY"

    return "NORMAL_VOLATILITY"


def classify_trend(row):
    """
    Infer a broad trend regime from available momentum/RSI.

    This is intentionally conservative because we don't want to
    invent a trend feature that wasn't present in the original data.
    """

    rsi = row.get(
        "rsi",
        np.nan,
    )

    momentum = row.get(
        "momentum",
        np.nan,
    )

    if (
        pd.isna(rsi)
        and pd.isna(momentum)
    ):
        return "UNKNOWN"

    if (
        pd.notna(rsi)
        and pd.notna(momentum)
    ):

        if (
            rsi >= 55
            and momentum > 0
        ):
            return "BULLISH_TREND"

        if (
            rsi <= 45
            and momentum < 0
        ):
            return "BEARISH_TREND"

        return "MIXED_TREND"

    if pd.notna(rsi):

        if rsi >= 55:
            return "BULLISH_TREND"

        if rsi <= 45:
            return "BEARISH_TREND"

        return "MIXED_TREND"

    if momentum > 0:
        return "BULLISH_TREND"

    if momentum < 0:
        return "BEARISH_TREND"

    return "MIXED_TREND"


# =====================================================================
# LOAD
# =====================================================================

def load_data():

    if not os.path.exists(
        INPUT_FILE
    ):

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    if df.empty:

        raise RuntimeError(
            "Action decision file is empty."
        )

    required_columns = [
        "action",
        "action_name",
        "position",
        "future_1d_return",
        "future_3d_return",
        "future_5d_return",
        "future_10d_return",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    return df


# =====================================================================
# ADD REGIMES
# =====================================================================

def add_regimes(
    df: pd.DataFrame,
):

    result = df.copy()

    # -------------------------------------------------------------
    # Convert numerical columns.
    # -------------------------------------------------------------

    for column in [
        "rsi",
        "momentum",
        "atr",
        "future_1d_return",
        "future_3d_return",
        "future_5d_return",
        "future_10d_return",
    ]:

        if column in result.columns:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    # -------------------------------------------------------------
    # ATR median.
    # -------------------------------------------------------------

    median_atr = (
        result["atr"].median()
        if "atr" in result.columns
        else np.nan
    )

    # -------------------------------------------------------------
    # Regime classification.
    # -------------------------------------------------------------

    if "rsi" in result.columns:

        result[
            "rsi_regime"
        ] = result[
            "rsi"
        ].apply(
            classify_rsi
        )

    else:

        result[
            "rsi_regime"
        ] = "UNKNOWN"

    if "momentum" in result.columns:

        result[
            "momentum_regime"
        ] = result[
            "momentum"
        ].apply(
            classify_momentum
        )

    else:

        result[
            "momentum_regime"
        ] = "UNKNOWN"

    if "atr" in result.columns:

        result[
            "volatility_regime"
        ] = result[
            "atr"
        ].apply(
            lambda x:
                classify_atr(
                    x,
                    median_atr,
                )
        )

    else:

        result[
            "volatility_regime"
        ] = "UNKNOWN"

    result[
        "trend_regime"
    ] = result.apply(
        classify_trend,
        axis=1,
    )

    return result


# =====================================================================
# ACTION × REGIME ANALYSIS
# =====================================================================

def analyze_action_by_regime(
    df: pd.DataFrame,
    regime_column: str,
):

    rows = []

    for regime in sorted(
        df[
            regime_column
        ]
        .dropna()
        .unique()
    ):

        regime_df = df[
            df[
                regime_column
            ]
            == regime
        ]

        for action_name in [
            "SHORT",
            "HALF_SHORT",
            "FLAT",
            "HALF_LONG",
            "LONG",
        ]:

            subset = regime_df[
                regime_df[
                    "action_name"
                ]
                == action_name
            ]

            if subset.empty:
                continue

            row = {
                "regime":
                    regime,

                "action_name":
                    action_name,

                "samples":
                    int(len(subset)),
            }

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

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# ACTION DISTRIBUTION BY REGIME
# =====================================================================

def action_distribution_by_regime(
    df: pd.DataFrame,
    regime_column: str,
):

    rows = []

    for regime in sorted(
        df[
            regime_column
        ]
        .dropna()
        .unique()
    ):

        subset = df[
            df[
                regime_column
            ]
            == regime
        ]

        counts = (
            subset[
                "action_name"
            ]
            .value_counts()
        )

        total = len(subset)

        row = {
            "regime":
                regime,

            "samples":
                int(total),
        }

        for action_name in [
            "SHORT",
            "HALF_SHORT",
            "FLAT",
            "HALF_LONG",
            "LONG",
        ]:

            row[
                action_name
                + "_pct"
            ] = float(
                counts.get(
                    action_name,
                    0,
                )
                / total
                * 100.0
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# BEST / WORST ACTIONS
# =====================================================================

def find_best_actions(
    analysis_df: pd.DataFrame,
):

    rows = []

    for regime in sorted(
        analysis_df[
            "regime"
        ]
        .unique()
    ):

        subset = analysis_df[
            analysis_df[
                "regime"
            ]
            == regime
        ].copy()

        subset = subset[
            subset[
                "samples"
            ]
            >= 5
        ]

        if subset.empty:
            continue

        for horizon in [
            5,
            10,
        ]:

            column = (
                f"avg_{horizon}d_return"
            )

            ranked = subset.sort_values(
                column,
                ascending=False,
            )

            best = ranked.iloc[
                0
            ]

            worst = ranked.iloc[
                -1
            ]

            rows.append(
                {
                    "regime":
                        regime,

                    "horizon":
                        horizon,

                    "best_action":
                        best[
                            "action_name"
                        ],

                    "best_avg_return":
                        float(
                            best[
                                column
                            ]
                        ),

                    "best_samples":
                        int(
                            best[
                                "samples"
                            ]
                        ),

                    "worst_action":
                        worst[
                            "action_name"
                        ],

                    "worst_avg_return":
                        float(
                            worst[
                                column
                            ]
                        ),

                    "worst_samples":
                        int(
                            worst[
                                "samples"
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# PRINT REPORT
# =====================================================================

def print_action_distribution(
    df,
    regime_column,
):

    distribution = (
        action_distribution_by_regime(
            df,
            regime_column,
        )
    )

    print()
    print("=" * 80)
    print(
        "ACTION DISTRIBUTION BY "
        + regime_column.upper()
    )
    print("=" * 80)

    for _, row in (
        distribution.iterrows()
    ):

        print()
        print(
            f"{row['regime']} "
            f"(samples={int(row['samples'])})"
        )

        for action_name in [
            "SHORT",
            "HALF_SHORT",
            "FLAT",
            "HALF_LONG",
            "LONG",
        ]:

            print(
                f"  "
                f"{action_name:12s}: "
                f"{row[action_name + '_pct']:6.2f}%"
            )


def print_best_actions(
    best_df,
):

    print()
    print("=" * 80)
    print("BEST / WORST ACTIONS BY REGIME")
    print("=" * 80)

    for _, row in (
        best_df.iterrows()
    ):

        print(
            f"{row['regime']} "
            f"| {int(row['horizon'])}D"
        )

        print(
            f"  BEST : "
            f"{row['best_action']:12s} "
            f"{row['best_avg_return'] * 100:+.3f}% "
            f"(n={int(row['best_samples'])})"
        )

        print(
            f"  WORST: "
            f"{row['worst_action']:12s} "
            f"{row['worst_avg_return'] * 100:+.3f}% "
            f"(n={int(row['worst_samples'])})"
        )

        print()


# =====================================================================
# MAIN
# =====================================================================

def main():

    print()
    print("=" * 80)
    print("V9 MARKET REGIME ANALYSIS")
    print("=" * 80)

    print(
        f"Input:\n{INPUT_FILE}"
    )

    # =================================================================
    # LOAD
    # =================================================================

    df = load_data()

    print()
    print(
        f"Decisions loaded: "
        f"{len(df)}"
    )

    print(
        "Columns:"
    )

    print(
        ", ".join(
            df.columns
        )
    )

    # =================================================================
    # REGIMES
    # =================================================================

    df = add_regimes(
        df
    )

    # =================================================================
    # OUTPUT DIRECTORY
    # =================================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # =================================================================
    # SAVE ENRICHED DATA
    # =================================================================

    enriched_path = os.path.join(
        OUTPUT_DIR,
        "v9_action_decisions_with_regimes.csv",
    )

    df.to_csv(
        enriched_path,
        index=False,
    )

    # =================================================================
    # ANALYZE EACH REGIME
    # =================================================================

    regime_columns = [
        "rsi_regime",
        "momentum_regime",
        "volatility_regime",
        "trend_regime",
    ]

    all_analysis = []

    all_distributions = []

    for regime_column in (
        regime_columns
    ):

        analysis = (
            analyze_action_by_regime(
                df,
                regime_column,
            )
        )

        analysis[
            "regime_type"
        ] = regime_column

        all_analysis.append(
            analysis
        )

        distribution = (
            action_distribution_by_regime(
                df,
                regime_column,
            )
        )

        distribution[
            "regime_type"
        ] = regime_column

        all_distributions.append(
            distribution
        )

        print_action_distribution(
            df,
            regime_column,
        )

    # =================================================================
    # COMBINE
    # =================================================================

    action_regime_analysis = (
        pd.concat(
            all_analysis,
            ignore_index=True,
        )
    )

    action_distribution = (
        pd.concat(
            all_distributions,
            ignore_index=True,
        )
    )

    # =================================================================
    # BEST / WORST
    # =================================================================

    best_worst = []

    for regime_column in (
        regime_columns
    ):

        subset = (
            action_regime_analysis[
                action_regime_analysis[
                    "regime_type"
                ]
                == regime_column
            ]
        )

        if subset.empty:
            continue

        best = find_best_actions(
            subset
        )

        best[
            "regime_type"
        ] = regime_column

        best_worst.append(
            best
        )

    if best_worst:

        best_worst_df = (
            pd.concat(
                best_worst,
                ignore_index=True,
            )
        )

    else:

        best_worst_df = pd.DataFrame()

    # =================================================================
    # PRINT
    # =================================================================

    if not best_worst_df.empty:

        print_best_actions(
            best_worst_df
        )

    # =================================================================
    # SAVE
    # =================================================================

    analysis_path = os.path.join(
        OUTPUT_DIR,
        "v9_action_by_regime.csv",
    )

    distribution_path = os.path.join(
        OUTPUT_DIR,
        "v9_action_distribution_by_regime.csv",
    )

    best_worst_path = os.path.join(
        OUTPUT_DIR,
        "v9_best_worst_actions_by_regime.csv",
    )

    action_regime_analysis.to_csv(
        analysis_path,
        index=False,
    )

    action_distribution.to_csv(
        distribution_path,
        index=False,
    )

    best_worst_df.to_csv(
        best_worst_path,
        index=False,
    )

    # =================================================================
    # SIMPLE MACHINE-READABLE SUMMARY
    # =================================================================

    summary = {
        "input_file":
            INPUT_FILE,

        "decision_count":
            int(len(df)),

        "regimes_analyzed":
            regime_columns,

        "output_files": {
            "enriched_decisions":
                enriched_path,

            "action_by_regime":
                analysis_path,

            "action_distribution":
                distribution_path,

            "best_worst":
                best_worst_path,
        },
    }

    summary_path = os.path.join(
        OUTPUT_DIR,
        "v9_regime_analysis_summary.json",
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

    # =================================================================
    # COMPLETE
    # =================================================================

    print()
    print("=" * 80)
    print("V9 REGIME ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        "Saved:"
    )

    print(
        enriched_path
    )

    print(
        analysis_path
    )

    print(
        distribution_path
    )

    print(
        best_worst_path
    )

    print(
        summary_path
    )

    print("=" * 80)


if __name__ == "__main__":

    main()