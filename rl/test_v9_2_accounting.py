from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yfinance as yf

# ---------------------------------------------------------------------
# Make project root importable when executing rl\*.py directly
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from indicators import IndicatorEngine
from rl.v6_inference import PPOV6Inference
from rl.trading_env_v9_2 import (
    ACTION_MAP,
    StockTradingEnvV92,
)


# =====================================================================
# CONFIG
# =====================================================================

SYMBOL = "RELIANCE.NS"

INITIAL_BALANCE = 100000.0

TRANSACTION_COST = 0.0005

EPISODE_LENGTH = 252


# =====================================================================
# DATA
# =====================================================================

def load_data():

    print("Downloading:", SYMBOL)

    df = yf.download(
        SYMBOL,
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise RuntimeError(
            "Yahoo Finance returned empty data."
        )

    if (
        hasattr(df.columns, "nlevels")
        and df.columns.nlevels > 1
    ):
        df.columns = df.columns.get_level_values(0)

    df = IndicatorEngine(df).calculate_all()

    df = PPOV6Inference.add_context_features(df)

    df = df.dropna().copy()

    print("Prepared rows:", len(df))

    if len(df) < 300:
        raise RuntimeError(
            f"Not enough prepared rows: {len(df)}"
        )

    return df


# =====================================================================
# ENVIRONMENT CREATION
# =====================================================================

def create_env(df):

    env = StockTradingEnvV92(
        df,
        initial_balance=INITIAL_BALANCE,
        transaction_cost=TRANSACTION_COST,
        episode_length=EPISODE_LENGTH,
        random_start=False,
    )

    return env


# =====================================================================
# PRINT INFO
# =====================================================================

def print_info(info):

    if not isinstance(info, dict):
        print("INFO TYPE:", type(info))
        print("INFO:", info)
        return

    print("INFO KEYS:")

    for key in sorted(info.keys()):

        value = info[key]

        if isinstance(
            value,
            (float, np.floating),
        ):
            print(
                f"  {key:25s}: "
                f"{float(value):+.10f}"
            )
        else:
            print(
                f"  {key:25s}: "
                f"{value}"
            )


# =====================================================================
# SINGLE ACTION TEST
# =====================================================================

def test_action(
    df,
    action,
):

    action_name = ACTION_MAP[
        action
    ]["name"]

    expected_position = ACTION_MAP[
        action
    ]["position"]

    print()
    print("-" * 80)
    print(
        f"ACTION {action}: "
        f"{action_name}"
    )
    print("-" * 80)

    env = create_env(df)

    observation, reset_info = env.reset(
        seed=42
    )

    print(
        "Initial observation shape:",
        observation.shape,
    )

    print(
        "Initial reset info:"
    )

    print_info(reset_info)

    (
        next_observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    print()
    print(
        "Expected position:",
        expected_position,
    )

    print(
        "Reward:",
        f"{reward:+.10f}",
    )

    print(
        "Terminated:",
        terminated,
    )

    print(
        "Truncated:",
        truncated,
    )

    print()
    print("STEP INFO:")

    print_info(info)

    # -------------------------------------------------------------
    # Extract values using several possible names.
    # This makes the diagnostic robust to V9.2 naming differences.
    # -------------------------------------------------------------

    market_return = info.get(
        "market_return",
        np.nan,
    )

    strategy_return = info.get(
        "strategy_return",
        np.nan,
    )

    portfolio_value = info.get(
        "portfolio_value",
        np.nan,
    )

    equity = info.get(
        "equity",
        np.nan,
    )

    position = info.get(
        "position",
        np.nan,
    )

    current_price = info.get(
        "price",
        np.nan,
    )

    previous_price = info.get(
        "previous_price",
        np.nan,
    )

    print()
    print("IMPORTANT ACCOUNTING VALUES")
    print(
        "market_return     :",
        market_return,
    )

    print(
        "strategy_return   :",
        strategy_return,
    )

    print(
        "portfolio_value   :",
        portfolio_value,
    )

    print(
        "equity            :",
        equity,
    )

    print(
        "position          :",
        position,
    )

    print(
        "previous_price    :",
        previous_price,
    )

    print(
        "current_price     :",
        current_price,
    )

    # -------------------------------------------------------------
    # Basic sanity checks
    # -------------------------------------------------------------

    if np.isfinite(position):

        print()

        if abs(
            float(position)
            - expected_position
        ) < 1e-6:

            print(
                "POSITION CHECK: PASS"
            )

        else:

            print(
                "POSITION CHECK: FAIL"
            )

    else:

        print(
            "POSITION CHECK: INFO VALUE NOT FOUND"
        )

    # -------------------------------------------------------------
    # Portfolio-value sanity check
    # -------------------------------------------------------------

    if np.isfinite(portfolio_value):

        expected_value = (
            INITIAL_BALANCE
            * (
                1.0
                + expected_position
                * float(market_return)
            )
        )

        print()
        print(
            "Approx expected portfolio "
            "value:",
            f"{expected_value:.4f}",
        )

        print(
            "Actual portfolio value:",
            f"{float(portfolio_value):.4f}",
        )

        difference = (
            float(portfolio_value)
            - expected_value
        )

        print(
            "Difference:",
            f"{difference:+.4f}",
        )

    elif np.isfinite(equity):

        print()
        print(
            "Equity value found instead "
            "of portfolio_value:"
        )

        print(
            f"Equity: {float(equity):.4f}"
        )

    else:

        print()
        print(
            "WARNING: Could not find "
            "portfolio/equity value in info."
        )

    return {
        "action": action,
        "action_name": action_name,
        "expected_position": expected_position,
        "position": position,
        "market_return": market_return,
        "strategy_return": strategy_return,
        "portfolio_value": portfolio_value,
        "equity": equity,
        "reward": reward,
        "info": info,
    }


# =====================================================================
# MULTI-STEP LONG TEST
# =====================================================================

def test_long_episode(df):

    print()
    print("=" * 80)
    print("MULTI-STEP LONG ACCOUNTING TEST")
    print("=" * 80)

    env = create_env(df)

    observation, info = env.reset(
        seed=42
    )

    starting_balance = INITIAL_BALANCE

    values = [
        starting_balance
    ]

    returns = []

    positions = []

    for step in range(20):

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(4)  # LONG

        position = info.get(
            "position",
            np.nan,
        )

        portfolio_value = info.get(
            "portfolio_value",
            info.get(
                "equity",
                np.nan,
            ),
        )

        strategy_return = info.get(
            "strategy_return",
            np.nan,
        )

        market_return = info.get(
            "market_return",
            np.nan,
        )

        positions.append(position)

        returns.append(
            strategy_return
        )

        if np.isfinite(
            portfolio_value
        ):

            values.append(
                float(portfolio_value)
            )

        print(
            f"Step {step + 1:02d} | "
            f"market={market_return:+.6f} | "
            f"strategy={strategy_return:+.6f} | "
            f"position={position} | "
            f"value={portfolio_value}"
        )

        if terminated or truncated:
            break

    print()
    print(
        "Starting balance:",
        f"{starting_balance:.2f}",
    )

    print(
        "Ending value:",
        f"{values[-1]:.2f}",
    )

    if len(values) > 1:

        total_change = (
            values[-1]
            / values[0]
            - 1.0
        )

        print(
            "Equity change:",
            f"{total_change * 100:+.4f}%",
        )

        if abs(total_change) > 1e-8:

            print(
                "LONG EQUITY MOVEMENT: PASS"
            )

        else:

            print(
                "LONG EQUITY MOVEMENT: FAIL"
            )

    else:

        print(
            "Could not obtain portfolio "
            "values."
        )

    unique_positions = sorted(
        set(
            float(x)
            for x in positions
            if np.isfinite(x)
        )
    )

    print(
        "Observed LONG positions:",
        unique_positions,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 80)
    print("V9.2 ACCOUNTING / EQUITY SANITY TEST")
    print("=" * 80)

    print()
    print(
        "This test checks whether:"
    )

    print(
        "LONG  -> portfolio follows market"
    )

    print(
        "SHORT -> portfolio moves opposite market"
    )

    print(
        "FLAT  -> portfolio remains unchanged"
    )

    print()

    df = load_data()

    env = create_env(df)

    print()
    print(
        "Observation space:",
        env.observation_space,
    )

    print(
        "Action space:",
        env.action_space,
    )

    print()
    print("=" * 80)
    print("TESTING ALL FIVE ACTIONS")
    print("=" * 80)

    results = []

    for action in range(5):

        results.append(
            test_action(
                df,
                action,
            )
        )

    # =================================================================
    # SUMMARY
    # =================================================================

    print()
    print("=" * 80)
    print("ACTION ACCOUNTING SUMMARY")
    print("=" * 80)

    for result in results:

        print(
            f"{result['action']} | "
            f"{result['action_name']:12s} | "
            f"position="
            f"{result['expected_position']:+.1f} | "
            f"market="
            f"{result['market_return']} | "
            f"strategy="
            f"{result['strategy_return']} | "
            f"reward="
            f"{result['reward']:+.8f}"
        )

    # =================================================================
    # LONG MULTI-STEP TEST
    # =================================================================

    test_long_episode(df)

    # =================================================================
    # FINAL VERDICT
    # =================================================================

    print()
    print("=" * 80)
    print("V9.2 ACCOUNTING TEST COMPLETE")
    print("=" * 80)

    print()
    print(
        "DO NOT TRAIN ANOTHER MODEL YET."
    )

    print(
        "We first verify that the evaluator "
        "and environment calculate equity correctly."
    )

    print()


if __name__ == "__main__":
    main()