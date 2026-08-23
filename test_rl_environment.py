"""
Test the custom reinforcement learning trading environment.

Run:

    python test_rl_environment.py

This does NOT train an RL model.

It only verifies that the environment behaves correctly.
"""

import numpy as np

from stock_fetcher import StockFetcher
from indicators import IndicatorEngine
from rl.trading_env import StockTradingEnv


# ============================================================
# CONFIGURATION
# ============================================================

TICKER = "RELIANCE"


# ============================================================
# LOAD MARKET DATA
# ============================================================

print()
print("=" * 70)
print("LOADING MARKET DATA")
print("=" * 70)

stock = StockFetcher(
    TICKER
)

data = stock.fetch_all()

history = IndicatorEngine(
    data["history"]
).calculate_all()


print(
    f"Resolved symbol: {stock.symbol}"
)

print(
    f"Rows before environment: {len(history)}"
)


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

print()
print("=" * 70)
print("CREATING RL ENVIRONMENT")
print("=" * 70)


env = StockTradingEnv(
    dataframe=history,
    initial_balance=100000.0,
    transaction_cost=0.0005,
)


print(
    f"Observation space: {env.observation_space}"
)

print(
    f"Action space: {env.action_space}"
)

print(
    f"Observation size: "
    f"{env.observation_space.shape}"
)


# ============================================================
# RESET
# ============================================================

print()
print("=" * 70)
print("TESTING RESET")
print("=" * 70)


observation, info = env.reset(
    seed=42
)


print(
    f"Observation shape: "
    f"{observation.shape}"
)

print(
    f"Observation dtype: "
    f"{observation.dtype}"
)

print(
    f"Initial balance: "
    f"{info['balance']:.2f}"
)

print(
    f"Initial position: "
    f"{info['position']}"
)


# ============================================================
# CHECK OBSERVATION
# ============================================================

assert (
    observation.shape
    == env.observation_space.shape
), (
    "Observation shape does not match "
    "observation_space."
)


assert (
    observation.dtype
    == np.float32
), (
    "Observation dtype must be float32."
)


assert np.all(
    np.isfinite(observation)
), (
    "Observation contains NaN or infinity."
)


print(
    "✅ Observation validation passed."
)


# ============================================================
# TEST ACTIONS
# ============================================================

print()
print("=" * 70)
print("TESTING ACTIONS")
print("=" * 70)


actions = [
    0,  # HOLD
    1,  # BUY
    0,  # HOLD
    2,  # SELL
    0,  # HOLD
    1,  # BUY
]


for step_number, action in enumerate(
    actions,
    start=1,
):

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    print(
        f"""
Step {step_number}
----------------
Action       : {info['action_name']}
Position     : {info['position_name']}
Price        : {info['current_price']:.2f}
Next Price   : {info['next_price']:.2f}
Market Return: {info['market_return']:.6f}
Net Return   : {info['net_return']:.6f}
Reward       : {reward:.6f}
Equity       : {info['equity']:.2f}
Drawdown     : {info['drawdown'] * 100:.2f}%
Trades       : {info['total_trades']}
"""
    )

    assert (
        observation.shape
        == env.observation_space.shape
    )

    assert np.all(
        np.isfinite(observation)
    )

    assert np.isfinite(
        reward
    )

    if terminated or truncated:

        print(
            "Episode ended."
        )

        break


# ============================================================
# RANDOM ACTION TEST
# ============================================================

print()
print("=" * 70)
print("TESTING RANDOM ACTIONS")
print("=" * 70)


observation, info = env.reset(
    seed=123
)


random_steps = 0

while random_steps < 100:

    action = env.action_space.sample()

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(action)

    assert np.all(
        np.isfinite(observation)
    )

    assert np.isfinite(
        reward
    )

    random_steps += 1

    if terminated or truncated:

        break


print(
    f"Random-action test completed: "
    f"{random_steps} steps."
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("RL ENVIRONMENT TEST COMPLETE")
print("=" * 70)

print(
    "✅ Environment created successfully."
)

print(
    "✅ Observation space works."
)

print(
    "✅ Action space works."
)

print(
    "✅ BUY/HOLD/SELL actions work."
)

print(
    "✅ Rewards are numeric."
)

print(
    "✅ Observations contain no NaN/Inf."
)

print(
    "✅ Random actions run successfully."
)

print()
print(
    "The environment is ready for Gymnasium/SB3 validation."
)