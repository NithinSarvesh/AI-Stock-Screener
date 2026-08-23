from decision_engine import HybridDecisionEngine


engine = HybridDecisionEngine()


tests = [

    {
        "name": "Bullish agreement",
        "quant": 80,
        "technical": "BUY",
        "rl": {
            "name": "LONG",
            "position": 1.0,
        },
    },

    {
        "name": "Bullish technical, bearish RL",
        "quant": 80,
        "technical": "BUY",
        "rl": {
            "name": "SHORT",
            "position": -1.0,
        },
    },

    {
        "name": "Neutral",
        "quant": 50,
        "technical": "HOLD",
        "rl": {
            "name": "FLAT",
            "position": 0.0,
        },
    },
]


print("=" * 70)
print("HYBRID DECISION ENGINE TEST")
print("=" * 70)


for test in tests:

    result = engine.decide(
        quantitative_score=test["quant"],
        technical_signal=test["technical"],
        rl_result=test["rl"],
    )

    print()
    print(test["name"])
    print("-" * 70)

    print(
        "Quant score:",
        result.quantitative_score
    )

    print(
        "Technical:",
        result.technical_score
    )

    print(
        "RL:",
        result.rl_action,
        result.rl_position
    )

    print(
        "Final score:",
        result.final_score
    )

    print(
        "FINAL:",
        result.final_signal
    )

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)