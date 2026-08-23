"""
Hybrid PPO V6 multi-stock sanity test.

Run from project root:
    python rl/test_hybrid_multi_stock.py

This is NOT a backtest. It checks whether the live inference pipeline
produces sensible outputs across the project's standard stock universe.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stock_fetcher import StockFetcher
from indicators import IndicatorEngine
from support_resistance import SupportResistance
from candlestick import CandlePattern
from core.signal_engine import SignalEngine
from rl.v6_inference import get_v6_signal
from rl.decision_engine import HybridDecisionEngine


TICKERS = [
    "RELIANCE",
    "TCS",
    "INFY",
    "ICICIBANK",
    "SBIN",
    "LT",
    "ITC",
]


def run_one(ticker, engine):
    data = StockFetcher(ticker).fetch_all()

    history = data["history"].copy()
    history = IndicatorEngine(history).calculate_all()
    history = history.dropna(subset=["Close"])

    if len(history) < 50:
        raise RuntimeError(f"Only {len(history)} usable rows")

    latest = history.iloc[-1]

    levels = SupportResistance(history).calculate()
    pattern = CandlePattern(history).detect()

    analysis = SignalEngine(
        latest=latest,
        levels=levels,
        pattern=pattern,
    ).analyze()

    rl_result = get_v6_signal(history, current_position=0.0)

    decision = engine.decide(
        quantitative_score=analysis["score"],
        technical_signal=analysis["signal"],
        rl_result=rl_result,
    )

    return {
        "ticker": ticker,
        "resolved": data.get("stock_symbol", "unknown"),
        "price": float(latest["Close"]),
        "quant": float(analysis["score"]),
        "technical": analysis["signal"],
        "ppo": rl_result["name"],
        "position": float(rl_result["position"]),
        "hybrid_score": float(decision.final_score),
        "final": decision.final_signal,
        "confidence": analysis.get("confidence", "N/A"),
    }


def main():
    print("=" * 100)
    print("HYBRID PPO V6 — MULTI-STOCK SANITY TEST")
    print("=" * 100)
    print()
    print("This tests live inference only. It is NOT a profitability backtest.")
    print()

    engine = HybridDecisionEngine()
    results = []

    for ticker in TICKERS:
        print("-" * 100)
        print(f"TESTING: {ticker}")

        try:
            result = run_one(ticker, engine)
            results.append(result)

            print(
                f"Price ₹{result['price']:.2f} | "
                f"Quant {result['quant']:.1f} | "
                f"Technical {result['technical']} | "
                f"PPO {result['ppo']} ({result['position']:+.1f}) | "
                f"Hybrid {result['hybrid_score']:.1f} | "
                f"FINAL {result['final']}"
            )

        except Exception as exc:
            print(f"FAILED: {ticker} -> {exc}")

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        f"{'Ticker':<12}"
        f"{'Quant':>8}"
        f"{'Technical':>12}"
        f"{'PPO':>14}"
        f"{'Position':>10}"
        f"{'Hybrid':>10}"
        f"{'FINAL':>14}"
    )
    print("-" * 100)

    for r in results:
        print(
            f"{r['ticker']:<12}"
            f"{r['quant']:>8.1f}"
            f"{r['technical']:>12}"
            f"{r['ppo']:>14}"
            f"{r['position']:>10.1f}"
            f"{r['hybrid_score']:>10.1f}"
            f"{r['final']:>14}"
        )

    print()
    print("Interpretation:")
    print("  • This checks that PPO + existing analysis + hybrid logic work together.")
    print("  • It does NOT prove that the resulting signals are profitable.")
    print("  • If every stock produces the same PPO action, investigate model behavior.")
    print("  • If outputs vary with market conditions, proceed to paper/backtest validation.")
    print()


if __name__ == "__main__":
    main()
