"""
Hybrid decision engine.

Combines:
    - Quantitative score
    - Technical/trade direction
    - PPO V6 reinforcement-learning position

The PPO model is treated as an additional signal,
not as an unquestionable trading decision.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HybridDecision:

    final_signal: str

    final_score: float

    technical_score: float

    quantitative_score: float

    rl_score: float

    rl_action: str

    rl_position: float

    explanation: list


class HybridDecisionEngine:

    """
    Hybrid decision system.

    Default weights intentionally keep RL below the
    quantitative + technical components.

    Quantitative : 40%
    Technical    : 35%
    RL           : 25%
    """

    QUANT_WEIGHT = 0.40
    TECH_WEIGHT = 0.35
    RL_WEIGHT = 0.25

    # ---------------------------------------------------------------
    # Convert PPO position to a -100 ... +100 score
    # ---------------------------------------------------------------

    @staticmethod
    def rl_position_to_score(position: float) -> float:

        position = float(position)

        # V6 position range is [-1, +1]
        position = max(-1.0, min(1.0, position))

        return position * 100.0

    # ---------------------------------------------------------------
    # Convert common BUY / SELL / HOLD terminology into score
    # ---------------------------------------------------------------

    @staticmethod
    def signal_to_score(signal: Optional[str]) -> float:

        if signal is None:
            return 0.0

        value = str(signal).strip().upper()

        # Keep the technical signal strength instead of treating
        # BUY and STRONG BUY as identical.
        if value in {
            "STRONG BUY",
            "STRONG_BUY",
        }:
            return 100.0

        if value in {
            "BUY",
            "LONG",
        }:
            return 80.0

        if value in {
            "WEAK BUY",
            "WEAK_BUY",
            "HALF LONG",
            "HALF_LONG",
        }:
            return 50.0

        if value in {
            "STRONG SELL",
            "STRONG_SELL",
        }:
            return -100.0

        if value in {
            "SELL",
            "SHORT",
        }:
            return -80.0

        if value in {
            "WEAK SELL",
            "WEAK_SELL",
            "HALF SHORT",
            "HALF_SHORT",
        }:
            return -50.0

        # WAIT / HOLD / NEUTRAL and unknown signals contribute
        # no directional score.
        return 0.0

    # ---------------------------------------------------------------
    # Normalize technical signal terminology
    # ---------------------------------------------------------------

    @staticmethod
    def normalize_signal(signal: Optional[str]) -> str:

        if signal is None:
            return "WAIT"

        value = str(signal).strip().upper()

        aliases = {
            "STRONG_BUY": "STRONG BUY",
            "STRONG_SELL": "STRONG SELL",
            "WEAK_BUY": "WEAK BUY",
            "WEAK_SELL": "WEAK SELL",
            "HALF_LONG": "HALF LONG",
            "HALF_SHORT": "HALF SHORT",
            "LONG": "BUY",
            "SHORT": "SELL",
        }

        return aliases.get(value, value)

    # ---------------------------------------------------------------
    # Final classification
    # ---------------------------------------------------------------

    @staticmethod
    def classify(score: float) -> str:

        if score >= 60:
            return "STRONG BUY"

        if score >= 30:
            return "BUY"

        if score <= -60:
            return "STRONG SELL"

        if score <= -30:
            return "SELL"

        return "HOLD"

    # ---------------------------------------------------------------
    # Generate decision
    # ---------------------------------------------------------------

    def decide(
        self,
        quantitative_score: float,
        technical_signal: Optional[str],
        rl_result: Optional[dict],
    ) -> HybridDecision:

        quant = float(
            max(
                -100.0,
                min(100.0, quantitative_score),
            )
        )

        technical = self.signal_to_score(
            technical_signal
        )

        if rl_result is None:

            rl_score = 0.0
            rl_action = "UNAVAILABLE"
            rl_position = 0.0

        else:

            rl_position = float(
                rl_result.get("position", 0.0)
            )

            rl_score = self.rl_position_to_score(
                rl_position
            )

            rl_action = str(
                rl_result.get("name", "UNKNOWN")
            )

        # -----------------------------------------------------------
        # Weighted score
        # -----------------------------------------------------------

        final_score = (
            quant * self.QUANT_WEIGHT
            +
            technical * self.TECH_WEIGHT
            +
            rl_score * self.RL_WEIGHT
        )

        final_score = max(
            -100.0,
            min(100.0, final_score),
        )

        final_signal = self.classify(
            final_score
        )

        # -----------------------------------------------------------
        # Technical safety gate
        #
        # A WAIT technical setup means there is no confirmed
        # directional trade setup. Quant/PPO may still produce a
        # positive weighted score, but they must not turn WAIT into
        # BUY. The score remains available for ranking/diagnostics.
        # -----------------------------------------------------------

        normalized_technical = self.normalize_signal(
            technical_signal
        )

        if normalized_technical in {
            "WAIT",
            "HOLD",
            "NEUTRAL",
            "",
        } and final_score > 0:

            final_signal = "HOLD"

        # A bearish technical setup also blocks a bullish final
        # recommendation. The opposite direction is allowed.
        elif normalized_technical in {
            "WEAK SELL",
            "SELL",
            "STRONG SELL",
        } and final_score > 0:

            final_signal = "HOLD"

        # A bullish technical setup should not be turned into a
        # SELL solely by the weighted score. Keep the disagreement
        # as HOLD rather than issuing a contradictory trade signal.
        elif normalized_technical in {
            "WEAK BUY",
            "BUY",
            "STRONG BUY",
        } and final_score < 0:

            final_signal = "HOLD"

        # -----------------------------------------------------------
        # Explanation
        # -----------------------------------------------------------

        explanation = []

        explanation.append(
            f"Quantitative component: "
            f"{quant:.1f}/100"
        )

        explanation.append(
            f"Technical component: "
            f"{technical:.1f}/100 "
            f"({normalized_technical})"
        )

        if final_signal == "HOLD" and final_score > 0:
            if normalized_technical in {
                "WAIT",
                "HOLD",
                "NEUTRAL",
                "",
            }:
                explanation.append(
                    "Technical safety gate: WAIT setup "
                    "prevents a bullish final recommendation."
                )
            elif normalized_technical in {
                "WEAK SELL",
                "SELL",
                "STRONG SELL",
            }:
                explanation.append(
                    "Technical safety gate: bearish technical "
                    "direction prevents a bullish final recommendation."
                )

        elif final_signal == "HOLD" and final_score < 0:
            if normalized_technical in {
                "WEAK BUY",
                "BUY",
                "STRONG BUY",
            }:
                explanation.append(
                    "Technical safety gate: bullish technical "
                    "direction prevents a bearish final recommendation."
                )

        if rl_result is None:

            explanation.append(
                "PPO V6: unavailable"
            )

        else:

            explanation.append(
                f"PPO V6: {rl_action} "
                f"(position {rl_position:+.1f})"
            )

        explanation.append(
            f"Final weighted score: "
            f"{final_score:.1f}/100"
        )

        explanation.append(
            f"Final decision: "
            f"{final_signal}"
        )

        return HybridDecision(

            final_signal=final_signal,

            final_score=final_score,

            technical_score=technical,

            quantitative_score=quant,

            rl_score=rl_score,

            rl_action=rl_action,

            rl_position=rl_position,

            explanation=explanation,
        )