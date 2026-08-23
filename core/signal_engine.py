from typing import Any, Dict, List, Optional


class SignalEngine:
    """
    Unified quantitative signal engine.

    This engine is the single source of truth for:

    - Quantitative score
    - Trend bias
    - Momentum bias
    - Volume bias
    - Structure bias
    - Pattern bias
    - Final BUY / WAIT / SELL signal
    - Trade setup
    - Signal conflicts

    IMPORTANT:
    The engine intentionally separates:

        SCORE
        from
        FINAL SIGNAL

    A stock can have a reasonably high score but still produce
    WAIT when the underlying evidence is contradictory.
    """

    def __init__(
        self,
        latest,
        levels: Optional[Dict[str, Any]] = None,
        pattern: Optional[Dict[str, Any]] = None,
    ):
        self.latest = latest
        self.levels = levels or {}
        self.pattern = pattern or {}

    # =========================================================
    # SAFE VALUE HELPERS
    # =========================================================

    def _value(
        self,
        column: str,
        default: float = 0.0,
    ) -> float:

        try:
            value = self.latest[column]

            if value is None:
                return default

            value = float(value)

            if value != value:  # NaN
                return default

            return value

        except Exception:
            return default

    def _text(
        self,
        value,
        default: str = "",
    ) -> str:

        if value is None:
            return default

        return str(value).strip()

    # =========================================================
    # TREND ANALYSIS
    # =========================================================

    def _trend_analysis(self) -> Dict[str, Any]:

        close = self._value("Close")

        ema20 = self._value("EMA20")
        ema50 = self._value("EMA50")
        ema200 = self._value("EMA200")

        adx = self._value("ADX")

        score = 0
        reasons: List[str] = []

        # -----------------------------------------------------
        # EMA structure
        # -----------------------------------------------------

        if (
            ema20 > ema50
            and ema50 > ema200
        ):

            score += 15

            bias = "bullish"

            reasons.append(
                "🟢 EMA structure is bullish: "
                "EMA20 > EMA50 > EMA200."
            )

        elif (
            ema20 < ema50
            and ema50 < ema200
        ):

            score += 0

            bias = "bearish"

            reasons.append(
                "🔴 EMA structure is bearish: "
                "EMA20 < EMA50 < EMA200."
            )

        else:

            score += 7

            bias = "neutral"

            reasons.append(
                "🟡 EMA structure is mixed."
            )

        # -----------------------------------------------------
        # Price vs EMA200
        # -----------------------------------------------------

        if close > ema200:

            score += 8

            if bias == "bearish":

                reasons.append(
                    "🟡 Price is above EMA200 despite "
                    "the bearish EMA structure."
                )

            else:

                reasons.append(
                    "🟢 Price is above EMA200."
                )

        else:

            score += 0

            if bias == "bullish":

                reasons.append(
                    "🟡 Price is below EMA200 despite "
                    "the bullish short-term EMA structure."
                )

            else:

                reasons.append(
                    "🔴 Price is below EMA200."
                )

        # -----------------------------------------------------
        # ADX
        # -----------------------------------------------------

        if adx >= 25:

            score += 7

            reasons.append(
                f"🟢 ADX {adx:.1f} confirms a meaningful trend."
            )

        elif adx >= 20:

            score += 4

            reasons.append(
                f"🟡 ADX {adx:.1f} indicates a developing trend."
            )

        else:

            score += 2

            reasons.append(
                f"🟡 ADX {adx:.1f} indicates a weak trend."
            )

        return {
            "score": score,
            "max_score": 30,
            "bias": bias,
            "adx": adx,
            "reasons": reasons,
        }

    # =========================================================
    # MOMENTUM ANALYSIS
    # =========================================================

    def _momentum_analysis(self) -> Dict[str, Any]:

        rsi = self._value("RSI")

        macd = self._value("MACD")
        macd_signal = self._value("MACD_SIGNAL")

        stoch = self._value("STOCH_RSI")

        score = 0
        reasons: List[str] = []

        bullish_points = 0
        bearish_points = 0

        # -----------------------------------------------------
        # RSI
        # -----------------------------------------------------

        if 50 <= rsi < 65:

            score += 10
            bullish_points += 2

            reasons.append(
                f"🟢 RSI {rsi:.1f} shows healthy bullish momentum."
            )

        elif 45 <= rsi < 50:

            score += 7
            bullish_points += 1

            reasons.append(
                f"🟡 RSI {rsi:.1f} is neutral-to-bullish."
            )

        elif 65 <= rsi < 70:

            score += 7
            bullish_points += 1

            reasons.append(
                f"🟡 RSI {rsi:.1f} is bullish but becoming extended."
            )

        elif 30 <= rsi < 45:

            score += 4
            bearish_points += 1

            reasons.append(
                f"🟡 RSI {rsi:.1f} indicates weak momentum."
            )

        elif rsi >= 70:

            score += 2
            bearish_points += 2

            reasons.append(
                f"🔴 RSI {rsi:.1f} is overbought."
            )

        else:

            score += 3
            bullish_points += 1

            reasons.append(
                f"🟡 RSI {rsi:.1f} is oversold."
            )

        # -----------------------------------------------------
        # MACD
        # -----------------------------------------------------

        if macd > macd_signal:

            score += 10
            bullish_points += 2

            reasons.append(
                "🟢 MACD is above its signal line."
            )

        elif macd < macd_signal:

            score += 0
            bearish_points += 2

            reasons.append(
                "🔴 MACD is below its signal line."
            )

        else:

            score += 5

            reasons.append(
                "🟡 MACD is near its signal line."
            )

        # -----------------------------------------------------
        # Stochastic RSI
        # -----------------------------------------------------

        if 20 <= stoch <= 80:

            score += 5

            reasons.append(
                f"🟢 Stoch RSI {stoch:.1f} is in a usable range."
            )

        elif stoch > 80:

            score += 2
            bearish_points += 1

            reasons.append(
                f"🟡 Stoch RSI {stoch:.1f} is overbought."
            )

        else:

            score += 4
            bullish_points += 1

            reasons.append(
                f"🟡 Stoch RSI {stoch:.1f} is oversold."
            )

        # -----------------------------------------------------
        # Determine momentum bias
        # -----------------------------------------------------

        if bullish_points >= bearish_points + 2:

            bias = "bullish"

        elif bearish_points >= bullish_points + 2:

            bias = "bearish"

        else:

            bias = "neutral"

        return {
            "score": score,
            "max_score": 25,
            "bias": bias,
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "stoch_rsi": stoch,
            "reasons": reasons,
        }

    # =========================================================
    # VOLUME ANALYSIS
    # =========================================================

    def _volume_analysis(self) -> Dict[str, Any]:

        close = self._value("Close")

        vwap = self._value("VWAP")

        volume = self._value("Volume")
        volume_avg = self._value("Volume_Avg")

        score = 0
        reasons: List[str] = []

        bullish_points = 0
        bearish_points = 0

        # -----------------------------------------------------
        # VWAP
        # -----------------------------------------------------

        if close > vwap:

            score += 8
            bullish_points += 1

            reasons.append(
                "🟢 Price is above VWAP."
            )

        else:

            score += 2
            bearish_points += 1

            reasons.append(
                "🔴 Price is below VWAP."
            )

        # -----------------------------------------------------
        # Volume
        # -----------------------------------------------------

        if volume_avg > 0:

            volume_ratio = (
                volume / volume_avg
            )

        else:

            volume_ratio = 1.0

        if volume_ratio >= 1.5:

            score += 7

            if close > vwap:

                bullish_points += 2

                reasons.append(
                    f"🟢 Strong volume confirmation: "
                    f"{volume_ratio:.2f}× average."
                )

            else:

                bearish_points += 2

                reasons.append(
                    f"🔴 Strong volume while below VWAP: "
                    f"{volume_ratio:.2f}× average."
                )

        elif volume_ratio >= 1.0:

            score += 5

            reasons.append(
                f"🟡 Volume is around average: "
                f"{volume_ratio:.2f}×."
            )

        else:

            score += 2

            reasons.append(
                f"🟡 Volume is below average: "
                f"{volume_ratio:.2f}×."
            )

        if bullish_points >= bearish_points + 1:

            bias = "bullish"

        elif bearish_points >= bullish_points + 1:

            bias = "bearish"

        else:

            bias = "neutral"

        return {
            "score": score,
            "max_score": 15,
            "bias": bias,
            "volume_ratio": volume_ratio,
            "reasons": reasons,
        }

    # =========================================================
    # STRUCTURE ANALYSIS
    # =========================================================

    def _structure_analysis(self) -> Dict[str, Any]:

        close = self._value("Close")

        support = self.levels.get(
            "support",
            close,
        )

        resistance = self.levels.get(
            "resistance",
            close,
        )

        try:
            support = float(support)
        except Exception:
            support = close

        try:
            resistance = float(resistance)
        except Exception:
            resistance = close

        score = 0
        reasons: List[str] = []

        if resistance > support:

            total_range = (
                resistance - support
            )

            position = (
                close - support
            ) / total_range

        else:

            position = 0.5

        position = max(
            0.0,
            min(1.0, position),
        )

        # -----------------------------------------------------
        # Price location
        # -----------------------------------------------------

        if position >= 0.65:

            score += 10

            bias = "bullish"

            reasons.append(
                "🟢 Price is positioned in the upper part "
                "of the current support/resistance range."
            )

        elif position <= 0.35:

            score += 3

            bias = "bearish"

            reasons.append(
                "🔴 Price is positioned near the lower part "
                "of the current support/resistance range."
            )

        else:

            score += 6

            bias = "neutral"

            reasons.append(
                "🟡 Price is positioned in the middle of "
                "the current technical range."
            )

        # -----------------------------------------------------
        # Distance from resistance
        # -----------------------------------------------------

        if resistance > close:

            resistance_distance = (
                resistance - close
            ) / close

        else:

            resistance_distance = 0.0

        if (
            resistance_distance > 0
            and resistance_distance <= 0.02
        ):

            score += 2

            reasons.append(
                "🟡 Price is very close to resistance; "
                "breakout confirmation is important."
            )

        else:

            score += 5

        return {
            "score": score,
            "max_score": 15,
            "bias": bias,
            "support": support,
            "resistance": resistance,
            "range_position": position,
            "reasons": reasons,
        }

    # =========================================================
    # PATTERN ANALYSIS
    # =========================================================

    def _pattern_analysis(self) -> Dict[str, Any]:

        pattern_name = self._text(
            self.pattern.get(
                "pattern",
                "None",
            ),
            "None",
        )

        confidence = self.pattern.get(
            "confidence",
            0,
        )

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0

        bullish_patterns = {
            "Hammer",
            "Bullish Engulfing",
            "Morning Star",
            "Piercing Line",
            "Bullish Harami",
            "Three White Soldiers",
        }

        bearish_patterns = {
            "Shooting Star",
            "Bearish Engulfing",
            "Evening Star",
            "Dark Cloud Cover",
            "Bearish Harami",
            "Three Black Crows",
        }

        if pattern_name in bullish_patterns:

            bias = "bullish"

            score = min(
                10,
                5 + confidence / 20,
            )

            reason = (
                f"🟢 {pattern_name} is a bullish "
                f"candlestick pattern."
            )

        elif pattern_name in bearish_patterns:

            bias = "bearish"

            score = min(
                10,
                5 + confidence / 20,
            )

            reason = (
                f"🔴 {pattern_name} is a bearish "
                f"candlestick pattern."
            )

        else:

            bias = "neutral"

            score = 5

            reason = (
                f"🟡 No strong directional candlestick "
                f"pattern detected ({pattern_name})."
            )

        return {
            "score": round(score, 2),
            "max_score": 10,
            "bias": bias,
            "pattern": pattern_name,
            "confidence": confidence,
            "reasons": [reason],
        }

    # =========================================================
    # VOLATILITY ANALYSIS
    # =========================================================

    def _volatility_analysis(self) -> Dict[str, Any]:

        close = self._value("Close")

        atr = self._value("ATR")

        if close > 0:

            atr_percent = (
                atr / close
            ) * 100

        else:

            atr_percent = 0

        # -----------------------------------------------------
        # We don't reward extreme volatility.
        #
        # The goal is not "low volatility = good".
        # The goal is "manageable volatility = better".
        # -----------------------------------------------------

        if atr_percent <= 2:

            score = 5

            regime = "low"

            reason = (
                f"🟢 ATR is {atr_percent:.2f}% of price; "
                "volatility is relatively manageable."
            )

        elif atr_percent <= 4:

            score = 4

            regime = "normal"

            reason = (
                f"🟡 ATR is {atr_percent:.2f}% of price; "
                "volatility is normal."
            )

        elif atr_percent <= 7:

            score = 3

            regime = "elevated"

            reason = (
                f"🟡 ATR is {atr_percent:.2f}% of price; "
                "volatility is elevated."
            )

        else:

            score = 1

            regime = "high"

            reason = (
                f"🔴 ATR is {atr_percent:.2f}% of price; "
                "volatility is high."
            )

        return {
            "score": score,
            "max_score": 5,
            "bias": "neutral",
            "atr": atr,
            "atr_percent": atr_percent,
            "regime": regime,
            "reasons": [reason],
        }

    # =========================================================
    # FINAL SIGNAL
    # =========================================================

    def _determine_final_signal(
        self,
        total_score: float,
        trend_bias: str,
        momentum_bias: str,
        volume_bias: str,
        structure_bias: str,
        pattern_bias: str,
    ) -> Dict[str, Any]:

        bullish_count = sum(
            bias == "bullish"
            for bias in [
                trend_bias,
                momentum_bias,
                volume_bias,
                structure_bias,
                pattern_bias,
            ]
        )

        bearish_count = sum(
            bias == "bearish"
            for bias in [
                trend_bias,
                momentum_bias,
                volume_bias,
                structure_bias,
                pattern_bias,
            ]
        )

        conflicts: List[str] = []

        # -----------------------------------------------------
        # Major trend / momentum conflict
        # -----------------------------------------------------

        if (
            trend_bias == "bearish"
            and momentum_bias == "bullish"
        ):

            conflicts.append(
                "Primary trend is bearish while momentum "
                "is bullish."
            )

        if (
            trend_bias == "bullish"
            and momentum_bias == "bearish"
        ):

            conflicts.append(
                "Primary trend is bullish while momentum "
                "is bearish."
            )

        # -----------------------------------------------------
        # Trend / volume conflict
        # -----------------------------------------------------

        if (
            trend_bias == "bearish"
            and volume_bias == "bullish"
        ):

            conflicts.append(
                "Trend is bearish while price/volume "
                "conditions are improving."
            )

        if (
            trend_bias == "bullish"
            and volume_bias == "bearish"
        ):

            conflicts.append(
                "Trend is bullish while price/volume "
                "conditions are weakening."
            )

        # -----------------------------------------------------
        # Final decision
        # -----------------------------------------------------

        if (
            total_score >= 80
            and trend_bias == "bullish"
            and momentum_bias == "bullish"
            and bullish_count >= 4
        ):

            signal = "STRONG BUY"

        elif (
            total_score >= 65
            and trend_bias == "bullish"
            and momentum_bias != "bearish"
            and bullish_count >= 3
        ):

            signal = "BUY"

        elif (
            total_score <= 25
            and trend_bias == "bearish"
            and momentum_bias == "bearish"
            and bearish_count >= 4
        ):

            signal = "STRONG SELL"

        elif (
            total_score <= 35
            and trend_bias == "bearish"
            and momentum_bias != "bullish"
            and bearish_count >= 3
        ):

            signal = "SELL"

        else:

            signal = "WAIT"

        # -----------------------------------------------------
        # Confidence
        # -----------------------------------------------------

        dominant_count = max(
            bullish_count,
            bearish_count,
        )

        confidence = 50 + (
            dominant_count * 8
        )

        if conflicts:

            confidence -= (
                len(conflicts) * 8
            )

        confidence = max(
            35,
            min(95, confidence),
        )

        return {
            "signal": signal,
            "confidence": round(
                confidence
            ),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "conflicts": conflicts,
        }

    # =========================================================
    # TRADE SETUP
    # =========================================================

    def _build_trade_setup(
        self,
        signal: str,
    ) -> Dict[str, Any]:

        close = self._value("Close")

        atr = self._value(
            "ATR",
            close * 0.02,
        )

        support = self.levels.get(
            "support",
            close - atr * 1.5,
        )

        resistance = self.levels.get(
            "resistance",
            close + atr * 1.5,
        )

        try:
            support = float(support)
        except Exception:
            support = close - atr * 1.5

        try:
            resistance = float(resistance)
        except Exception:
            resistance = close + atr * 1.5

        # -----------------------------------------------------
        # No active trade for WAIT
        # -----------------------------------------------------

        if signal == "WAIT":

            return {
                "valid": False,
                "entry": None,
                "stop_loss": None,
                "target1": None,
                "target2": None,
                "target3": None,
                "risk_per_share": None,
                "risk_reward_target1": None,
                "reason": (
                    "No active trade setup. "
                    "Wait for signal confirmation."
                ),
            }

        # -----------------------------------------------------
        # BUY setup
        # -----------------------------------------------------

        if "BUY" in signal:

            entry = close

            # Use support if it is actually below price.
            if support < entry:

                stop_loss = max(
                    support,
                    entry - (1.5 * atr),
                )

            else:

                stop_loss = (
                    entry - (1.5 * atr)
                )

            risk = entry - stop_loss

            if risk <= 0:

                risk = max(
                    atr,
                    entry * 0.01,
                )

                stop_loss = (
                    entry - risk
                )

            target1 = entry + (
                risk * 1.5
            )

            target2 = entry + (
                risk * 2.5
            )

            target3 = entry + (
                risk * 3.5
            )

            rr = (
                target1 - entry
            ) / risk

            return {
                "valid": True,
                "side": "LONG",
                "entry": entry,
                "stop_loss": stop_loss,
                "target1": target1,
                "target2": target2,
                "target3": target3,
                "risk_per_share": risk,
                "risk_reward_target1": rr,
                "reason": (
                    "Bullish setup confirmed by the "
                    "unified signal engine."
                ),
            }

        # -----------------------------------------------------
        # SELL setup
        # -----------------------------------------------------

        entry = close

        if resistance > entry:

            stop_loss = min(
                resistance,
                entry + (1.5 * atr),
            )

        else:

            stop_loss = (
                entry + (1.5 * atr)
            )

        risk = stop_loss - entry

        if risk <= 0:

            risk = max(
                atr,
                entry * 0.01,
            )

            stop_loss = (
                entry + risk
            )

        target1 = entry - (
            risk * 1.5
        )

        target2 = entry - (
            risk * 2.5
        )

        target3 = entry - (
            risk * 3.5
        )

        rr = (
            entry - target1
        ) / risk

        return {
            "valid": True,
            "side": "SHORT",
            "entry": entry,
            "stop_loss": stop_loss,
            "target1": target1,
            "target2": target2,
            "target3": target3,
            "risk_per_share": risk,
            "risk_reward_target1": rr,
            "reason": (
                "Bearish setup confirmed by the "
                "unified signal engine."
            ),
        }

    # =========================================================
    # PUBLIC METHOD
    # =========================================================

    def analyze(self) -> Dict[str, Any]:

        trend = self._trend_analysis()

        momentum = self._momentum_analysis()

        volume = self._volume_analysis()

        structure = self._structure_analysis()

        pattern = self._pattern_analysis()

        volatility = self._volatility_analysis()

        total_score = (
            trend["score"]
            + momentum["score"]
            + volume["score"]
            + structure["score"]
            + pattern["score"]
            + volatility["score"]
        )

        max_score = (
            trend["max_score"]
            + momentum["max_score"]
            + volume["max_score"]
            + structure["max_score"]
            + pattern["max_score"]
            + volatility["max_score"]
        )

        normalized_score = (
            total_score / max_score
        ) * 100

        normalized_score = round(
            normalized_score,
            1,
        )

        final = self._determine_final_signal(
            normalized_score,
            trend["bias"],
            momentum["bias"],
            volume["bias"],
            structure["bias"],
            pattern["bias"],
        )

        setup = self._build_trade_setup(
            final["signal"]
        )

        all_reasons = []

        all_reasons.extend(
            trend["reasons"]
        )

        all_reasons.extend(
            momentum["reasons"]
        )

        all_reasons.extend(
            volume["reasons"]
        )

        all_reasons.extend(
            structure["reasons"]
        )

        all_reasons.extend(
            pattern["reasons"]
        )

        all_reasons.extend(
            volatility["reasons"]
        )

        return {
            "score": normalized_score,
            "signal": final["signal"],
            "confidence": final["confidence"],

            "trend": trend,
            "momentum": momentum,
            "volume": volume,
            "structure": structure,
            "pattern": pattern,
            "volatility": volatility,

            "bullish_count": final[
                "bullish_count"
            ],

            "bearish_count": final[
                "bearish_count"
            ],

            "conflicts": final[
                "conflicts"
            ],

            "reasons": all_reasons,

            "setup": setup,
        }