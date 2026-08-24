"""
Fundamentals snapshot engine.

Organizes the raw yfinance `.info` dictionary (already fetched by
StockFetcher.get_info()) into the same category groups a
fundamentals page would use: Valuation, Profitability, Per Share,
Dividends, Financial Health, Growth, Market Data.

IMPORTANT LIMITATION — please read before wiring this up:

Yahoo Finance's fundamentals coverage for NSE/BSE tickers is
noticeably thinner than for US tickers. Large caps (RELIANCE, TCS,
INFY, HDFCBANK...) usually come back reasonably complete; smaller
names often return None for half of these fields. This engine
reports "N/A" rather than guessing or fabricating a number, and
`coverage_ratio()` tells you, per stock, what fraction of fields
Yahoo actually returned — render that so users know how much to
trust the snapshot for a given symbol.

If you need investor-grade Indian fundamentals (quarterly results,
shareholding pattern, promoter pledge %, corporate actions,
concalls), Yahoo Finance is not a reliable source for that. That
needs NSE's own data (nsepython / nselib) or a paid vendor
(Screener.in has an API, Tickertape, Finology). That is a separate,
larger project — not something to bolt on today.
"""

from typing import Any, Dict, Optional


def _fmt_number(value, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_percent(value, already_fraction: bool = True, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if already_fraction:
            v *= 100
        return f"{v:,.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_large_number(value, is_indian: bool = True) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if is_indian:
        if abs(value) >= 1e7:
            return f"₹ {value / 1e7:,.2f} Cr"
        return f"₹ {value:,.0f}"

    if abs(value) >= 1e12:
        return f"$ {value / 1e12:,.2f} T"
    if abs(value) >= 1e9:
        return f"$ {value / 1e9:,.2f} B"
    return f"$ {value:,.0f}"


class FundamentalsEngine:

    def __init__(self, info: Dict[str, Any], is_indian: bool = True):
        self.info = info or {}
        self.is_indian = is_indian

    def _get(self, *keys, default=None):
        for key in keys:
            value = self.info.get(key)
            if value is not None:
                return value
        return default

    # =========================================================
    # CATEGORY GROUPS
    # =========================================================

    def valuation(self) -> Dict[str, str]:
        return {
            "Trailing P/E": _fmt_number(self._get("trailingPE")),
            "Forward P/E": _fmt_number(self._get("forwardPE")),
            "Price / Book": _fmt_number(self._get("priceToBook")),
            "Price / Sales (TTM)": _fmt_number(self._get("priceToSalesTrailing12Months")),
            "EV / EBITDA": _fmt_number(self._get("enterpriseToEbitda")),
            "EV / Revenue": _fmt_number(self._get("enterpriseToRevenue")),
            "PEG Ratio": _fmt_number(self._get("pegRatio", "trailingPegRatio")),
        }

    def profitability(self) -> Dict[str, str]:
        return {
            "Return on Equity": _fmt_percent(self._get("returnOnEquity")),
            "Return on Assets": _fmt_percent(self._get("returnOnAssets")),
            "Gross Margin": _fmt_percent(self._get("grossMargins")),
            "Operating Margin": _fmt_percent(self._get("operatingMargins")),
            "Net Profit Margin": _fmt_percent(self._get("profitMargins")),
        }

    def per_share(self) -> Dict[str, str]:
        return {
            "EPS (TTM)": _fmt_number(self._get("trailingEps")),
            "EPS (Forward)": _fmt_number(self._get("forwardEps")),
            "Book Value / Share": _fmt_number(self._get("bookValue")),
            "Revenue / Share": _fmt_number(self._get("revenuePerShare")),
        }

    def dividends(self) -> Dict[str, str]:
        return {
            "Dividend Yield": _fmt_percent(self._get("dividendYield")),
            "Dividend Rate": _fmt_number(self._get("dividendRate")),
            "Payout Ratio": _fmt_percent(self._get("payoutRatio")),
            "5Y Avg Dividend Yield": _fmt_percent(
                self._get("fiveYearAvgDividendYield"), already_fraction=False
            ),
        }

    def financial_health(self) -> Dict[str, str]:
        return {
            "Debt / Equity": _fmt_number(self._get("debtToEquity")),
            "Current Ratio": _fmt_number(self._get("currentRatio")),
            "Quick Ratio": _fmt_number(self._get("quickRatio")),
            "Total Cash": _fmt_large_number(self._get("totalCash"), self.is_indian),
            "Total Debt": _fmt_large_number(self._get("totalDebt"), self.is_indian),
        }

    def growth(self) -> Dict[str, str]:
        return {
            "Revenue Growth (YoY)": _fmt_percent(self._get("revenueGrowth")),
            "Earnings Growth (YoY)": _fmt_percent(self._get("earningsGrowth")),
            "Quarterly Earnings Growth": _fmt_percent(self._get("earningsQuarterlyGrowth")),
        }

    def market_data(self) -> Dict[str, str]:
        face_value = self._get("faceValue")
        return {
            "Market Cap": _fmt_large_number(self._get("marketCap"), self.is_indian),
            "Beta": _fmt_number(self._get("beta")),
            "52 Week High": _fmt_number(self._get("fiftyTwoWeekHigh")),
            "52 Week Low": _fmt_number(self._get("fiftyTwoWeekLow")),
            "Avg Volume (10d)": _fmt_number(self._get("averageDailyVolume10Day"), decimals=0),
            "Shares Outstanding": _fmt_number(self._get("sharesOutstanding"), decimals=0),
            "Face Value": _fmt_number(face_value) if face_value else "N/A",
        }

    # =========================================================
    # DATA QUALITY DIAGNOSTIC
    # =========================================================

    def coverage_ratio(self) -> float:
        """
        Fraction of fields above that Yahoo actually returned a
        value for. Display this next to the snapshot so the user
        knows how much to trust it for this particular symbol.
        """

        groups = [
            self.valuation(), self.profitability(), self.per_share(),
            self.dividends(), self.financial_health(), self.growth(),
            self.market_data(),
        ]

        total = 0
        filled = 0

        for group in groups:
            for value in group.values():
                total += 1
                if value != "N/A":
                    filled += 1

        return round(filled / total, 2) if total else 0.0

    def snapshot(self) -> Dict[str, Dict[str, str]]:
        return {
            "Valuation": self.valuation(),
            "Profitability": self.profitability(),
            "Per Share": self.per_share(),
            "Dividends": self.dividends(),
            "Financial Health": self.financial_health(),
            "Growth": self.growth(),
            "Market Data": self.market_data(),
        }
