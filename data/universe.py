"""
Dynamic Indian equity universe.

Builds a clean list of liquid NSE stocks suitable for
machine-learning research.

This module does NOT train the PPO model.
It only decides which stocks are eligible for training.
"""

from __future__ import annotations

import time
from typing import List

import yfinance as yf


# ---------------------------------------------------------------------
# INITIAL RESEARCH UNIVERSE
# ---------------------------------------------------------------------

# Start with a controlled liquid universe.
#
# We deliberately do NOT start with every NSE security.
# The first objective is a reliable dataset, not maximum quantity.

NIFTY_100 = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "BHARTIARTL",
    "ICICIBANK",
    "INFY",
    "SBIN",
    "LICI",
    "HINDUNILVR",
    "ITC",
    "L&T",
    "BAJFINANCE",
    "HCLTECH",
    "MARUTI",
    "SUNPHARMA",
    "KOTAKBANK",
    "M&M",
    "AXISBANK",
    "ULTRACEMCO",
    "NTPC",
    "TITAN",
    "ADANIENT",
    "ONGC",
    "WIPRO",
    "TATASTEEL",
    "POWERGRID",
    "NESTLEIND",
    "COALINDIA",
    "ASIANPAINT",
    "JSWSTEEL",
    "BAJAJFINSV",
    "ADANIPORTS",
    "TECHM",
    "HINDALCO",
    "GRASIM",
    "BEL",
    "TRENT",
    "HDFCLIFE",
    "SBILIFE",
    "EICHERMOT",
    "CIPLA",
    "TATAMOTORS",
    "DRREDDY",
    "DIVISLAB",
    "APOLLOHOSP",
    "BRITANNIA",
    "INDUSINDBK",
    "HEROMOTOCO",
    "BPCL",
    "TATA2",
    "SHRIRAMFIN",
    "BAJAJ-AUTO",
    "TATACONSUM",
    "HAVELLS",
    "DABUR",
    "PIDILITIND",
    "GODREJCP",
    "TORNTPHARM",
    "ICICIPRULI",
    "INDIGO",
    "SIEMENS",
    "ABB",
    "DLF",
    "VBL",
    "ZOMATO",
    "JIOFIN",
    "MAXHEALTH",
    "INDIANB",
    "PNB",
    "BANKBARODA",
    "CANBK",
    "UNIONBANK",
    "IDFCFIRSTB",
    "RECLTD",
    "PFC",
    "IOC",
    "GAIL",
    "VEDL",
    "HINDPETRO",
    "SAIL",
    "JINDALSTEL",
    "NHPC",
    "IRFC",
    "HAL",
    "BHEL",
    "BEL",
    "INDUSTOWER",
    "DLF",
    "MOTHERSON",
    "TVSMOTOR",
    "ASHOKLEY",
    "MUTHOOTFIN",
    "CHOLAFIN",
    "POLYCAB",
    "SRF",
    "CUMMINSIND",
    "LTIM",
    "PERSISTENT",
    "COFORGE",
]


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

MIN_ROWS = 900

# Number of stocks we initially want to train on.
TARGET_STOCK_COUNT = 100

# Pause between downloads to avoid hammering the data provider.
DOWNLOAD_DELAY = 0.25


# ---------------------------------------------------------------------
# SYMBOL NORMALIZATION
# ---------------------------------------------------------------------

def yahoo_symbol(symbol: str) -> str:
    """
    Convert an NSE symbol to Yahoo Finance format.
    """

    symbol = symbol.strip().upper()

    # Yahoo uses LT.NS rather than L&T.NS
    replacements = {
        "L&T": "LT",
        "M&M": "M&M",
        "TATA2": "TATACHEM",
    }

    symbol = replacements.get(symbol, symbol)

    return f"{symbol}.NS"


# ---------------------------------------------------------------------
# HISTORY CHECK
# ---------------------------------------------------------------------

def has_sufficient_history(
    symbol: str,
    period: str = "5y",
    min_rows: int = MIN_ROWS,
) -> bool:

    ticker = yahoo_symbol(symbol)

    try:

        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return False

        # Handle MultiIndex columns returned by some yfinance versions.
        if hasattr(df.columns, "levels") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(
            subset=["Close"]
        )

        return len(df) >= min_rows

    except Exception:

        return False


# ---------------------------------------------------------------------
# BUILD UNIVERSE
# ---------------------------------------------------------------------

def get_training_universe(
    target_count: int = TARGET_STOCK_COUNT,
) -> List[str]:

    print("=" * 70)
    print("BUILDING INDIAN EQUITY TRAINING UNIVERSE")
    print("=" * 70)

    selected = []

    # Remove duplicates while preserving order.
    candidates = list(
        dict.fromkeys(NIFTY_100)
    )

    print(
        f"Candidate stocks: {len(candidates)}"
    )

    for index, symbol in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"[{index:03d}/{len(candidates):03d}] "
            f"Checking {symbol}...",
            end=" ",
        )

        valid = has_sufficient_history(
            symbol
        )

        if valid:

            selected.append(symbol)

            print(
                f"OK ({len(selected)}/{target_count})"
            )

        else:

            print("REJECT")

        if len(selected) >= target_count:
            break

        time.sleep(
            DOWNLOAD_DELAY
        )

    print()
    print("=" * 70)
    print(
        f"FINAL UNIVERSE: {len(selected)} stocks"
    )
    print("=" * 70)

    for symbol in selected:
        print(
            f"{symbol} -> {yahoo_symbol(symbol)}"
        )

    return selected


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":

    universe = get_training_universe()

    print()
    print("Training universe:")
    print(universe)