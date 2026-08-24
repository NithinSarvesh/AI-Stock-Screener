"""
Static NSE sector classification for the heatmap page.

Yahoo Finance's own `sector`/`industry` fields exist but are
US-GICS-flavoured and inconsistent for NSE tickers (and cost one
extra request per stock to fetch). For a fast-loading heatmap it's
better to keep a small curated map here, grouped the way NSE's own
sectoral indices (and Kite's Markets tab) group them.

This covers the Nifty-50-ish universe already used elsewhere in the
project (see data/universe.py). Extend SECTOR_MAP as you add more
stocks to the universe — unmapped symbols fall into "Other".
"""

from typing import Dict, List

SECTOR_MAP: Dict[str, str] = {

    # Financial Services
    "HDFCBANK": "Financial Services",
    "ICICIBANK": "Financial Services",
    "SBIN": "Financial Services",
    "KOTAKBANK": "Financial Services",
    "AXISBANK": "Financial Services",
    "BAJFINANCE": "Financial Services",
    "BAJAJFINSV": "Financial Services",
    "INDUSINDBK": "Financial Services",
    "SHRIRAMFIN": "Financial Services",
    "JIOFIN": "Financial Services",
    "PNB": "Financial Services",
    "BANKBARODA": "Financial Services",
    "CANBK": "Financial Services",
    "UNIONBANK": "Financial Services",
    "IDFCFIRSTB": "Financial Services",
    "MUTHOOTFIN": "Financial Services",
    "CHOLAFIN": "Financial Services",
    "RECLTD": "Financial Services",
    "PFC": "Financial Services",
    "ICICIPRULI": "Financial Services",
    "HDFCLIFE": "Financial Services",
    "SBILIFE": "Financial Services",

    # Oil, Gas & Consumable Fuels
    "RELIANCE": "Oil, Gas & Consumable Fuels",
    "ONGC": "Oil, Gas & Consumable Fuels",
    "COALINDIA": "Oil, Gas & Consumable Fuels",
    "IOC": "Oil, Gas & Consumable Fuels",
    "BPCL": "Oil, Gas & Consumable Fuels",
    "GAIL": "Oil, Gas & Consumable Fuels",
    "HINDPETRO": "Oil, Gas & Consumable Fuels",

    # Information Technology
    "TCS": "Information Technology",
    "INFY": "Information Technology",
    "HCLTECH": "Information Technology",
    "WIPRO": "Information Technology",
    "TECHM": "Information Technology",
    "LTIM": "Information Technology",
    "PERSISTENT": "Information Technology",
    "COFORGE": "Information Technology",

    # Telecommunication
    "BHARTIARTL": "Telecommunication",
    "INDUSTOWER": "Telecommunication",

    # Automobiles
    "MARUTI": "Automobiles",
    "M&M": "Automobiles",
    "TATAMOTORS": "Automobiles",
    "EICHERMOT": "Automobiles",
    "HEROMOTOCO": "Automobiles",
    "BAJAJ-AUTO": "Automobiles",
    "TVSMOTOR": "Automobiles",
    "ASHOKLEY": "Automobiles",
    "MOTHERSON": "Automobiles",

    # Fast Moving Consumer Goods
    "HINDUNILVR": "Fast Moving Consumer Goods",
    "ITC": "Fast Moving Consumer Goods",
    "NESTLEIND": "Fast Moving Consumer Goods",
    "BRITANNIA": "Fast Moving Consumer Goods",
    "TATACONSUM": "Fast Moving Consumer Goods",
    "DABUR": "Fast Moving Consumer Goods",
    "GODREJCP": "Fast Moving Consumer Goods",
    "VBL": "Fast Moving Consumer Goods",

    # Construction
    "LT": "Construction",
    "DLF": "Construction",

    # Metals & Mining
    "TATASTEEL": "Metals & Mining",
    "JSWSTEEL": "Metals & Mining",
    "HINDALCO": "Metals & Mining",
    "VEDL": "Metals & Mining",
    "SAIL": "Metals & Mining",
    "JINDALSTEL": "Metals & Mining",

    # Power
    "NTPC": "Power",
    "POWERGRID": "Power",
    "NHPC": "Power",

    # Healthcare
    "SUNPHARMA": "Healthcare",
    "CIPLA": "Healthcare",
    "DRREDDY": "Healthcare",
    "DIVISLAB": "Healthcare",
    "APOLLOHOSP": "Healthcare",
    "TORNTPHARM": "Healthcare",
    "MAXHEALTH": "Healthcare",

    # Consumer Durables
    "TITAN": "Consumer Durables",
    "HAVELLS": "Consumer Durables",

    # Capital Goods
    "SIEMENS": "Capital Goods",
    "ABB": "Capital Goods",
    "BEL": "Capital Goods",
    "HAL": "Capital Goods",
    "BHEL": "Capital Goods",
    "CUMMINSIND": "Capital Goods",
    "POLYCAB": "Capital Goods",
    "SRF": "Capital Goods",

    # Construction Materials
    "ULTRACEMCO": "Construction Materials",
    "GRASIM": "Construction Materials",
    "PIDILITIND": "Construction Materials",
    "ASIANPAINT": "Construction Materials",

    # Services / Consumer Services
    "ADANIPORTS": "Services",
    "ADANIENT": "Services",
    "INDIGO": "Services",
    "TRENT": "Consumer Services",
    "ZOMATO": "Consumer Services",

    # Diversified / Other
    "LICI": "Diversified",
    "IRFC": "Financial Services",
}


def get_sector(symbol: str) -> str:
    symbol = symbol.strip().upper().replace(".NS", "").replace(".BO", "")
    return SECTOR_MAP.get(symbol, "Other")


def sectors_for(symbols: List[str]) -> Dict[str, str]:
    return {symbol: get_sector(symbol) for symbol in symbols}
