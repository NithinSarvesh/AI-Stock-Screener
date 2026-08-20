import yfinance as yf

from config import Config


class StockFetcher:

    def __init__(self, symbol: str):

        symbol = symbol.upper().strip()

        self.original_symbol = symbol
        self.symbol = None
        self.ticker = None

        # If user already entered exchange
        if "." in symbol:

            self.symbols_to_try = [symbol]

        else:

            # Try NSE → BSE → US
            self.symbols_to_try = [
                symbol + ".NS",
                symbol + ".BO",
                symbol
            ]

    def _connect(self):

        if self.ticker is not None:
            return

        for symbol in self.symbols_to_try:

            try:

                ticker = yf.Ticker(symbol)

                history = ticker.history(
                    period=Config.HISTORY_PERIOD
                )

                if not history.empty:

                    self.symbol = symbol
                    self.ticker = ticker
                    return

            except Exception:
                pass

        raise Exception(
            f"Stock '{self.original_symbol}' not found on NSE, BSE or US markets."
        )

    # --------------------------------------------------
    # Historical Data
    # --------------------------------------------------

    def get_history(self):

        self._connect()

        try:

            return self.ticker.history(
                period="1y",
                auto_adjust=True
            )

        except Exception:

            raise Exception(
                "Unable to fetch historical market data."
            )

    # --------------------------------------------------
    # Company Information
    # --------------------------------------------------

    def get_info(self):

        self._connect()

        try:
            info = self.ticker.info
        except Exception:
            info = {}

        try:
            fast_info = self.ticker.fast_info
        except Exception:
            fast_info = {}

        return {
            "info": info,
            "fast_info": fast_info
        }

    # --------------------------------------------------
    # News
    # --------------------------------------------------

    def get_news(self):

        self._connect()

        try:
            return self.ticker.news

        except Exception:
            return []

    # --------------------------------------------------
    # USD / INR
    # --------------------------------------------------

    def get_usd_inr_rate(self):

        try:

            usd = yf.Ticker("USDINR=X")

            history = usd.history(period="1d")

            return float(history["Close"].iloc[-1])

        except Exception:

            return 85.0

    # --------------------------------------------------
    # Fetch Everything
    # --------------------------------------------------

    def fetch_all(self):

        company = self.get_info()

        return {

            "history": self.get_history(),

            "info": company["info"],

            "fast_info": company["fast_info"],

            "news": self.get_news(),

            "usd_inr": self.get_usd_inr_rate()

        }