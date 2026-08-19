import yfinance as yf
from config import Config


class StockFetcher:

    def __init__(self, symbol: str):

        symbol = symbol.upper().strip()

        self.original_symbol = symbol
        self.symbol = None
        self.ticker = None

        # If user already entered an exchange
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

    def get_history(self):

        self._connect()

        try:

            return self.ticker.history(
                period=Config.HISTORY_PERIOD
            )

        except Exception:

            raise Exception(
                "Unable to fetch historical market data."
            )

    def get_info(self):

        self._connect()

        try:

            return self.ticker.info

        except Exception:

            return {}

    def get_news(self):

        self._connect()

        try:

            return self.ticker.news

        except Exception:

            return []

    def fetch_all(self):

        return {

            "history": self.get_history(),

            "info": self.get_info(),

            "news": self.get_news()

        }