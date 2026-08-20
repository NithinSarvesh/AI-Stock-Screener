import ta
import pandas as pd

from config import Config


class IndicatorEngine:

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    # --------------------------------------------------
    # EMA
    # --------------------------------------------------

    def calculate_ema(self):

        self.df["EMA20"] = ta.trend.EMAIndicator(
            close=self.df["Close"],
            window=Config.FAST_EMA
        ).ema_indicator()

        self.df["EMA50"] = ta.trend.EMAIndicator(
            close=self.df["Close"],
            window=Config.MID_EMA
        ).ema_indicator()

        self.df["EMA200"] = ta.trend.EMAIndicator(
            close=self.df["Close"],
            window=Config.LONG_EMA
        ).ema_indicator()

    # --------------------------------------------------
    # RSI
    # --------------------------------------------------

    def calculate_rsi(self):

        self.df["RSI"] = ta.momentum.RSIIndicator(
            close=self.df["Close"],
            window=Config.RSI_PERIOD
        ).rsi()

    # --------------------------------------------------
    # MACD
    # --------------------------------------------------

    def calculate_macd(self):

        macd = ta.trend.MACD(
            close=self.df["Close"],
            window_fast=Config.MACD_FAST,
            window_slow=Config.MACD_SLOW,
            window_sign=Config.MACD_SIGNAL
        )

        self.df["MACD"] = macd.macd()
        self.df["MACD_SIGNAL"] = macd.macd_signal()
        self.df["MACD_HISTOGRAM"] = macd.macd_diff()

    # --------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------

    def calculate_bollinger(self):

        bb = ta.volatility.BollingerBands(
            close=self.df["Close"],
            window=Config.BOLLINGER_PERIOD
        )

        self.df["BB_UPPER"] = bb.bollinger_hband()
        self.df["BB_MIDDLE"] = bb.bollinger_mavg()
        self.df["BB_LOWER"] = bb.bollinger_lband()

    # --------------------------------------------------
    # Volume Indicators
    # --------------------------------------------------

    def calculate_volume(self):

        self.df["VWAP"] = ta.volume.VolumeWeightedAveragePrice(
            high=self.df["High"],
            low=self.df["Low"],
            close=self.df["Close"],
            volume=self.df["Volume"]
        ).volume_weighted_average_price()

        self.df["AVG_VOLUME"] = (
            self.df["Volume"]
            .rolling(window=20)
            .mean()
        )

    # --------------------------------------------------
    # ATR
    # --------------------------------------------------

    def calculate_atr(self):

        atr = ta.volatility.AverageTrueRange(
            high=self.df["High"],
            low=self.df["Low"],
            close=self.df["Close"]
        )

        self.df["ATR"] = atr.average_true_range()

    # --------------------------------------------------
    # ADX
    # --------------------------------------------------

    def calculate_adx(self):

        adx = ta.trend.ADXIndicator(
            high=self.df["High"],
            low=self.df["Low"],
            close=self.df["Close"]
        )

        self.df["ADX"] = adx.adx()

    # --------------------------------------------------
    # OBV
    # --------------------------------------------------

    def calculate_obv(self):

        obv = ta.volume.OnBalanceVolumeIndicator(
            close=self.df["Close"],
            volume=self.df["Volume"]
        )

        self.df["OBV"] = obv.on_balance_volume()

    # --------------------------------------------------
    # Stochastic RSI
    # --------------------------------------------------

    def calculate_stoch_rsi(self):

        stoch = ta.momentum.StochRSIIndicator(
            close=self.df["Close"]
        )

        self.df["STOCH_RSI"] = stoch.stochrsi()

    # --------------------------------------------------
    # Calculate Everything
    # --------------------------------------------------

    def calculate_all(self):

        self.calculate_ema()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger()
        self.calculate_volume()
        self.calculate_atr()
        self.calculate_adx()
        self.calculate_obv()
        self.calculate_stoch_rsi()

        return self.df