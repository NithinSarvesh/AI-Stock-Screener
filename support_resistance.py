import pandas as pd


class SupportResistance:

    def __init__(self, history: pd.DataFrame):

        self.history = history.copy()

    def calculate(self):

        recent = self.history.tail(30)

        support = recent["Low"].min()

        resistance = recent["High"].max()

        return {
            "support": support,
            "resistance": resistance
        }