import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ChartBuilder:

    def __init__(self, df):
        self.df = df

    def create_dashboard(self):

        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.50, 0.15, 0.20, 0.15],
            subplot_titles=(
                "Price",
                "Volume",
                "RSI",
                "MACD"
            )
        )

        # ----------------------------------------------------
        # Candlestick
        # ----------------------------------------------------

        fig.add_trace(

            go.Candlestick(

                x=self.df.index,

                open=self.df["Open"],

                high=self.df["High"],

                low=self.df["Low"],

                close=self.df["Close"],

                name="Price"

            ),

            row=1,
            col=1
        )

        # ----------------------------------------------------
        # EMA20
        # ----------------------------------------------------

        fig.add_trace(

            go.Scatter(

                x=self.df.index,

                y=self.df["EMA20"],

                mode="lines",

                name="EMA20",

                line=dict(width=2)

            ),

            row=1,
            col=1

        )

        # ----------------------------------------------------

        fig.add_trace(

            go.Scatter(

                x=self.df.index,

                y=self.df["EMA50"],

                mode="lines",

                name="EMA50",

                line=dict(width=2)

            ),

            row=1,
            col=1

        )

        # ----------------------------------------------------

        fig.add_trace(

            go.Scatter(

                x=self.df.index,

                y=self.df["EMA200"],

                mode="lines",

                name="EMA200",

                line=dict(width=2)

            ),

            row=1,
            col=1

        )

        # ----------------------------------------------------
        # Bollinger Bands
        # ----------------------------------------------------

        fig.add_trace(

            go.Scatter(

                x=self.df.index,

                y=self.df["BB_UPPER"],

                name="Upper BB",

                line=dict(dash="dot")

            ),

            row=1,
            col=1

        )

        fig.add_trace(

            go.Scatter(

                x=self.df.index,

                y=self.df["BB_LOWER"],

                name="Lower BB",

                line=dict(dash="dot")

            ),

            row=1,
            col=1

        )

        # ----------------------------------------------------
        # Volume
        # ----------------------------------------------------

        fig.add_trace(

            go.Bar(

                x=self.df.index,

                y=self.df["Volume"],

                name="Volume"

            ),

            row=2,
            col=1

        )

        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        fig.add_trace(

            go.Scatter(

                x=self.df.index,

                y=self.df["RSI"],

                name="RSI"

            ),

            row=3,
            col=1

        )

        fig.add_hline(

            y=70,

            row=3,

            col=1,

            line_dash="dash",

            line_color="red"

        )

        fig.add_hline(

            y=30,

            row=3,

            col=1,

            line_dash="dash",

            line_color="green"

        )

        # ----------------------------------------------------
        # MACD
        # ----------------------------------------------------

        fig.add_trace(

            go.Scatter(

                x=self.df.index,

                y=self.df["MACD"],

                name="MACD"

            ),

            row=4,
            col=1

        )

        fig.add_trace(

            go.Scatter(

                x=self.df.index,

                y=self.df["MACD_SIGNAL"],

                name="Signal"

            ),

            row=4,
            col=1

        )

        fig.add_trace(

            go.Bar(

                x=self.df.index,

                y=self.df["MACD_HISTOGRAM"],

                name="Histogram"

            ),

            row=4,
            col=1

        )

        # ----------------------------------------------------

        fig.update_layout(

            height=950,

            template="plotly_dark",

            xaxis_rangeslider_visible=False,

            hovermode="x unified",

            legend=dict(
                orientation="h",
                y=1.02
            )

        )

        fig.update_yaxes(fixedrange=False)

        fig.update_xaxes(fixedrange=False)

        return fig
    