# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html, Input, Output

import plotly.express as px
import pandas as pd

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})
fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

stock_data = px.data.stocks(indexed=True)-1
area_figs = px.area(stock_data, facet_col="company", facet_col_wrap=2)

app.layout = html.Div(children=[
    html.H1(children='Test Dash'),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),

    dcc.Dropdown(
        id="ticker",
        options=["AMZN", "FB", "NFLX"],
        value="AMZN",
        clearable=False,
    ),
    dcc.Graph(id="time-series-chart"),

    dcc.Graph(
        id='areas',
        figure=area_figs,
    )
])

@app.callback(
    Output("time-series-chart", "figure"),
    Input("ticker", "value"))
def display_time_series(ticker):
    df = px.data.stocks()
    fig = px.line(df, x='date', y=ticker)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
