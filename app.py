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

app.layout = html.Div([
                html.Div([html.H1('Team 039 Project Dashboard: Home Credit Default Risk Analysis',style={'textAlign': 'center'})]),
       html.Div(children=[dcc.Tabs([
        dcc.Tab(label='Project Analysis', children=[
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
]),

dcc.Tab(label='Dummy Sliders', children=[
        dcc.Input(id="Placeholder 1", type="number", placeholder="Placeholder 1"),
        dcc.Input(id="Placeholder 2", type="number", placeholder="Placeholder 2"),
        dcc.Input(id="Placeholder 3", type="number", placeholder="Placeholder 3"),
        dcc.Input(id="Placeholder 4", type="number", placeholder="Placeholder 4"),
        dcc.Slider(0, 2000, 1, value=5,marks=None, id='my-range-slider',tooltip={"placement": "bottom", "always_visible": True}),
        html.Hr(),
        html.Div(id="number-out")])
])])])
@app.callback(
    Output("time-series-chart", "figure"),
    Input("ticker", "value"))
def display_time_series(ticker):
    df = px.data.stocks()
    fig = px.line(df, x='date', y=ticker)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
