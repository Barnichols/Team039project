# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour
import joblib

import plotly.express as px
import pandas as pd



app = Dash(__name__,external_stylesheets=[dbc.themes.SPACELAB])

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
        html.Div(html.H6('Here you can put the undersampling model to the test. There are three features below that you can manipulate and see how they effect your liklihood of defaulting. Try putting in your numbers and see how it goes! \nThe median values from the dataset are displayed as placeholders for reference.')),
        (html.Div(children=['AMT_CREDIT: Credit amount of the loan (must be positive)',html.Br(),dcc.Input(id="AMT_CREDIT", type="number", placeholder=585000,min=0)])),
        html.Br(),
        (html.Div(['DAYS_BIRTH: Client\'s age in days at the time of application',html.Br(),dcc.Input(id="DAYS_BIRTH", type="number", placeholder=13883.5,min=0)])),
        html.Br(),
       (html.Div(['DAYS_EMPLOYED: How many days before the application the person started current employment',html.Br(),dcc.Input(id="DAYS_EMPLOYED", type="number", placeholder=1680.5,min=0)])),
       html.Div(dbc.Button("Run Model",id='runmodel',n_clicks=0,className='me-2')),
       html.Div(id='outputstate',children=[]),
        html.Hr(),
        html.Div([html.H1('',style={'textAlign': 'center'})])])
])])])
@app.callback(
    Output("time-series-chart", "figure"),
    Input("ticker", "value"))
def display_time_series(ticker):
    df = px.data.stocks()
    fig = px.line(df, x='date', y=ticker)
    return fig

@app.callback(
    [Output('outputstate','children')],
    [Input(component_id='runmodel',component_property='n_clicks')],
    [State(component_id ='AMT_CREDIT',component_property='value' ),
     State(component_id = 'DAYS_BIRTH',component_property='value'),
     State(component_id = 'DAYS_EMPLOYED',component_property='value')]
     )
def runmodels(nclicks,AMT_CREDIT,DAYS_BIRTH,DAYS_EMPLOYED):
    if AMT_CREDIT is None:
        raise PreventUpdate
    elif DAYS_BIRTH is None:
        raise PreventUpdate
    elif DAYS_EMPLOYED is None:
        raise PreventUpdate
    else:
        model = joblib.load('undersampling_RFC.pkl')
        foo = np.array([0,193500,585000,29209.5,495000,0.020246,-13883.5,-1680.5,-4001,-3197,8,1,1,0,1,0,0,2,2,2,12,0,0,0,0,0,0,0.4984382311,0.5963049427,0.5136937663,0.0938,0.07875,0.9831,0.7688,0.0239,0,0.1379,0.1667,0.2083,0.0499,0.0756,0.08385,0,0.0028,0.0924,0.0767,0.9826,0.7713,0.0214,0,0.1379,0.1667,0.2083,0.04665,0.0808,0.0805,0,0,0.0937,0.0783,0.9831,0.7719,0.0238,0,0.1379,0.1667,0.2083,0.0502,0.077,0.08445,0,0.0021,0.0774,0,0,0,0,-967,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0])
        foo[2] = AMT_CREDIT
        foo[6] = -DAYS_BIRTH
        foo[7] = -DAYS_EMPLOYED
        myans = model.predict(foo.reshape(1,-1))
        if myans[0] == 0:
            myans = 'Not Default!'
        else:
            myans = 'Default :('
        return [f'Model predicts that you will: {myans}']

if __name__ == '__main__':
    app.run_server(debug=True)
