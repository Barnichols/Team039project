# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import dash
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
    "Feature Name": ['EXT_SOURCE_3',
                     'EXT_SOURCE_1',
                     'EXT_SOURCE_2',
                     'DAYS_REGISTRATION',
                     'DAYS_BIRTH',
                     'DAYS_EMPLOYED',
                     'DAYS_LAST_PHONE_CHANGE',
                     'YEARS_BEGINEXPLUATATION_MEDI',
                     'LANDAREA_MODE',
                     'AMT_CREDIT',
                     'COMMONAREA_MODE',
                     'LIVINGAPARTMENTS_AVG'],
    "Mean Decrease in Impurity": [
        0.05709844694331767,
        0.040486775913126785,
        0.02898956120671305,
        0.025508123803522204,
        0.023971109383082355,
        0.022424636185079692,
        0.018719973658598222,
        0.017385638068831873,
        0.01726286263265718,
        0.016982041510066038,
        0.016597071037180035,
        0.01649704056420024,
    ]
})

fig = px.bar(
    df,
    x="Feature Name",
    y="Mean Decrease in Impurity",
    barmode="group",
    title="Mean Decrease in Impurity for Top 12 Features",
    height=650
)

stock_data = px.data.stocks(indexed=True)-1
area_figs = px.area(stock_data, facet_col="company", facet_col_wrap=2)

app.layout = html.Div(style={"fontFamily": ["Open Sans", "verdana", "arial", "sans-serif"]},children=
    [
        html.Div([
            html.H1('Team 039 Project Dashboard: Home Credit Default Risk Analysis'),
        ]),
        html.Div(children=[dcc.Tabs([
            
    dcc.Tab(label='Model Prediction', children=[
        html.Div(
            html.H6(
                'Here you can put the undersampling model to the test. There are three features below that you can manipulate and see how they effect your liklihood of defaulting. Try putting in your numbers and see how it goes! \nThe median values from the dataset are displayed as placeholders for reference.'
            )
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        (
                            html.Div(children=[
                                html.Div('AMT_CREDIT: Credit amount of the loan (must be positive)'),
                                html.Div(dcc.Input(
                                    id="AMT_CREDIT", type="number", placeholder=585000, min=0
                                ))
                            ],
                                className="combo-box"
                            )),
                        (
                            html.Div(children=[
                                html.Div('DAYS_BIRTH: Client\'s age in days at the time of application'),
                                html.Div(dcc.Input(id="DAYS_BIRTH", type="number", placeholder=13883.5,min=6570
))
                            ],
                                className="combo-box"
                            )
                        ),
                        (
                            html.Div(children=[
                                html.Div('DAYS_EMPLOYED: How many days before the application the person started current employment'),
                                html.Div(dcc.Input(id="DAYS_EMPLOYED", type="number", placeholder=1680.5, min=0))
                            ],
                                className="combo-box"
                            )
                        ),
                        html.Div(children=[
                                dbc.Button("Run Gradient Boosting Machine Model",id='rungbm',n_clicks=0,className='me-2'),
                                dbc.Button("Run Random Forest Classifier Model",id='runrfc',n_clicks=0,className='me-2')
                            ], className="btn-ctn"
                            ),
                        html.Div(id='outputstate', children=[]),
                    ], className="form")
                    ],
            className="form-ctn"
                ),
            ]
        ),dcc.Tab(label='Project Analysis', children=[
            dcc.Graph(
                id='mean-decrease-in-impurity',
                figure=fig
            )]
    )])])])

@app.callback(
    [Output('outputstate','children')],
    [Input(component_id='rungbm',component_property='n_clicks'),
     Input(component_id='runrfc',component_property='n_clicks')],
    [State(component_id ='AMT_CREDIT',component_property='value' ),
     State(component_id = 'DAYS_BIRTH',component_property='value'),
     State(component_id = 'DAYS_EMPLOYED',component_property='value')]
     )
def runmodel(nclicks1,nclicks2, AMT_CREDIT, DAYS_BIRTH, DAYS_EMPLOYED):
    if AMT_CREDIT is None:
        raise PreventUpdate
    elif DAYS_BIRTH is None:
        raise PreventUpdate
    elif DAYS_EMPLOYED is None:
        raise PreventUpdate
    else:
        if dash.callback_context.triggered[0]["prop_id"] == 'rungbm.n_clicks':
            model = joblib.load('GBMboost.pkl')
            mdl = 'GBM'
        else:
            model = joblib.load('RFC_AL.pkl')
            mdl = 'RFC'
            
        foo = np.array([0.00000000e+00,  2.02500000e+05,  6.63264000e+05,  6.97770000e+04, 6.30000000e+05,  1.91010000e-02, -2.00380000e+04, -4.45800000e+03, -2.17500000e+03, -3.50300000e+03,  5.00000000e+00,  1.00000000e+00, 1.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00, 0.00000000e+00,  2.00000000e+00,  2.00000000e+00,  2.00000000e+00, 1.40000000e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  5.06771262e-01, 6.99786830e-01,  6.10991328e-01,  9.28000000e-02,  7.81000000e-02, 9.81600000e-01,  7.55200000e-01,  2.27000000e-02,  0.00000000e+00, 1.37900000e-01,  1.66700000e-01,  2.08300000e-01,  4.83000000e-02, 7.56000000e-02,  7.70000000e-02,  0.00000000e+00,  3.80000000e-03, 8.51000000e-02,  7.70000000e-02,  9.81600000e-01,  7.58300000e-01, 2.03000000e-02,  0.00000000e+00,  1.37900000e-01,  1.66700000e-01, 2.08300000e-01,  4.62000000e-02,  8.17000000e-02,  7.51000000e-02, 0.00000000e+00,  1.20000000e-03,  9.26000000e-02,  7.78000000e-02, 9.81600000e-01,  7.58500000e-01,  2.23000000e-02,  0.00000000e+00, 1.37900000e-01,  1.66700000e-01,  2.08300000e-01,  4.88000000e-02, 7.70000000e-02,  7.76000000e-02,  0.00000000e+00,  3.10000000e-03, 7.07000000e-02,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00, -8.56000000e+02,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  4.00000000e+00, 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00, 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00, 0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])
        foo[2] = AMT_CREDIT
        foo[6] = -DAYS_BIRTH
        foo[7] = -DAYS_EMPLOYED
        myans = model.predict_proba(foo.reshape(1,-1))
        bar = dash.callback_context
        return [f'The {mdl} model predicts that you have a {round(myans[0][1]*100,2)}% chance of defaulting']



if __name__ == '__main__':
    app.run_server(debug=True)
