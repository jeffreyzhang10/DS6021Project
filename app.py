import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from data_loader import load_weekly_data, load_supplementary

#input_data, output_data = load_weekly_data()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Title', children=[
            html.H1('DS6021 Final Project')
        ]),
        dcc.Tab(label='Data Table', children=[
            html.H2('Data Table')
        ]),
        dcc.Tab(label='Data Analysis / EDA / KMeans', children=[
            html.H2('Data Analysis / EDA / KMeans')
        ]),
        dcc.Tab(label='Linear Regression', children=[
            html.H2('Linear Regression'),
            html.Label('Alpha'),
            dcc.Slider(id='alpha-slider', min=0, max=1, step=0.05, value=0.1),
            html.Label('L1 Ratio'),
            dcc.Slider(id='l1-slider', min=0, max=1, step=0.1, value=0.5),
            html.Label('Test Size'),
            dcc.Slider(id='testsize-slider', min=0, max=0.9, step=0.1, value=0.2),
            html.Div(id='linreg-output')
        ]),
        dcc.Tab(label='Logistic Regression', children=[
            html.H2('Logistic Regression')
        ]),
        dcc.Tab(label='KNN (K-Nearest Neighbor)', children=[
            html.H2('KNN (K-Nearest Neighbor)')
        ]),
        dcc.Tab(label='External Models (T-SNE, UMAP, etc.)', children=[
            html.H2('External Models (T-SNE, UMAP, etc.)')
        ])
    ])
])

if __name__ == '__main__':
    app.run(debug=True)