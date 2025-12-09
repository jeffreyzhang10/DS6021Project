import numpy as np
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from dash.dash_table import DataTable
from app_functions import build_model, fit_elastic_net, get_elasticnet_coefficients

# Load data
data = pd.read_csv("input_data_clean.csv")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

# Layout
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Title', children=[
            html.H1('DS6021 Final Project'),
            html.H3('Emmett Hannam, Jarrett Markman, Weston Williams, Jeffrey Zhang')
        ]),

        dcc.Tab(label='Data Table', children=[
            html.H2('Data Table'),
            DataTable(
                id='datatable',
                data=data.head(5000).to_dict('records'),   # show top 5k rows
                columns=[{'name': i, 'id': i} for i in data.columns],
                page_size=20,
                filter_action='native',
                sort_action='native',
                style_table={'overflowX': 'scroll'}
            )
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

            html.Br(),
            html.H3("RMSE Results"),
            html.Div(id='linreg-output'),

            html.Br(),
            html.H3("Elastic Net Coefficients (X_clean)"),
            html.Div(id='coef-table-x'),

            html.Br(),
            html.H3("Elastic Net Coefficients (Y_clean)"),
            html.Div(id='coef-table-y')#,

            #html.Br(),
            #html.H3("Actual vs Predicted Scatterplots"),
            #dcc.Graph(id='scatter-plot')
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


# Callback

@app.callback(
    Output('linreg-output', 'children'),
    Output('coef-table-x', 'children'),
    Output('coef-table-y', 'children'),
    #Output('scatter-plot', 'figure'),
    Input('alpha-slider', 'value'),
    Input('l1-slider', 'value'),
    Input('testsize-slider', 'value')
)
def update_linear_regression(alpha, l1_ratio, test_size):

    model_x, X_test_x, y_test_x, preds_x, nums_x, cats_x, rmse_x= fit_elastic_net(
        data,
        target="x_clean",
        alpha=alpha,
        l1_ratio=l1_ratio,
        test_size=test_size
    )
    #rmse_x = model_x.rmse
    coef_x = get_elasticnet_coefficients(model_x, nums_x, cats_x)

    model_y, X_test_y, y_test_y, preds_y, nums_y, cats_y, rmse_y = fit_elastic_net(
        data,
        target="y_clean",
        alpha=alpha,
        l1_ratio=l1_ratio,
        test_size=test_size
    )
    #rmse_y = model_y.rmse
    coef_y = get_elasticnet_coefficients(model_y, nums_y, cats_y)

    #fig = make_scatterplot(y_test_x, preds_x, y_test_y, preds_y)

    coef_table_x = html.Table([
        html.Tr([html.Th("Feature"), html.Th("Coefficient")])] +
        [html.Tr([html.Td(r.feature), html.Td(round(r.coefficient, 4))]) for _, r in coef_x.iterrows()]
    )

    coef_table_y = html.Table([
        html.Tr([html.Th("Feature"), html.Th("Coefficient")])] +
        [html.Tr([html.Td(r.feature), html.Td(round(r.coefficient, 4))]) for _, r in coef_y.iterrows()]
    )

    return (
        f"RMSE X_clean: {rmse_x:.4f} — RMSE Y_clean: {rmse_y:.4f}",
        coef_table_x,
        coef_table_y#,
        #fig
    )

# Run app
if __name__ == '__main__':
    app.run(debug=True)