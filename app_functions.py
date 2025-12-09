#app_functions.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from plotly.subplots import make_subplots
import plotly.graph_objs as go

def build_model(input_data, target):
    if target == "x_clean":
        # Set numerical and categorical features
        nums = ['prev_x', 'o_clean', 'dir_radians', 'y_clean', 's_clean', 'a_clean', 'v_x', 'v_y']
        cats = ['player_side', 'player_role']
        X = input_data[nums + cats]
        y = input_data[['x_clean']]
    elif target == "y_clean":
        nums = ['prev_y', 'o_clean', 'dir_radians', 'x_clean', 's_clean', 'a_clean', 'v_x', 'v_y']
        cats = ['player_side', 'player_role']
        X = input_data[nums + cats]
        y = input_data[['y_clean']]
    return nums, cats, X, y

def fit_elastic_net(input_data, target, alpha=0.1, l1_ratio=0.5, test_size=0.2, random_state=22903):
    # Build the data
    nums, cats, X, y = build_model(input_data, target)
    y = y.values.ravel()  # flatten to 1D
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Preprocessing
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), nums),
        ("cat", OneHotEncoder(), cats)
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("elasticnet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000, random_state=random_state))
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)  # evaluate on test set
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    model.rmse = rmse

    return model, X_test, y_test, preds, nums, cats, rmse

def get_elasticnet_coefficients(model, nums, cats):
    """
    Extract coefficients from a fitted ElasticNet pipeline and return as a DataFrame.
    """
    # Preprocessor and model
    preprocessor = model.named_steps['preprocess']
    enet = model.named_steps['elasticnet']
    
    # Get feature names from preprocessing
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cats)
    feature_names = np.concatenate([nums, cat_features])
    
    # Get coefficients
    coefs = enet.coef_
    
    # Combine into DataFrame
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs
    }).sort_values(by='coefficient', key=abs, ascending=False)  # sort by magnitude
    
    return coef_df

def make_scatterplot(y_test_x, preds_x, y_test_y, preds_y):

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "X_clean: Actual vs Predicted",
        "Y_clean: Actual vs Predicted"
    ))

    fig.add_trace(
        go.Scatter(x=y_test_x, y=preds_x, mode='markers'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=y_test_x, y=y_test_x, mode='lines'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=y_test_y, y=preds_y, mode='markers'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=y_test_y, y=y_test_y, mode='lines'),
        row=1, col=2
    )

    fig.update_layout(height=500, width=1100)
    return fig