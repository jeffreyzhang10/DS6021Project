# Import packages
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

# Create build model function for building data for a linear regression model
def build_model(input_data, target):
    """
    Input: input_data, target

    Description: Build the data for a linear regression model

    Output: nums, cats, X, y
    """
    if target == "x_clean": # If the target is x_clean
        # Set numerical and categorical features
        nums = ['prev_x', 'o_clean', 'dir_radians', 'y_clean', 's_clean', 'a_clean', 'v_x', 'v_y']
        cats = ['player_side', 'player_role']
        X = input_data[nums + cats]
        y = input_data[['x_clean']]
    elif target == "y_clean": # If the target is y_clean
        nums = ['prev_y', 'o_clean', 'dir_radians', 'x_clean', 's_clean', 'a_clean', 'v_x', 'v_y']
        cats = ['player_side', 'player_role']
        X = input_data[nums + cats]
        y = input_data[['y_clean']]
    else: # If the target is not x_clean or y_clean
        raise ValueError("Invalid target. Must be 'x_clean' or 'y_clean'.")
    return nums, cats, X, y

# Function to fit an elastic net model based on parameters
def fit_elastic_net(input_data, target, alpha=0.1, l1_ratio=0.5, test_size=0.2):
    """
    Input: input_data, target, alpha, l1_ratio, test_size

    Description: Fit an elastic net model based on parameters

    Output: model, X_test, y_test, preds, nums, cats, rmse
    """
    # Build the data
    nums, cats, X, y = build_model(input_data, target)
    y = y.values.ravel()  # Flatten to 1D
    
    # Train/test split based on test size 
    X_train, X_test, y_train, y_test = train_test_split(
        # Split x, y, input test_size parameter
        X, y, test_size=test_size, random_state=22903
    )
    
    # Set model preprocessing with standard scaler and one hot encoder for numerical and categorical features
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), nums),
        ("cat", OneHotEncoder(), cats)
    ])

    # Use preprocessing and elastic net to set the model pipeline
    model = Pipeline([
        ("preprocess", preprocess),
        # Build elastic net model based on given alpha and l1_ratio
        ("elasticnet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000, random_state=22903))
    ])
    # Fit the model, make predictions, and get model RMSE
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return model, X_test, y_test, preds, nums, cats, rmse

# Build function to get the elastic net coefficients for a given model
def get_elasticnet_coefficients(model, nums, cats):
    """
    Extract coefficients from a fitted ElasticNet pipeline and return as a DataFrame.
    """
    # Preprocessor and model
    # Get preprocessor and elastic net for model
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