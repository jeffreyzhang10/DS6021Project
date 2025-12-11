import streamlit as st
import pandas as pd
import numpy as np
from app_functions import build_model, fit_elastic_net, get_elasticnet_coefficients
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression


data = pd.read_csv("input_data_clean.csv")


st.set_page_config(page_title="DS6021 Final Project", layout="wide")

# tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Title",
    "Data Table",
    "Data Analysis / EDA",
    "PCA / KMeans / PCR",
    "Linear Regression",
    "Logistic Regression",
    "KNN",
    "External Models"
])


with tab1:
    st.title("DS6021 Final Project")
    st.write("Emmett Hannam, Jarrett Markman, Weston Williams, Jeffrey Zhang")

# data table
with tab2:
    st.header("Data Table")
    st.dataframe(data.head(5000))  

# eda
with tab3:
    st.header("Data Analysis / EDA")
    st.write("Add your EDA visualizations here.")

with tab4:


    st.header("PCA / KMeans / PCR")

    st.subheader("Original PCA / K-Means Plot With 8 Clusters")
    st.write("Given the complexity of play types and formations in football, we initially chose 8 clusters in an attempt to shape our data into defined groups that were part of different styles. ")
    st.image("kmeans8.png", caption="K-Means with 8 Clusters (Initial)")

    st.subheader("PCA / K-Means Plot With 4 Clusters")
    st.write("Based on the elbow and silhouette methods, our group determined that 4 clusters was the most appropriate number, which can be shown below.")
    st.image("kmeans4.png", caption="K-Means with 8 Clusters (Initial)")

    st.subheader("Principal Component Regression (PCR)")

    num_components = st.slider(
        "Select number of PCA components Here!",
        min_value=1,
        max_value=8, # len features, only 8 features later on 
        value=2, 
        step=1
    )

    st.write(f"Using **{num_components}** principal components for PCR...")

    data = pd.read_csv("input_data_clean.csv")

    # interested in offense only data
    offense = data [data ['player_side'] == 'Offense'].copy()
    # get pre snap data only
    pre_snap = offense.loc[offense.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].idxmin()]

    features = ['prev_x', 'prev_y', 'o_clean', 'dir_radians', 's_clean', 'a_clean', 'v_x', 'v_y']
    X = offense[features].values
    y = offense['x_clean'].values

    X_train, X_test, y_train, y_test = train_test_split( # Standard train-test split
    X, y, test_size=0.25, random_state=42)


    pcr_pipe = Pipeline([ # FULL PCR pipeline
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components = num_components)), # changed here 
    ("linreg", LinearRegression())])

    pcr_pipe.fit(X_train, y_train) # fitting the PCR model on the training data

    train_r2 = pcr_pipe.score(X_train, y_train)
    test_r2 = pcr_pipe.score(X_test, y_test)

    y_train_pred = pcr_pipe.predict(X_train)
    y_test_pred = pcr_pipe.predict(X_test)

    train_rmse = np.sqrt(np.mean((y_train - y_train_pred)**2))
    test_rmse = np.sqrt(np.mean((y_test - y_test_pred)**2))

    # Model Performance 

    st.subheader("PCR Performance Metrics")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Train R²", f"{train_r2:.3f}")
        st.metric("Train RMSE", f"{train_rmse:.2f}")

    with col2:
        st.metric("Test R²", f"{test_r2:.3f}")
        st.metric("Test RMSE", f"{test_rmse:.2f}")

    # Exploring Variance
    pca_model = pcr_pipe.named_steps["pca"]

    st.subheader("Explained Variance")
    st.write(f"Total variance explained: **{pca_model.explained_variance_ratio_.sum():.3f}**")

    pcr_df = pd.DataFrame({ # table the results 
        "PC #": np.arange(1, num_components + 1),
        "PCR Variance Explained": pca_model.explained_variance_ratio_
    })
    st.dataframe(pcr_df, width='stretch') # widen to the entire length of the app 


with tab5:
    st.header("Linear Regression (Elastic Net)")

    # sliders
    alpha = st.slider("Alpha", 0.0, 1.0, 0.1, step=0.05)
    l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, step=0.1)
    test_size = st.slider("Test Size", 0.0, 0.9, 0.2, step=0.1)

    if st.button("Run Models"):
        with st.spinner("Fitting Elastic Net models..."):

            model_x, X_test_x, y_test_x, preds_x, nums_x, cats_x, rmse_x = fit_elastic_net(
                data, target="x_clean", alpha=alpha, l1_ratio=l1_ratio, test_size=test_size
            )
            coef_x = get_elasticnet_coefficients(model_x, nums_x, cats_x)

            model_y, X_test_y, y_test_y, preds_y, nums_y, cats_y, rmse_y = fit_elastic_net(
                data, target="y_clean", alpha=alpha, l1_ratio=l1_ratio, test_size=test_size
            )
            coef_y = get_elasticnet_coefficients(model_y, nums_y, cats_y)

        st.subheader("RMSE Results")
        st.write(f"**RMSE X_clean:** {rmse_x:.4f}")
        st.write(f"**RMSE Y_clean:** {rmse_y:.4f}")

        st.subheader("Elastic Net Coefficients — X_clean")
        st.dataframe(coef_x.style.format({"coefficient": "{:.4f}"}))

        st.subheader("Elastic Net Coefficients — Y_clean")
        st.dataframe(coef_y.style.format({"coefficient": "{:.4f}"}))


with tab6:
    st.header("Logistic Regression")
    st.write("Add logistic regression functions here.")

with tab7:
    st.header("KNN")
    st.write("Add KNN modeling here.")

with tab8:
    st.header("External Models (TSNE, UMAP)")
    st.write("Add models here.")
