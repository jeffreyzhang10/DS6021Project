import streamlit as st
import pandas as pd
import numpy as np
from app_functions import build_model, fit_elastic_net, get_elasticnet_coefficients

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
