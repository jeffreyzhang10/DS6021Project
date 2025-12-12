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

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import umap.umap_ as umap


data = pd.read_csv("input_data_clean.csv")

st.set_page_config(page_title="DS6021 Final Project", layout="wide")

st.image("banner.jpg", width='stretch')

# tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Introduction",
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

    st.write('This project looks to use data from the **NFL Big Data Bowl 2026** competition on Kaggle to predict player movements on a play-by-play basis. ical movement patterns. Throughout the project, we will conduct exploratory data analysis, as well as use various machine learning techniques to gauge any interpretable insights and evalutate the performance of our chosen models.')

    st.write('Our Team Goals: Team Goals: Identifying play type, player location, intended player target, predicting pass completion')

    st.write('Research Question: Can we use various machine learning methods to accurately classify play type and the intended receiver, and predict player field location and probability of pass completion?')

    st.write('Methods Explored: EDA, PCA, K-Means, PCR, Linear Regression, Logistic Regression, KNN, t-SNE, & UMAP')


# data table
with tab2:
    st.header("Data Table")
    st.dataframe(data.head(5000))  

    st.header("Variable Descriptions")

    st.markdown("""

    ### **Source of Data**

    These definitions directly from from the **NFL Big Data Bowl 2026** competition on Kaggle: (https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/data)    

    **game_id**  
    Game identifier (numeric). Unique for each NFL game.

    **play_id**  
    Play identifier (numeric). Not unique across games — resets each game.

    **player_to_predict**  
    Boolean flag indicating whether this player's x/y location is scored in the prediction task.

    **nfl_id**  
    Unique player ID number across all players (numeric).

    **frame_id**  
    Frame counter for each game_id/play_id. Starts at 1 for every play sequence.

    **play_direction**  
    Direction the offense is moving (`left` or `right`).

    **absolute_yardline_number**  
    Yardline distance to the opponent's end zone for the possession team (numeric).

    **player_name**  
    Player’s full name (text).

    **player_height**  
    Player height (e.g., `6-2`).  

    **player_weight**  
    Player weight in pounds (numeric).

    **player_birth_date**  
    Birth date in `YYYY-MM-DD` format.

    **player_position**  
    Player’s position on the field (e.g., WR, QB, RB, TE, CB).

    **player_side**  
    Indicates whether the player is on **Offense** or **Defense**.

    **player_role**  
    Contextual role on the play:  
    - *Defensive Coverage*  
    - *Targeted Receiver*  
    - *Passer*  
    - *Other Route Runner*

    **x**  
    Player’s longitudinal field position (0–120 yards).

    **y**  
    Player’s lateral field position (0–53.3 yards).

    **s**  
    Player speed in yards/second.

    **a**  
    Player acceleration in yards/second².

    **o**  
    Player orientation in degrees (0–360). Represents which way the player is facing.

    **dir**  
    Angle of player motion in degrees (0–360). Represents direction the player is **moving**, not facing.

    **num_frames_output**  
    Number of future frames to predict for that player for a given play (numeric).

    **ball_land_x**  
    Projected football landing location along field length (0–120 yards).

    **ball_land_y**  
    Projected football landing location along field width (0–53.3 yards).
        """)

# eda
with tab3:
    st.header("Data Analysis / EDA")
    st.write("Before doing any significant data analysis for modeling, our group felt it was import to conduct some exploratory data analysis to not only get a better feel for the information at hand, but understand the underlying distributions of the data set in order to guide our decision making going forward. ")

    st.subheader("Side-by-Side Histogram of Player Speed vs. Acceleration")
    st.image("edahistogram.png", caption="Histogram of Player Speed and Player Acceleration")

    st.write('From the plots above, we can immediately see a right-skewed distribution in both speed and acceleration, which indicates that most plays that occur rely on players standing still or moving moderately slowly, relatively speaking. ')
    st.write('This makes sense intuitively, as the data also reflects how often players stop and start moving again, which we can attribute to factors like changes in direction or acceleration.')

    st.subheader("Understanding Acceleration and Speed by Player Position")

    st.write('Building upon this, we wanted to see how the data reflected changes in position, such as skill players compared to linemen.')
    st.image("boxploteda.png", caption="Boxplot of Player Speed and Acceleration by Position")

    st.write('The boxplots clearly reflect our initial hypothesis from above, where positions like wide receivers and running backs are just faster in terms of speed and acceleration relative to those like nose and defensive tackles.')

    st.write('But perhaps most importantly, to be able to make any predictions on a play-by-play basis, we need to understand visually how players move, on a graph, beyond just how we see them run on the television in front of us')
    st.image('edavelocityvectors.png', caption = 'Example Velocity Vector Distribution Snapchat At A Specific Play/Frame')

    st.write('The length and magnitude of the arrows above only reinforce how dependent motion is on factors like position and role. Looking at arrows, which represent the movements of players on both side of the ball, there are equal and opposite reactions from the players, likely based on pre-snap movement indicated by the offense and matched by the defense.')

    st.write('Overall, this EDA helped our group especially in understanding how the spatial data reflected on a visual level. From here on out, we were able to use the data to begin constructing models based on the insights already gleaned, and highlight how important visualizations can be in helping us understand this seemingly complex arrangement of numbers. ')


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
    
    # sliders
    alpha = st.slider("Test Size (percent)", 0.1, 0.5, 0.25, step=0.05)
    k = st.slider("Number of Neighbors (k)", 1, 10, 3, step=2)
    folds = st.slider("Cross-Validation folds", 3, 10, 5, step=1)

with tab8:
    st.header("External Models (TSNE, UMAP)")

    st.write("Analogy: t-SNE & UMAP are to to PCA as non-linear methods like Random Forest are to Linear Regression")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.image("pcacomp.png", caption="PCA")

    with col2:
        st.image("umap-comp.png", caption="UMAP")

    with col3:
        st.image("t-snecomp.png", caption="T-SNE")

    st.header("Hyperparameter Tuning for UMAP (n_neighbors, min_dist)")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.image("umap50.png", caption="UMAP with n_neighbors = 5, min_dist = 0")

    with col2:
        st.image("umap150.png", caption="UMAP with n_neighbors = 15, min_dist = 0")

    with col3:
        st.image("umap500.png", caption="UMAP with n_neighbors = 50, min_dist = 0")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.image("umap5.5.png", caption="UMAP with n_neighbors = 5, min_dist = 0.5")

    with col2:
        st.image("umap15.5.png", caption="UMAP with n_neighbors = 15, min_dist = 0.5")

    with col3:
        st.image("umap50.5.png", caption="UMAP with n_neighbors = 50, min_dist = 0.5")


    st.header("Hyperparameter Tuning for t-SNE (perplexity)")

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.image("tsne5.png", caption="t-SNE with Perplexity = 5")

    with col2:
        st.image("tsne30.png", caption="t-SNE with Perplexity = 30")

    with col3:
        st.image("tsne50.png", caption="t-SNE with Perplexity = 50")
