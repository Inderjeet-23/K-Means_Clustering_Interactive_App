import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d


def plot_voronoi(X, kmeans, added_points):
    # Create a meshgrid of points to plot the Voronoi diagram
    fig, ax = plt.subplots()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Predict the cluster labels for the meshgrid points
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Predict the cluster labels for the data points
    y_pred = kmeans.predict(X)

    # Plot the Voronoi diagram and the data points
    # ax.figure(figsize=(10, 8))
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.8, edgecolors='k')
    # ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='r')
    if added_points:
        for point in added_points:
            plt.scatter(point[:, 0], point[:, 1], s=5, linewidths=3, color='red')   
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('KMeans Clustering')
    ax.legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig)


# Set up the Streamlit app layout
st.set_page_config(page_title='K Means Clustering')
st.title('K Means Clustering App')
st.write("---")

st.sidebar.write("Create random dataset")

if st.session_state.get('X') is None:
    # st.session_state.added_points = []
    st.session_state.X, st.session_state.y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    st.session_state.kmeans = KMeans(n_clusters=3, random_state=42)
    st.session_state.kmeans.fit(st.session_state.X)

# Create a random dataset
n_samples = st.sidebar.number_input('Number of samples', min_value=100, max_value=1000, value=200)
n_centers = st.sidebar.number_input('Number of centers', min_value=1, max_value=20, value=3)

# Streamlit app to add random data points and visualize the changing predictions
st.title('KMeans Clustering')
st.write('This app allows you to add random data points to an initial clustered dataset and visualize the changing predictions using a Voronoi diagram.')
st.write('Initial dataset:')

n_points = st.sidebar.number_input('Number of points to add', min_value=0, max_value=100, value=15)
added_points = []
X = st.session_state.X
kmeans = st.session_state.kmeans

if n_points:
    for i in range(n_points):
        new_point = np.random.uniform(low=X.min(), high=X.max(), size=(1, 2))
        added_points.append(new_point)
        # Add the random data point to the dataset
        X = np.vstack((X, new_point))
    
    # Fit KMeans to the new dataset
    kmeans = KMeans(n_clusters=n_centers, random_state=42).fit(X)

    st.write(f'Updated dataset after adding {n_points} random data point(s):')
    st.write('---')
    st.write('Voronoi diagram:')
    st.write('---')
    plot_voronoi(X, kmeans, added_points)
    st.session_state.X = X
    st.session_state.kmeans = kmeans



