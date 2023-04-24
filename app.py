import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd
from csv_create import blob_data
import atexit

# Function to plot the Voronoi diagram


def plot_voronoi(X, kmeans, added_points):
    # Create a meshgrid of points to plot the Voronoi diagram
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predict the cluster labels for the meshgrid points
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Predict the cluster labels for the data points
    y_pred = kmeans.predict(X)

    # Plot the Voronoi diagram and the data points
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.8, edgecolors='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
                :, 1], marker='*', s=300, c='r')
    if added_points:
        for point in added_points:
            plt.scatter(point[:, 0], point[:, 1], marker='x',
                        s=200, linewidths=3, color='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('KMeans Clustering')
    st.pyplot(plt)


def run_kmeans_app():
    # Create a clustered dataset using make_blobs
    df = pd.read_csv('blobs.csv')
    X = df[['feature1', 'feature2']].values

    # Initialize KMeans with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)

    # Streamlit app to add random data points and visualize the changing predictions
    st.title('KMeans Clustering')
    st.write('This app allows you to add random data points to an initial clustered dataset and visualize the changing predictions using a Voronoi diagram.')
    st.write('Initial dataset:')
    plot_voronoi(X, kmeans, added_points=[])

    # Add random data points to the initial dataset
    num_points = 1
    added_points = []

    button = st.button(f'Add random data point(s)')
    if button:
        # Generate a random data point
        for i in range(num_points):
            new_point = np.random.uniform(
                low=X.min(), high=X.max(), size=(1, 2))
            # added_points.append(new_point)
            # Add the random data point to the dataset
            X = np.vstack((X, new_point))
        # Update the KMeans model with the new data point
        kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
        df_new = pd.DataFrame(new_point, columns=["feature1", "feature2"])
        df_new["label"] = kmeans.predict(new_point)
        with open("blobs.csv", "a") as f:
            df_new.to_csv(f, header=False, index=False)
        # Plot the updated Voronoi diagram
        st.write(f'Updated dataset after adding {i+1} random data point(s):')
        plot_voronoi(X, kmeans, added_points)

    # Concatenate the added points with the initial dataset
    if added_points:
        added_points = np.vstack(added_points)
        X = np.vstack((X[:-len(added_points)], added_points))


if __name__ == '__main__':
    run_kmeans_app()

atexit.register(blob_data)
