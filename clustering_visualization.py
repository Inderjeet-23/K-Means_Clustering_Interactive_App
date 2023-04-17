import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap

# Generate initial dataset
X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# Fit KMeans model to initial dataset
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Define custom color map for clusters
colors = ['tab:blue', 'tab:orange', 'tab:green']
cmap = ListedColormap(colors)

# Create plot and plot initial dataset
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap=cmap)

# Compute Voronoi diagram of initial cluster centers
vor = Voronoi(kmeans.cluster_centers_)

# Draw Voronoi diagram
voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False)

# Define function to add new point to dataset


def add_point(event):
    global X

    # Add new point to dataset
    X = np.vstack([X, [event.xdata, event.ydata]])

    # Fit KMeans to updated dataset
    kmeans.fit(X)

    # Compute Voronoi diagram of updated dataset
    vor = Voronoi(kmeans.cluster_centers_)

    # Clear current plot and redraw updated dataset and Voronoi diagram
    ax.clear()
    ax.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap=cmap)
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False)

    plt.draw()


# Connect mouse click event to add_point function
cid = fig.canvas.mpl_connect('button_press_event', add_point)

# Show plot
plt.show()
