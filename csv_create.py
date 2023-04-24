from sklearn.datasets import make_blobs
import pandas as pd
import os


def blob_data():
    # Specify the file path
    file_path = "blobs.csv"

    # Check if the file exists
    if os.path.isfile(file_path):
        # Delete the file if it exists
        os.remove(file_path)

    # Generate 100 datapoints using make_blobs
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

    # Save data to a CSV file
    df = pd.DataFrame(X, columns=["feature1", "feature2"])
    df["label"] = y
    df.to_csv("blobs.csv", index=False)
