"""Data loaders and preprocessing utilities."""
import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer,
    make_moons, make_circles, make_blobs,
    make_classification,
)
from sklearn.preprocessing import StandardScaler


def standardize(X):
    return StandardScaler().fit_transform(X)


def load_iris_data():
    d = load_iris()
    return standardize(d.data), d.target, d.feature_names, d.target_names


def load_wine_data():
    d = load_wine()
    return standardize(d.data), d.target, d.feature_names, d.target_names


def load_breast_cancer_data():
    d = load_breast_cancer()
    return standardize(d.data), d.target, d.feature_names, d.target_names


def load_moons(n=300, noise=0.15):
    X, y = make_moons(n_samples=n, noise=noise, random_state=42)
    return standardize(X), y


def load_circles(n=300, noise=0.1):
    X, y = make_circles(n_samples=n, noise=noise, factor=0.5, random_state=42)
    return standardize(X), y


def load_blobs(n=300, centers=3):
    X, y = make_blobs(n_samples=n, centers=centers, random_state=42)
    return standardize(X), y


def load_classification(n=500, features=10, classes=3):
    X, y = make_classification(n_samples=n, n_features=features,
                                n_classes=classes, n_informative=features//2,
                                random_state=42)
    return standardize(X), y


def csv_to_xy(df: pd.DataFrame, target_col: str):
    y = df[target_col].values
    X = df.drop(columns=[target_col]).select_dtypes(include=np.number).values
    return standardize(X), y
