"""Dimension reduction utilities for protein sequence embeddings."""

import os
import numpy as np
import pickle
from typing import Tuple
from sklearn.decomposition import PCA
from typing import Any


def reduce_dimensions(
    embeddings: np.ndarray, method: str = "pca", n_components: int = 50
) -> Tuple[np.ndarray, Any]:
    """
    Reduce embedding dimensions using specified method.

    Args:
        embeddings: Array of embeddings to reduce (shape: n_samples, n_features)
        method: Reduction method ('pca' currently supported)
        n_components: Number of components to reduce to

    Returns:
        Tuple of (reduced_embeddings, reducer)
    """
    if method.lower() == "pca":
        reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(embeddings)
        return reduced, reducer
    else:
        raise ValueError(f"Unsupported reduction method: {method}")


def apply_reducer(embeddings: np.ndarray, reducer: Any) -> np.ndarray:
    """
    Apply a fitted reducer to new embeddings.

    Args:
        embeddings: Array of embeddings to reduce
        reducer: Fitted reducer (e.g., PCA instance)

    Returns:
        Reduced embeddings
    """
    return reducer.transform(embeddings)


def save_reducer(reducer: Any, file_path: str) -> None:
    """
    Save a fitted reducer to file.

    Args:
        reducer: Fitted reducer (e.g., PCA instance)
        file_path: Path to save the reducer
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    # Save reducer
    with open(file_path, "wb") as f:
        pickle.dump(reducer, f)


def load_reducer(file_path: str) -> Any:
    """
    Load a previously saved reducer.

    Args:
        file_path: Path to the saved reducer

    Returns:
        Loaded reducer
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reducer file not found: {file_path}")

    with open(file_path, "rb") as f:
        return pickle.load(f)
