"""Tests for embedding dimension reduction functionality."""

import pytest
import numpy as np
import os
import tempfile
from mmseqspy.embeddings.reduction import (
    reduce_dimensions,
    apply_reducer,
    save_reducer,
    load_reducer,
)


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    # Create high-dimensional embeddings
    return np.random.random((10, 100))  # 10 samples, 100 dimensions


@pytest.fixture
def temp_reducer_path():
    """Create a temporary file path for reducer."""
    fd, path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def test_reduce_dimensions(sample_embeddings):
    """Test dimension reduction."""
    # Reduce dimensions with PCA
    n_components = 5
    reduced, reducer = reduce_dimensions(
        sample_embeddings, method="pca", n_components=n_components
    )

    # Check shape of reduced embeddings
    assert reduced.shape == (sample_embeddings.shape[0], n_components)

    # Check that reducer is a PCA instance
    from sklearn.decomposition import PCA

    assert isinstance(reducer, PCA)

    # Check error handling for unsupported method
    with pytest.raises(ValueError):
        reduce_dimensions(sample_embeddings, method="unsupported")


def test_apply_reducer(sample_embeddings):
    """Test applying a fitted reducer to new data."""
    # First, fit a reducer
    n_components = 5
    _, reducer = reduce_dimensions(
        sample_embeddings, method="pca", n_components=n_components
    )

    # Create new embeddings
    new_embeddings = np.random.random((5, 100))  # 5 new samples

    # Apply reducer
    reduced = apply_reducer(new_embeddings, reducer)

    # Check shape
    assert reduced.shape == (new_embeddings.shape[0], n_components)


def test_save_load_reducer(sample_embeddings, temp_reducer_path):
    """Test saving and loading a reducer."""
    # First, fit a reducer
    _, reducer = reduce_dimensions(sample_embeddings, method="pca", n_components=5)

    # Save reducer
    save_reducer(reducer, temp_reducer_path)

    # Check file exists
    assert os.path.exists(temp_reducer_path)

    # Load reducer
    loaded_reducer = load_reducer(temp_reducer_path)

    # Create new embeddings
    new_embeddings = np.random.random((5, 100))

    # Apply both reducers and check results are the same
    reduced_original = apply_reducer(new_embeddings, reducer)
    reduced_loaded = apply_reducer(new_embeddings, loaded_reducer)

    assert np.allclose(reduced_original, reduced_loaded)

    # Check error handling for nonexistent file
    with pytest.raises(FileNotFoundError):
        load_reducer("nonexistent_file.pkl")
