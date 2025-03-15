"""Tests for embedding storage functionality."""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from mmseqspy.embeddings.storage import (
    store_embeddings_in_df,
    store_embeddings_in_hdf,
    get_embeddings_from_df,
    get_embeddings_from_hdf,
    list_embeddings_in_hdf,
)


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return [np.ones((5, 10)), np.zeros((5, 10)), np.full((5, 10), 0.5)]


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": ["protein1", "protein2", "protein3"],
            "sequence": ["ACDEF", "KLMNP", "QRSTV"],
        }
    )


@pytest.fixture
def temp_hdf_path():
    """Create a temporary HDF5 file path."""
    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def test_store_embeddings_in_df(sample_df, sample_embeddings):
    """Test storing embeddings in DataFrame."""
    # Store embeddings
    embedding_col = "test_embedding"
    result_df = store_embeddings_in_df(sample_df, sample_embeddings, embedding_col)

    # Check that embeddings were stored correctly
    assert embedding_col in result_df.columns
    assert len(result_df[embedding_col]) == len(sample_embeddings)
    assert np.array_equal(result_df[embedding_col].iloc[0], sample_embeddings[0])

    # Check that shape column was added
    assert f"{embedding_col}_shape" in result_df.columns
    assert result_df[f"{embedding_col}_shape"].iloc[0] == str(
        sample_embeddings[0].shape
    )


def test_store_and_get_from_hdf(sample_embeddings, temp_hdf_path):
    """Test storing and retrieving embeddings from HDF5."""
    protein_ids = ["protein1", "protein2", "protein3"]
    embedding_type = "test_embedding"

    # Store embeddings
    references = store_embeddings_in_hdf(
        sample_embeddings, protein_ids, embedding_type, temp_hdf_path
    )

    # Check references
    assert len(references) == len(sample_embeddings)
    assert all(ref.startswith(embedding_type) for ref in references)

    # Retrieve embeddings
    retrieved = get_embeddings_from_hdf(references, temp_hdf_path)

    # Check retrieval
    assert len(retrieved) == len(sample_embeddings)
    for i in range(len(sample_embeddings)):
        assert np.array_equal(retrieved[i], sample_embeddings[i])


def test_list_embeddings_in_hdf(sample_embeddings, temp_hdf_path):
    """Test listing embeddings in HDF5."""
    # Store two types of embeddings
    protein_ids = ["protein1", "protein2", "protein3"]
    store_embeddings_in_hdf(sample_embeddings, protein_ids, "type1", temp_hdf_path)
    store_embeddings_in_hdf(sample_embeddings, protein_ids, "type2", temp_hdf_path)

    # List all embeddings
    all_embeddings = list_embeddings_in_hdf(temp_hdf_path)
    assert "type1" in all_embeddings
    assert "type2" in all_embeddings
    assert len(all_embeddings["type1"]) == len(protein_ids)

    # List specific type
    type1_embeddings = list_embeddings_in_hdf(temp_hdf_path, "type1")
    assert "type1" in type1_embeddings
    assert "type2" not in type1_embeddings


def test_get_embeddings_from_df(sample_df, sample_embeddings):
    """Test retrieving embeddings from DataFrame."""
    # Store embeddings
    embedding_col = "test_embedding"
    df = sample_df.copy()
    df[embedding_col] = sample_embeddings

    # Retrieve embeddings
    retrieved = get_embeddings_from_df(df, embedding_col)

    # Check retrieval
    assert len(retrieved) == len(sample_embeddings)
    for i in range(len(sample_embeddings)):
        assert np.array_equal(retrieved[i], sample_embeddings[i])

    # Check error handling
    with pytest.raises(ValueError):
        get_embeddings_from_df(df, "nonexistent_col")
