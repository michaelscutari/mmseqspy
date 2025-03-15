"""Tests for the embeddings API."""

import pytest
import pandas as pd
import numpy as np
from mmseqspy.embeddings import (
    add_embeddings,
    get_embeddings,
    list_available_embedders,
    register_embedder,
)


# Sample data for tests
@pytest.fixture
def sample_df():
    """Create a small sample DataFrame for testing."""
    return pd.DataFrame(
        {"id": ["seq1", "seq2", "seq3"], "sequence": ["ACDEFGH", "KLTWYV", "MNPQRS"]}
    )


def test_list_available_embedders():
    """Test listing available embedders."""
    embedders = list_available_embedders()
    assert isinstance(embedders, list)
    assert len(embedders) > 0
    assert "blosum62" in embedders
    assert "aac" in embedders
    assert "property" in embedders  # Renamed from physico to property


def test_register_embedder():
    """Test registering a custom embedder."""
    from mmseqspy.embeddings.baseline import BaseEmbedder

    # Create a custom embedder
    class CustomEmbedder(BaseEmbedder):
        def generate(self, sequence, pooling="auto", max_length=None):
            # Simple implementation for testing
            return np.ones(len(sequence))

    # Register the custom embedder
    register_embedder("custom", CustomEmbedder)

    # Check that it was registered
    embedders = list_available_embedders()
    assert "custom" in embedders

    # Test using the custom embedder
    df = pd.DataFrame({"sequence": ["ACDEFGH"]})
    result = add_embeddings(df, "custom")
    assert "custom_embedding" in result.columns
    assert np.array_equal(result["custom_embedding"][0], np.ones(7))


def test_add_embeddings_basic(sample_df):
    """Test adding embeddings to a DataFrame."""
    # Add BLOSUM62 embeddings
    result = add_embeddings(sample_df, "blosum62")

    # Check that the embedding column was added
    assert "blosum62_embedding" in result.columns

    # Check that the original DataFrame was not modified
    assert "blosum62_embedding" not in sample_df.columns

    # Check the embeddings
    embeddings = result["blosum62_embedding"]
    assert len(embeddings) == 3

    # Check the dimensions
    first_emb = embeddings.iloc[0]
    assert first_emb.shape == (7, 20)  # 7 residues × 20 features


def test_add_embeddings_with_pooling(sample_df):
    """Test adding embeddings with different pooling options."""
    # Test each pooling option
    pooling_options = ["none", "mean", "max", "sum"]

    for pooling in pooling_options:
        result = add_embeddings(sample_df, "blosum62", pooling=pooling)
        embeddings = result["blosum62_embedding"]

        if pooling == "none":
            # Should have shape (seq_len, 20)
            assert embeddings.iloc[0].shape == (7, 20)
        else:
            # Should have shape (20,)
            assert embeddings.iloc[0].shape == (20,)


def test_add_embeddings_with_max_length(sample_df):
    """Test adding embeddings with max_length constraint."""
    # Limit to first 3 residues
    result = add_embeddings(sample_df, "blosum62", max_length=3)

    # Check the embeddings
    embeddings = result["blosum62_embedding"]
    assert embeddings.iloc[0].shape == (3, 20)  # 3 residues × 20 features

    # Check that shorter sequences are not affected
    shorter_df = pd.DataFrame({"sequence": ["AC"]})
    result = add_embeddings(shorter_df, "blosum62", max_length=5)
    assert result["blosum62_embedding"].iloc[0].shape == (
        2,
        20,
    )  # 2 residues × 20 features


def test_get_embeddings(sample_df):
    """Test retrieving embeddings from a DataFrame."""
    # Add embeddings
    df_with_emb = add_embeddings(sample_df, "aac")

    # Get embeddings as dictionary
    emb_dict = get_embeddings(df_with_emb, "aac")
    assert isinstance(emb_dict, dict)
    assert len(emb_dict) == 3
    assert isinstance(emb_dict[0], np.ndarray)

    # Get embeddings as array
    emb_array = get_embeddings(df_with_emb, "aac", as_array=True)
    assert isinstance(emb_array, np.ndarray)
    assert emb_array.shape == (3, 20)  # 3 sequences × 20 features


def test_get_embeddings_error_handling():
    """Test error handling in get_embeddings."""
    df = pd.DataFrame({"sequence": ["ACDEF"]})

    # Try to get embeddings that don't exist
    with pytest.raises(ValueError):
        get_embeddings(df, "blosum62")

    # Add embeddings with different shapes
    df1 = pd.DataFrame(
        {
            "sequence": ["ACDEF", "K"],
            "blosum62_embedding": [np.zeros((5, 20)), np.zeros((1, 20))],
        }
    )

    # Try to get as array when shapes differ
    with pytest.raises(ValueError):
        get_embeddings(df1, "blosum62", as_array=True)
