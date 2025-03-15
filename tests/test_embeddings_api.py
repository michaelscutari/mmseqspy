"""Tests for the embeddings API functionality."""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from mmseqspy.embeddings import (
    register_embedder,
    list_available_embedders,
    embed_sequences,
    get_embeddings,
    BLOSUMEmbedder,
)


@pytest.fixture
def sample_df():
    """Create a robust sample DataFrame with diverse sequences for testing."""
    return pd.DataFrame(
        {
            "id": [
                "seq1",
                "seq2",
                "seq3",
                "seq4",
                "seq5",
                "seq6",
                "seq7",
                "seq8",
                "seq9",
                "seq10",
            ],
            "sequence": [
                "ACDEFGH",  # Standard mix of amino acids
                "KLTWYV",  # Hydrophobic-rich
                "MNPQRS",  # Polar-rich
                "ACDEGHK",  # Acidic-rich
                "KLMNPQR",  # Basic-rich
                "STUVWYA",  # Mixed properties
                "GGGGGGG",  # Homopolymer
                "ACACACA",  # Repeating pattern
                "RKRKRKR",  # Charged repeating pattern
                "ACDKLMN",  # Mixed amino acids
            ],
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


# Create a simple test embedder for registration tests
class SampleEmbedder(BLOSUMEmbedder):
    """Test embedder class."""

    def __init__(self):
        super().__init__()


# SECTION 1: Basic API Tests


def test_list_available_embedders():
    """Test listing available embedders."""
    embedders = list_available_embedders()
    assert isinstance(embedders, list)
    assert len(embedders) > 0
    assert "blosum62" in embedders
    assert "aac" in embedders
    assert "property" in embedders


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
    result = embed_sequences(df, "custom")
    assert "custom_embedding" in result.columns


# SECTION 2: DataFrame Storage Tests


def test_embed_sequences_dataframe(sample_df):
    """Test embedding sequences and storing in DataFrame."""
    # Embed sequences
    result_df = embed_sequences(sample_df, embedding_type="blosum62")

    # Check that embeddings were stored
    assert "blosum62_embedding" in result_df.columns
    assert len(result_df) == len(sample_df)

    # Check that the original DataFrame was not modified
    assert "blosum62_embedding" not in sample_df.columns

    # Check embedding shapes
    embeddings = result_df["blosum62_embedding"]
    assert len(embeddings) == 10

    # Check the dimensions
    first_emb = embeddings.iloc[0]
    assert first_emb.shape == (7, 20)  # 7 residues × 20 features


def test_embed_sequences_with_pooling(sample_df):
    """Test embedding sequences with different pooling options."""
    # Test each pooling option
    pooling_options = ["none", "mean", "max", "sum"]

    for pooling in pooling_options:
        result_df = embed_sequences(sample_df, "blosum62", pooling=pooling)
        embeddings = result_df["blosum62_embedding"]

        if pooling == "none":
            # Should have shape (seq_len, 20)
            assert embeddings.iloc[0].shape == (7, 20)
        else:
            # Should have shape (20,)
            assert embeddings.iloc[0].shape == (20,)


def test_embed_sequences_with_max_length(sample_df):
    """Test embedding sequences with max_length constraint."""
    # Limit to first 3 residues
    result_df = embed_sequences(sample_df, "blosum62", max_length=3)

    # Check the embeddings
    embeddings = result_df["blosum62_embedding"]
    assert embeddings.iloc[0].shape == (3, 20)  # 3 residues × 20 features

    # Check that shorter sequences are not affected
    shorter_df = pd.DataFrame({"sequence": ["AC"]})
    result = embed_sequences(shorter_df, "blosum62", max_length=5)
    assert result["blosum62_embedding"].iloc[0].shape == (
        2,
        20,
    )  # 2 residues × 20 features


# SECTION 3: HDF5 Storage Tests


def test_embed_sequences_hdf(sample_df, temp_hdf_path):
    """Test embedding sequences and storing in HDF5."""
    # Embed sequences with HDF5 storage
    result_df = embed_sequences(
        sample_df, embedding_type="blosum62", use_hdf=True, hdf_path=temp_hdf_path
    )

    # Check that references were stored
    assert "blosum62_ref" in result_df.columns
    assert len(result_df) == len(sample_df)

    # Check references format
    first_ref = result_df["blosum62_ref"].iloc[0]
    assert first_ref.startswith("blosum62/")

    # Try to get embeddings
    embeddings = get_embeddings(
        result_df, embedding_type="blosum62", hdf_path=temp_hdf_path
    )

    # Check embeddings
    assert len(embeddings) == len(sample_df)
    assert embeddings[0].shape == (7, 20)  # 7 residues × 20 features


# SECTION 4: Dimension Reduction Tests


def test_embed_sequences_with_reduction(sample_df):
    """Test embedding sequences with dimension reduction."""
    # Embed sequences with PCA reduction
    result_df = embed_sequences(
        sample_df, embedding_type="blosum62", reduce_dim="pca", n_components=5
    )

    # Check that reduced embeddings were stored
    assert "blosum62_pca5_embedding" in result_df.columns

    # Check embedding shapes
    first_embedding = result_df["blosum62_pca5_embedding"].iloc[0]
    assert len(first_embedding) == 5  # Reduced to 5 components


def test_embed_sequences_with_reduction_and_hdf(sample_df, temp_hdf_path):
    """Test embedding sequences with dimension reduction and HDF5 storage."""
    # Embed sequences with PCA reduction and HDF5 storage
    result_df = embed_sequences(
        sample_df,
        embedding_type="blosum62",
        reduce_dim="pca",
        n_components=5,
        use_hdf=True,
        hdf_path=temp_hdf_path,
    )

    # Check that references were stored
    assert "blosum62_pca5_ref" in result_df.columns

    # Get embeddings
    embeddings = get_embeddings(
        result_df, embedding_type="blosum62_pca5", hdf_path=temp_hdf_path
    )

    # Check embeddings
    assert len(embeddings) == len(sample_df)
    assert len(embeddings[0]) == 5  # Reduced to 5 components


# SECTION 5: Get Embeddings Tests


def test_get_embeddings_from_dataframe(sample_df):
    """Test retrieving embeddings from DataFrame."""
    # First, embed sequences
    df_with_emb = embed_sequences(sample_df, embedding_type="blosum62")

    # Get embeddings as dictionary
    embeddings = get_embeddings(df_with_emb, "blosum62")
    assert len(embeddings) == 10
    assert embeddings[0].shape == (7, 20)  # 7 residues × 20 features

    # Get embeddings as array
    emb_array = get_embeddings(df_with_emb, "blosum62", as_array=True)
    assert isinstance(emb_array, np.ndarray)
    assert len(emb_array) == 10  # 10 samples
    # Check the shape of the first embedding to ensure it's as expected
    assert emb_array[0].shape == (7, 20)  # 7 residues × 20 features


def test_get_embeddings_error_handling(sample_df):
    """Test error handling in get_embeddings."""
    # Try to get embeddings that don't exist
    with pytest.raises(ValueError):
        get_embeddings(sample_df, "blosum62")

    # Try to get embeddings from HDF5 without path
    df_with_refs = embed_sequences(
        sample_df,
        embedding_type="blosum62",
        use_hdf=True,
        hdf_path="temp.h5",  # This file won't exist, but that's fine for this test
    )

    with pytest.raises(ValueError):
        get_embeddings(df_with_refs, "blosum62")  # No hdf_path provided


def test_convenience_embedding_functions(sample_df):
    """Test the convenience functions for different embedding types."""
    from mmseqspy.embeddings import blosum62, blosum90, aac, property_embedding, onehot

    # Test each convenience function
    assert "blosum62_embedding" in blosum62(sample_df, sequence_col="sequence").columns
    assert "blosum90_embedding" in blosum90(sample_df, sequence_col="sequence").columns
    assert "aac_embedding" in aac(sample_df, sequence_col="sequence").columns
    assert (
        "property_embedding"
        in property_embedding(sample_df, sequence_col="sequence").columns
    )
    assert "onehot_embedding" in onehot(sample_df, sequence_col="sequence").columns


def test_register_embedder_validation():
    """Test validation when registering embedders with invalid classes."""
    from mmseqspy.embeddings import register_embedder

    # Create a class that doesn't inherit from BaseEmbedder
    class InvalidEmbedder:
        def generate(self, sequence):
            return None

    # Try to register it - should raise ValueError
    with pytest.raises(ValueError):
        register_embedder("invalid", InvalidEmbedder)


def test_embed_sequences_validation():
    """Test validation in embed_sequences function."""
    from mmseqspy.embeddings import embed_sequences
    import pandas as pd

    sample_df = pd.DataFrame({"sequence": ["ACDEF"]})

    # Test with use_hdf=True but no hdf_path
    with pytest.raises(ValueError):
        embed_sequences(sample_df, "blosum62", use_hdf=True)

    # Test with invalid reduction method
    with pytest.raises(ValueError):
        embed_sequences(sample_df, "blosum62", reduce_dim="invalid_method")


def test_embed_sequences_with_nonuniform_embeddings(sample_df):
    """Test dimension reduction on non-uniform embeddings."""
    from mmseqspy.embeddings import embed_sequences
    import numpy as np

    # Create a custom embedder that returns non-uniform embeddings
    from mmseqspy.embeddings.baseline import BaseEmbedder

    class NonUniformEmbedder(BaseEmbedder):
        def generate(self, sequence, pooling="none", max_length=None):
            # Return 2D for first sequence, 1D for others
            if len(sequence) > 6:
                return np.ones((3, 5))
            return np.ones(5)

    # Register the custom embedder
    from mmseqspy.embeddings import register_embedder

    register_embedder("non_uniform", NonUniformEmbedder)

    # This should trigger the dimension reduction code path for non-uniform embeddings
    result = embed_sequences(sample_df, "non_uniform", reduce_dim="pca", n_components=3)

    assert "non_uniform_pca3_embedding" in result.columns
