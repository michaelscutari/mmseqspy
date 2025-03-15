"""Tests for embedding utility functions."""

import pytest
import numpy as np
from mmseqspy.embeddings.utils import (
    pad_sequences,
    normalize_embeddings,
    compare_embeddings,
    concat_embeddings,
)


def test_pad_sequences():
    """Test sequence padding."""
    # Test sequences
    sequences = ["ACDEF", "KL", "MNPQRST"]

    # Pad to length 7 (longest sequence)
    padded = pad_sequences(sequences)
    assert len(padded) == 3
    assert padded[0] == "ACDEFXX"  # Post-padded
    assert padded[1] == "KLXXXXX"  # Post-padded
    assert padded[2] == "MNPQRST"  # No padding needed

    # Pad to specific length
    padded = pad_sequences(sequences, max_length=10)
    assert padded[0] == "ACDEFXXXXX"
    assert padded[1] == "KLXXXXXXXX"
    assert padded[2] == "MNPQRSTXXX"

    # Pre-padding
    padded = pad_sequences(sequences, padding="pre")
    assert padded[0] == "XXACDEF"
    assert padded[1] == "XXXXXKL"
    assert padded[2] == "MNPQRST"

    # Truncation
    padded = pad_sequences(sequences, max_length=3)
    assert padded[0] == "ACD"
    assert padded[1] == "KLX"
    assert padded[2] == "MNP"


def test_normalize_embeddings():
    """Test embedding normalization."""
    # Test embeddings
    embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Min-max normalization
    norm_minmax = normalize_embeddings(embeddings, method="minmax")
    assert norm_minmax.shape == embeddings.shape
    assert np.allclose(np.min(norm_minmax, axis=0), 0)
    assert np.allclose(np.max(norm_minmax, axis=0), 1)

    # Z-score normalization
    norm_zscore = normalize_embeddings(embeddings, method="zscore")
    assert norm_zscore.shape == embeddings.shape
    assert np.allclose(np.mean(norm_zscore, axis=0), 0)
    assert np.allclose(np.std(norm_zscore, axis=0), 1)

    # L2 normalization
    norm_l2 = normalize_embeddings(embeddings, method="l2")
    assert norm_l2.shape == embeddings.shape
    # Check that rows have unit length
    norms = np.sqrt(np.sum(norm_l2**2, axis=1))
    assert np.allclose(norms, 1)


def test_compare_embeddings():
    """Test embedding comparison metrics."""
    # Test embeddings
    emb1 = np.array([1, 0, 0])
    emb2 = np.array([0, 1, 0])
    emb3 = np.array([1, 1, 0])

    # Cosine similarity
    assert np.isclose(compare_embeddings(emb1, emb1, metric="cosine"), 1)
    assert np.isclose(compare_embeddings(emb1, emb2, metric="cosine"), 0)
    assert np.isclose(compare_embeddings(emb1, emb3, metric="cosine"), 1 / np.sqrt(2))

    # Euclidean distance
    assert np.isclose(compare_embeddings(emb1, emb1, metric="euclidean"), 0)
    assert np.isclose(compare_embeddings(emb1, emb2, metric="euclidean"), -np.sqrt(2))

    # Dot product
    assert np.isclose(compare_embeddings(emb1, emb1, metric="dot"), 1)
    assert np.isclose(compare_embeddings(emb1, emb2, metric="dot"), 0)
    assert np.isclose(compare_embeddings(emb1, emb3, metric="dot"), 1)

    # Error handling for different shapes
    with pytest.raises(ValueError):
        compare_embeddings(emb1, np.array([1, 2, 3, 4]))


def test_concat_embeddings():
    """Test embedding concatenation."""
    # Test embeddings
    emb1 = np.array([[1, 2], [3, 4]])
    emb2 = np.array([[5, 6], [7, 8]])

    # Concatenate
    concat = concat_embeddings([emb1, emb2])
    assert concat.shape == (2, 4)
    assert np.array_equal(concat, np.array([[1, 2, 5, 6], [3, 4, 7, 8]]))

    # Concatenate embeddings with different shapes but same first dimension
    emb3 = np.array([[9, 10, 11], [12, 13, 14]])
    concat = concat_embeddings([emb1, emb3])
    assert concat.shape == (2, 5)
    assert np.array_equal(concat, np.array([[1, 2, 9, 10, 11], [3, 4, 12, 13, 14]]))

    # Error handling for different first dimensions
    emb4 = np.array([[1, 2, 3]])
    with pytest.raises(ValueError):
        concat_embeddings([emb1, emb4])


def test_normalize_embeddings_zero_range():
    """Test normalize_embeddings with arrays that have zero range."""
    from mmseqspy.embeddings.utils import normalize_embeddings
    import numpy as np

    # Create an array with zero range in one dimension
    embeddings = np.array([[1, 2, 3], [1, 4, 6], [1, 8, 9]])
    # First column has all the same value

    # Should handle this gracefully
    result = normalize_embeddings(embeddings, method="minmax")
    assert np.all(result[:, 0] == 0)  # All zeros for constant column
    assert np.min(result[:, 1]) == 0 and np.max(result[:, 1]) == 1  # Normalized


def test_compare_embeddings_zero_norm():
    """Test compare_embeddings with zero-norm vectors."""
    from mmseqspy.embeddings.utils import compare_embeddings
    import numpy as np

    # Create a zero vector
    zero_emb = np.zeros(5)
    non_zero_emb = np.ones(5)

    # Should handle division by zero gracefully
    result = compare_embeddings(zero_emb, non_zero_emb, metric="cosine")
    assert result == 0  # Expected behavior for zero vector
