"""Tests for baseline embedders."""

import pytest
import numpy as np
from mmseqspy.embeddings import (
    BLOSUMEmbedder,
    AACompositionEmbedder,
    PropertyEmbedder,
    OneHotEmbedder,
)
from mmseqspy.embeddings.matrices import PROPERTY_SCALES


def test_blosum_embedder():
    """Test the BLOSUM embedder."""
    # Create embedder
    embedder = BLOSUMEmbedder()

    # Test single sequence
    sequence = "ACDEFGH"
    embedding = embedder.generate(sequence)

    # Check shape
    assert embedding.shape == (7, 20)  # 7 residues × 20 features

    # Test with pooling
    emb_mean = embedder.generate(sequence, pooling="mean")
    assert emb_mean.shape == (20,)

    # Test with max_length
    emb_truncated = embedder.generate(sequence, max_length=3)
    assert emb_truncated.shape == (3, 20)


def test_aac_embedder():
    """Test the amino acid composition embedder."""
    # Create embedder
    embedder = AACompositionEmbedder()

    # Test single sequence
    sequence = "ACDEFGH"
    embedding = embedder.generate(sequence)

    # Check shape
    assert embedding.shape == (20,)  # 20 amino acids

    # Check that it's a probability distribution (sums to 1)
    assert np.isclose(np.sum(embedding), 1.0)

    # Test with k=2 (dipeptides)
    di_embedder = AACompositionEmbedder(k=2)
    di_embedding = di_embedder.generate(sequence)
    assert di_embedding.shape == (400,)  # 20×20 dipeptides


def test_property_embedder():
    """Test the property embedder."""
    # Create embedder with default properties
    embedder = PropertyEmbedder()

    # Test single sequence
    sequence = "ACDEFGH"
    embedding = embedder.generate(sequence)

    # Check shape - should have default 4 properties
    assert embedding.shape == (7, 4)  # 7 residues × 4 properties

    # Test with specific properties
    props = ["hydrophobicity", "charge"]
    embedder = PropertyEmbedder(properties=props)
    embedding = embedder.generate(sequence)

    # Check shape
    assert embedding.shape == (7, 2)  # 7 residues × 2 properties

    # Test with pooling
    emb_mean = embedder.generate(sequence, pooling="mean")
    assert emb_mean.shape == (2,)  # 2 properties

    # Test with unknown property
    with pytest.raises(ValueError):
        PropertyEmbedder(properties=["unknown_property"])

    # Check all available properties work
    all_props = list(PROPERTY_SCALES.keys())
    embedder = PropertyEmbedder(properties=all_props)
    embedding = embedder.generate(sequence)
    assert embedding.shape[1] == len(all_props)


def test_onehot_embedder():
    """Test the one-hot embedder."""
    # Create embedder
    embedder = OneHotEmbedder()

    # Test single sequence
    sequence = "ACDEFGH"
    embedding = embedder.generate(sequence)

    # Check shape
    assert embedding.shape == (7, 20)  # 7 residues × 20 amino acids

    # Check that each row is one-hot
    assert np.all(np.sum(embedding, axis=1) == 1)

    # Test with non-standard amino acids
    sequence = "ACBUX"  # B, U, X are non-standard
    embedding = embedder.generate(sequence)

    # Check shape
    assert embedding.shape == (5, 20)

    # Check that rows for non-standard amino acids are all zeros
    assert np.sum(embedding[2]) == 0  # 'B' is non-standard
    assert np.sum(embedding[3]) == 0  # 'U' is non-standard
    assert np.sum(embedding[4]) == 0  # 'X' is non-standard


def test_pooling_options():
    """Test different pooling options."""
    sequence = "ACDEFGHIKLMNPQRSTVWY"

    # Using BLOSUM embedder
    embedder = BLOSUMEmbedder()

    # No pooling - should get per-residue embedding
    emb_none = embedder.generate(sequence, pooling="none")
    assert emb_none.shape == (20, 20)  # 20 residues × 20 features

    # Mean pooling - should get sequence-level embedding
    emb_mean = embedder.generate(sequence, pooling="mean")
    assert emb_mean.shape == (20,)  # 20 features

    # Max pooling
    emb_max = embedder.generate(sequence, pooling="max")
    assert emb_max.shape == (20,)

    # Sum pooling
    emb_sum = embedder.generate(sequence, pooling="sum")
    assert emb_sum.shape == (20,)

    # Auto pooling (should use embedder's default)
    emb_auto = embedder.generate(sequence, pooling="auto")
    assert emb_auto.shape == emb_none.shape  # BLOSUM default is "none"


def test_max_length():
    """Test max_length parameter."""
    sequence = "ACDEFGHIKLMNPQRSTVWY"  # 20 residues

    # Using BLOSUM embedder
    embedder = BLOSUMEmbedder()

    # Truncate to 10 residues
    emb_truncated = embedder.generate(sequence, max_length=10)
    assert emb_truncated.shape == (10, 20)  # 10 residues × 20 features

    # Should use first 10 residues
    emb_first10 = embedder.generate(sequence[:10])
    assert np.array_equal(emb_truncated, emb_first10)


def test_embedding_with_edge_cases():
    """Test embedders with edge case sequences."""
    # Empty sequence
    embedder = BLOSUMEmbedder()
    emb_empty = embedder.generate("")
    assert emb_empty.shape == (0, 20)

    # Very short sequence
    emb_short = embedder.generate("A")
    assert emb_short.shape == (1, 20)

    # Sequence with invalid amino acids
    emb_invalid = embedder.generate("ACDEFGH*#@")
    assert emb_invalid.shape == (10, 20)
