"""Protein sequence embedding functionality for MMseqsPy."""

from .api import (
    add_embeddings,
    get_embeddings,
    list_available_embedders,
    register_embedder,
)
from .baseline import (
    BLOSUMEmbedder,
    AACompositionEmbedder,
    PropertyEmbedder,
    OneHotEmbedder,
)


# Convenience functions for common embedding types
def blosum62(df, sequence_col="sequence", **kwargs):
    """Add BLOSUM62 embeddings to DataFrame."""
    return add_embeddings(df, "blosum62", sequence_col, **kwargs)


def aac(df, sequence_col="sequence", **kwargs):
    """Add amino acid composition embeddings to DataFrame."""
    return add_embeddings(df, "aac", sequence_col, **kwargs)


def property_embedding(df, sequence_col="sequence", **kwargs):
    """Add amino acid property embeddings to DataFrame."""
    return add_embeddings(df, "property", sequence_col, **kwargs)


def onehot(df, sequence_col="sequence", **kwargs):
    """Add one-hot encoded embeddings to DataFrame."""
    return add_embeddings(df, "onehot", sequence_col, **kwargs)


__all__ = [
    "add_embeddings",
    "get_embeddings",
    "list_available_embedders",
    "register_embedder",
    "BLOSUMEmbedder",
    "AACompositionEmbedder",
    "PropertyEmbedder",
    "OneHotEmbedder",
    "blosum62",
    "aac",
    "property_embedding",
    "onehot",
]
