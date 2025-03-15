"""Protein sequence embedding functionality for MMseqsPy."""

from .api import (
    embed_sequences,
    get_embeddings,
    list_available_embedders,
    register_embedder,
)
from .baseline import (
    BLOSUMEmbedder,
    AACompositionEmbedder,
    PropertyEmbedder,
    OneHotEmbedder,
    BLOSUM90Embedder,
    DiAACompositionEmbedder,
)
from .reduction import reduce_dimensions, apply_reducer, save_reducer, load_reducer
from .storage import (
    store_embeddings_in_df,
    store_embeddings_in_hdf,
    get_embeddings_from_df,
    get_embeddings_from_hdf,
    list_embeddings_in_hdf,
)


# Convenience functions for common embedding types
def blosum62(df, sequence_col="sequence", **kwargs):
    """Add BLOSUM62 embeddings to DataFrame."""
    return embed_sequences(df, "blosum62", sequence_col, **kwargs)


def blosum90(df, sequence_col="sequence", **kwargs):
    """Add BLOSUM90 embeddings to DataFrame."""
    return embed_sequences(df, "blosum90", sequence_col, **kwargs)


def aac(df, sequence_col="sequence", **kwargs):
    """Add amino acid composition embeddings to DataFrame."""
    return embed_sequences(df, "aac", sequence_col, **kwargs)


def property_embedding(df, sequence_col="sequence", **kwargs):
    """Add amino acid property embeddings to DataFrame."""
    return embed_sequences(df, "property", sequence_col, **kwargs)


def onehot(df, sequence_col="sequence", **kwargs):
    """Add one-hot encoded embeddings to DataFrame."""
    return embed_sequences(df, "onehot", sequence_col, **kwargs)


__all__ = [
    "embed_sequences",
    "get_embeddings",
    "list_available_embedders",
    "register_embedder",
    "BLOSUMEmbedder",
    "AACompositionEmbedder",
    "PropertyEmbedder",
    "OneHotEmbedder",
    "BLOSUM90Embedder",
    "DiAACompositionEmbedder",
    "reduce_dimensions",
    "apply_reducer",
    "save_reducer",
    "load_reducer",
    "store_embeddings_in_df",
    "store_embeddings_in_hdf",
    "get_embeddings_from_df",
    "get_embeddings_from_hdf",
    "list_embeddings_in_hdf",
    "blosum62",
    "blosum90",
    "aac",
    "property_embedding",
    "onehot",
]
