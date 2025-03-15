"""Core API for protein sequence embeddings."""

import pandas as pd
import numpy as np
from typing import Dict, List, Union

# Registry of available embedders
_EMBEDDER_REGISTRY = {}


def register_embedder(name: str, embedder_class):
    """
    Register a new embedder implementation for use with add_embeddings().

    Args:
        name: String identifier for the embedding type (e.g., "blosum62")
        embedder_class: Class that implements the BaseEmbedder interface

    Returns:
        None
    """
    from .baseline import BaseEmbedder

    # Validate that the class implements the required interface
    if not issubclass(embedder_class, BaseEmbedder):
        raise ValueError(
            f"Class {embedder_class.__name__} must inherit from BaseEmbedder"
        )

    # Add to registry
    _EMBEDDER_REGISTRY[name] = embedder_class


def get_embedder(name: str):
    """
    Get an embedder instance for a given embedding type.

    Args:
        name: Embedding type name

    Returns:
        Embedder instance
    """
    if name not in _EMBEDDER_REGISTRY:
        raise ValueError(
            f"Unknown embedding type: {name}. "
            f"Available types: {list(_EMBEDDER_REGISTRY.keys())}"
        )

    # Create a new instance of the embedder class
    return _EMBEDDER_REGISTRY[name]()


def list_available_embedders() -> List[str]:
    """
    List all registered embedding types.

    Returns:
        List of available embedding type names
    """
    return list(_EMBEDDER_REGISTRY.keys())


def add_embeddings(
    df: pd.DataFrame,
    embedding_type: str,
    sequence_col: str = "sequence",
    pooling: str = "auto",
    max_length: int = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Add embeddings to a DataFrame.

    Args:
        df: DataFrame containing protein sequences
        embedding_type: Type of embedding to generate
        sequence_col: Column containing sequences
        pooling: How to handle variable-length embeddings:
            - "none": Keep per-residue embeddings
            - "mean": Average across residues
            - "max": Take maximum value for each dimension
            - "sum": Sum across residues
            - "auto": Use embedding-specific default
        max_length: Maximum sequence length to consider:
            - None: Use full sequence
            - int: Truncate to this length
        **kwargs: Additional parameters for the specific embedder

    Returns:
        DataFrame with embeddings added as a new column
    """
    # Get embedder class
    embedder_class = _EMBEDDER_REGISTRY[embedding_type]

    # Create embedder instance with passed parameters
    embedder = embedder_class(**kwargs)

    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()

    # Column name for embeddings
    embedding_col = f"{embedding_type}_embedding"

    # Generate embeddings for each sequence
    result_df[embedding_col] = result_df[sequence_col].apply(
        lambda seq: embedder.generate(seq, pooling=pooling, max_length=max_length)
    )

    # Add metadata about embedding dimensions
    if len(result_df) > 0:
        first_embedding = result_df[embedding_col].iloc[0]
        result_df[f"{embedding_type}_shape"] = str(first_embedding.shape)

    return result_df


def get_embeddings(
    df: pd.DataFrame, embedding_type: str, as_array: bool = False
) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """
    Retrieve embeddings from a DataFrame.

    Args:
        df: DataFrame containing embeddings
        embedding_type: Type of embedding to retrieve
        as_array: Whether to return as a single numpy array (requires uniform shapes)

    Returns:
        If as_array is False: Dictionary mapping row indices to embeddings
        If as_array is True: Numpy array of embeddings
    """
    embedding_col = f"{embedding_type}_embedding"

    if embedding_col not in df.columns:
        raise ValueError(f"Embedding column '{embedding_col}' not found in DataFrame")

    if not as_array:
        # Return dictionary mapping indices to embeddings
        return {idx: embedding for idx, embedding in df[embedding_col].items()}
    else:
        # Check if embeddings are all the same shape
        first_shape = df[embedding_col].iloc[0].shape
        if not all(emb.shape == first_shape for emb in df[embedding_col]):
            raise ValueError(
                "Cannot convert to array: embeddings have different shapes"
            )

        # Convert to numpy array
        return np.array(df[embedding_col].tolist())
