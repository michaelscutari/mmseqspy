"""Core API for protein sequence embeddings."""

import pandas as pd
import numpy as np
from typing import List, Union, Optional
from .reduction import reduce_dimensions
from .storage import (
    store_embeddings_in_df,
    store_embeddings_in_hdf,
    get_embeddings_from_df,
    get_embeddings_from_hdf,
)

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
    Get the embedder for a given embedding type.

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

    # Return the embedder class (not an instance)
    return _EMBEDDER_REGISTRY[name]


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


def embed_sequences(
    df: pd.DataFrame,
    embedding_type: str,
    sequence_col: str = "sequence",
    pooling: str = "auto",
    max_length: Optional[int] = None,
    use_hdf: bool = False,
    hdf_path: Optional[str] = None,
    reduce_dim: Optional[str] = None,
    n_components: int = 50,
    **kwargs,
) -> pd.DataFrame:
    """
    Generate embeddings for sequences and store appropriately.

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
        max_length: Maximum sequence length to consider
        use_hdf: Whether to use HDF5 storage instead of DataFrame
        hdf_path: Path to HDF5 file (required if use_hdf=True)
        reduce_dim: Optional dimension reduction method
        n_components: Number of components for dimension reduction
        **kwargs: Additional parameters for the specific embedder

    Returns:
        DataFrame with embeddings or embedding references
    """
    # Validate HDF path if HDF storage is requested
    if use_hdf and not hdf_path:
        raise ValueError("hdf_path must be provided when use_hdf=True")

    # Get embedder
    embedder_class_or_instance = get_embedder(embedding_type)

    # Check if it's a class or instance
    if isinstance(embedder_class_or_instance, type):
        # It's a class, instantiate it
        embedder = embedder_class_or_instance(**kwargs)
    else:
        # It's already an instance
        embedder = embedder_class_or_instance

    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()

    # Generate embeddings for each sequence
    sequences = df[sequence_col].tolist()
    embeddings = [
        embedder.generate(seq, pooling=pooling, max_length=max_length)
        for seq in sequences
    ]

    # Apply dimension reduction if requested
    reducer = None
    if reduce_dim:
        # For dimension reduction, we need uniformly shaped data
        # Force pooling if embeddings are not already uniform
        if any(emb.ndim > 1 for emb in embeddings):
            # Apply mean pooling to get a single vector per sequence
            embeddings = [
                np.mean(emb, axis=0) if emb.ndim > 1 else emb for emb in embeddings
            ]

        # Convert to 2D array for reduction algorithms
        X = np.vstack(embeddings)
        reduced_X, reducer = reduce_dimensions(
            X, method=reduce_dim, n_components=n_components
        )
        embeddings = list(reduced_X)

        # Update embedding type to indicate reduction
        embedding_type = f"{embedding_type}_{reduce_dim}{n_components}"

    # Store embeddings
    if use_hdf:
        # Store in HDF5
        references = store_embeddings_in_hdf(
            embeddings=embeddings,
            protein_ids=df.index.astype(str).tolist(),
            embedding_type=embedding_type,
            hdf_path=hdf_path,
        )

        # Add references to DataFrame
        result_df[f"{embedding_type}_ref"] = references
    else:
        # Store directly in DataFrame
        result_df = store_embeddings_in_df(
            df=result_df,
            embeddings=embeddings,
            embedding_col=f"{embedding_type}_embedding",
        )

    # Return updated DataFrame
    return result_df


def get_embeddings(
    df: pd.DataFrame,
    embedding_type: str,
    as_array: bool = False,
    hdf_path: Optional[str] = None,
) -> Union[List[np.ndarray], np.ndarray]:
    """
    Retrieve embeddings from a DataFrame or HDF5 file.

    Args:
        df: DataFrame containing embeddings or references
        embedding_type: Type of embedding to retrieve
        as_array: Whether to return as a single numpy array (uses object dtype for variable-length sequences)
        hdf_path: Path to HDF5 file (required if references are stored)

    Returns:
        List of embedding arrays or a numpy object array containing embeddings
    """
    # Check for direct embeddings in DataFrame
    embedding_col = f"{embedding_type}_embedding"
    reference_col = f"{embedding_type}_ref"

    if embedding_col in df.columns:
        # Retrieve from DataFrame
        embeddings = get_embeddings_from_df(df, embedding_col)
    elif reference_col in df.columns:
        # Validate HDF path
        if not hdf_path:
            raise ValueError("hdf_path must be provided when retrieving from HDF5")

        # Retrieve from HDF5
        references = df[reference_col].tolist()
        embeddings = get_embeddings_from_hdf(references, hdf_path)
    else:
        raise ValueError(
            f"No embeddings found for type '{embedding_type}' in DataFrame"
        )

    # Return embeddings as list or numpy array based on user preference
    if not as_array:
        return embeddings

    # Convert to numpy array using object dtype to handle variable-length sequences
    # This preserves the original shape of each embedding
    return np.array(embeddings, dtype=object)
