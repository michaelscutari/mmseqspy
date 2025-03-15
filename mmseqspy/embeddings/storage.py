"""Storage utilities for protein sequence embeddings."""

import os
import numpy as np
import pandas as pd
import h5py
from typing import List, Dict, Optional


def store_embeddings_in_df(
    df: pd.DataFrame, embeddings: List[np.ndarray], embedding_col: str
) -> pd.DataFrame:
    """
    Store embeddings directly in a DataFrame column.

    Args:
        df: DataFrame to add embeddings to
        embeddings: List of embedding arrays
        embedding_col: Column name for storing embeddings

    Returns:
        DataFrame with embeddings added
    """
    result_df = df.copy()
    result_df[embedding_col] = embeddings

    # Add shape information column
    if len(embeddings) > 0:
        result_df[f"{embedding_col}_shape"] = str(embeddings[0].shape)

    return result_df


def store_embeddings_in_hdf(
    embeddings: List[np.ndarray],
    protein_ids: List[str],
    embedding_type: str,
    hdf_path: str,
) -> List[str]:
    """
    Store embeddings in an HDF5 file by protein ID.

    Args:
        embeddings: List of embedding arrays
        protein_ids: List of protein IDs (one per embedding)
        embedding_type: Type of embedding (used as group name)
        hdf_path: Path to HDF5 file

    Returns:
        List of reference strings pointing to stored embeddings
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(hdf_path)), exist_ok=True)

    # Open HDF5 file
    with h5py.File(hdf_path, "a") as f:
        # Create group for embedding type if it doesn't exist
        if embedding_type not in f:
            f.create_group(embedding_type)

        # Store each embedding
        references = []
        for protein_id, embedding in zip(protein_ids, embeddings):
            # Generate reference
            ref = f"{embedding_type}/{protein_id}"
            references.append(ref)

            # Store embedding with compression
            if protein_id in f[embedding_type]:
                del f[embedding_type][protein_id]  # Replace if exists

            f[embedding_type].create_dataset(
                protein_id, data=embedding, compression="gzip", compression_opts=4
            )

    return references


def get_embeddings_from_hdf(references: List[str], hdf_path: str) -> List[np.ndarray]:
    """
    Retrieve embeddings from an HDF5 file.

    Args:
        references: List of reference strings (embedding_type/protein_id)
        hdf_path: Path to HDF5 file

    Returns:
        List of embedding arrays
    """
    embeddings = []

    with h5py.File(hdf_path, "r") as f:
        for ref in references:
            if ref and "/" in ref:  # Validate reference format
                embedding = f[ref][:]
                embeddings.append(embedding)
            else:
                embeddings.append(None)

    return embeddings


def get_embeddings_from_df(df: pd.DataFrame, embedding_col: str) -> List[np.ndarray]:
    """
    Retrieve embeddings from a DataFrame column.

    Args:
        df: DataFrame containing embeddings
        embedding_col: Column name with embeddings

    Returns:
        List of embedding arrays
    """
    if embedding_col not in df.columns:
        raise ValueError(f"Embedding column '{embedding_col}' not found in DataFrame")

    return df[embedding_col].tolist()


def list_embeddings_in_hdf(
    hdf_path: str, embedding_type: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    List embeddings stored in an HDF5 file.

    Args:
        hdf_path: Path to HDF5 file
        embedding_type: Optional filter by embedding type

    Returns:
        Dict mapping embedding types to lists of protein IDs
    """
    if not os.path.exists(hdf_path):
        return {}

    result = {}
    with h5py.File(hdf_path, "r") as f:
        # If embedding type specified, list only that group
        if embedding_type:
            if embedding_type in f:
                result[embedding_type] = list(f[embedding_type].keys())
            else:
                result[embedding_type] = []
        # Otherwise list all groups
        else:
            for key in f.keys():
                result[key] = list(f[key].keys())

    return result
