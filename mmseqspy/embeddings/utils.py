"""Utility functions for embeddings."""

import numpy as np
from typing import List, Optional


def pad_sequences(
    sequences: List[str], max_length: Optional[int] = None, padding: str = "post"
) -> List[str]:
    """
    Pad sequences to a uniform length.

    Args:
        sequences: List of amino acid sequences
        max_length: Target length for padding (default: length of longest sequence)
        padding: 'pre' or 'post' to add padding before or after

    Returns:
        List of padded sequences
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded_sequences = []
    for seq in sequences:
        if len(seq) >= max_length:
            # Truncate if longer than max_length
            padded_sequences.append(seq[:max_length])
        else:
            # Pad with 'X' if shorter
            pad_length = max_length - len(seq)
            if padding == "post":
                padded_sequences.append(seq + "X" * pad_length)
            else:
                padded_sequences.append("X" * pad_length + seq)

    return padded_sequences


def normalize_embeddings(embeddings: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize embedding values.

    Args:
        embeddings: Embedding array to normalize
        method: Normalization method:
            - 'minmax': Scale to [0, 1]
            - 'zscore': Standardize to mean=0, std=1
            - 'l2': L2 normalization (unit length)

    Returns:
        Normalized embeddings
    """
    if method == "minmax":
        # Scale to [0, 1]
        min_val = np.min(embeddings, axis=0)
        max_val = np.max(embeddings, axis=0)
        range_val = max_val - min_val
        # Avoid division by zero
        range_val[range_val == 0] = 1
        return (embeddings - min_val) / range_val

    elif method == "zscore":
        # Standardize to mean=0, std=1
        mean_val = np.mean(embeddings, axis=0)
        std_val = np.std(embeddings, axis=0)
        # Avoid division by zero
        std_val[std_val == 0] = 1
        return (embeddings - mean_val) / std_val

    elif method == "l2":
        # L2 normalization (unit length)
        norms = np.sqrt(np.sum(embeddings**2, axis=1, keepdims=True))
        # Avoid division by zero
        norms[norms == 0] = 1
        return embeddings / norms

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compare_embeddings(
    emb1: np.ndarray, emb2: np.ndarray, metric: str = "cosine"
) -> float:
    """
    Calculate similarity between two embeddings.

    Args:
        emb1: First embedding
        emb2: Second embedding
        metric: Similarity metric:
            - 'cosine': Cosine similarity
            - 'euclidean': Euclidean distance
            - 'dot': Dot product

    Returns:
        Similarity score
    """
    # Ensure both embeddings are flattened to 1D
    emb1_flat = emb1.flatten()
    emb2_flat = emb2.flatten()

    if len(emb1_flat) != len(emb2_flat):
        raise ValueError(
            f"Embeddings have different dimensions: {emb1.shape} vs {emb2.shape}"
        )

    if metric == "cosine":
        # Cosine similarity
        norm1 = np.linalg.norm(emb1_flat)
        norm2 = np.linalg.norm(emb2_flat)
        if norm1 == 0 or norm2 == 0:
            return 0  # Avoid division by zero
        return np.dot(emb1_flat, emb2_flat) / (norm1 * norm2)

    elif metric == "euclidean":
        # Euclidean distance (converted to similarity by negation)
        return -np.linalg.norm(emb1_flat - emb2_flat)

    elif metric == "dot":
        # Dot product
        return np.dot(emb1_flat, emb2_flat)

    else:
        raise ValueError(f"Unknown metric: {metric}")


def concat_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Concatenate multiple embeddings.

    Args:
        embeddings: List of embedding arrays

    Returns:
        Concatenated embedding array
    """
    # Check if all embeddings have the same first dimension
    shapes = [e.shape[0] for e in embeddings]
    if len(set(shapes)) > 1:
        raise ValueError(f"Embeddings have different first dimensions: {shapes}")

    # Flatten and concatenate
    flat_embeddings = [e.reshape(e.shape[0], -1) for e in embeddings]
    return np.concatenate(flat_embeddings, axis=1)
