"""Utilities for generating synthetic test data for MMseqsPy."""

import random
import pandas as pd
import numpy as np
from typing import List, Optional

# Standard amino acids used in protein sequences
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def generate_random_sequence(length: int, seed: Optional[int] = None) -> str:
    """
    Generate a random protein sequence of specified length.

    Parameters:
        length: Length of the sequence to generate
        seed: Random seed for reproducibility

    Returns:
        A random protein sequence string
    """
    if seed is not None:
        random.seed(seed)

    return "".join(random.choices(AMINO_ACIDS, k=length))


def create_sequence_variant(
    sequence: str, identity: float, seed: Optional[int] = None
) -> str:
    """
    Create a variant of a sequence with a specified sequence identity.

    Parameters:
        sequence: Original sequence
        identity: Target sequence identity (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        A new sequence with the target identity to the original
    """
    if not 0 <= identity <= 1:
        raise ValueError(f"Identity must be between 0 and 1, got {identity}")

    if seed is not None:
        random.seed(seed)

    # Calculate number of positions to mutate
    seq_length = len(sequence)
    mutations = int(round(seq_length * (1 - identity)))

    # If no mutations are needed (100% identity), return the original
    if mutations == 0:
        return sequence

    # Convert to list for mutation
    seq_list = list(sequence)

    # Select positions to mutate (without repeats)
    positions = random.sample(range(seq_length), mutations)

    # Perform mutations
    for pos in positions:
        # Exclude the current amino acid to ensure a change
        alt_amino_acids = AMINO_ACIDS.replace(seq_list[pos], "")
        seq_list[pos] = random.choice(alt_amino_acids)

    return "".join(seq_list)


def create_cluster_dataset(
    n_clusters: int = 5,
    seqs_per_cluster: int = 10,
    seq_length: int = 100,
    within_identity: float = 0.9,
    between_identity: float = 0.3,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Create a dataset with well-defined sequence clusters.

    Parameters:
        n_clusters: Number of clusters to create
        seqs_per_cluster: Number of sequences in each cluster
        seq_length: Length of each sequence
        within_identity: Target identity within clusters (0.0-1.0)
        between_identity: Target identity between clusters (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sequence data organized in predictable clusters
    """
    if not 0 <= within_identity <= 1:
        raise ValueError(
            f"within_identity must be between 0 and 1, got {within_identity}"
        )
    if not 0 <= between_identity <= 1:
        raise ValueError(
            f"between_identity must be between 0 and 1, got {between_identity}"
        )
    if within_identity <= between_identity:
        raise ValueError(
            f"within_identity ({within_identity}) must be greater than "
            f"between_identity ({between_identity}) for clear cluster separation"
        )

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Initialize data storage
    data = []

    # Create each cluster
    for cluster_idx in range(n_clusters):
        # Create a seed sequence for this cluster
        cluster_seed = generate_random_sequence(
            seq_length, seed=None if seed is None else seed + cluster_idx
        )

        # Create variants within the cluster
        for seq_idx in range(seqs_per_cluster):
            seq_id = f"cluster{cluster_idx + 1}_seq{seq_idx + 1}"

            # First sequence in cluster is the prototype (seed sequence)
            if seq_idx == 0:
                sequence = cluster_seed
            else:
                # Create variant with high similarity to cluster seed
                sequence = create_sequence_variant(
                    cluster_seed,
                    within_identity,
                    seed=None if seed is None else seed + cluster_idx * 100 + seq_idx,
                )

            # Add some metadata for testing
            data.append(
                {
                    "id": seq_id,
                    "sequence": sequence,
                    "cluster_id": cluster_idx + 1,
                    "seq_length": len(sequence),
                    "property_value": np.random.normal(
                        cluster_idx, 0.5
                    ),  # Cluster-correlated property
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(data)

    return df


def create_identity_test_dataset(
    base_seq_length: int = 100,
    identity_levels: List[float] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
    variants_per_level: int = 3,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Create a dataset with sequences at specific identity levels to a base sequence.
    Useful for testing clustering at different identity thresholds.

    Parameters:
        base_seq_length: Length of the base sequence
        identity_levels: List of identity levels to create variants for
        variants_per_level: Number of variants to create at each identity level
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sequences at controlled identity levels
    """
    if seed is not None:
        random.seed(seed)

    # Create base sequence
    base_sequence = generate_random_sequence(base_seq_length, seed=seed)

    # Initialize data storage
    data = []

    # Add base sequence
    data.append(
        {
            "id": "base_seq",
            "sequence": base_sequence,
            "true_identity": 1.0,
            "group": "base",
        }
    )

    # Create variants at each identity level
    for identity in identity_levels:
        if identity == 1.0:  # Skip duplicates of base sequence
            continue

        for variant_idx in range(variants_per_level):
            # Create variant with specified identity to base sequence
            variant = create_sequence_variant(
                base_sequence,
                identity,
                seed=None if seed is None else seed + int(identity * 100) + variant_idx,
            )

            data.append(
                {
                    "id": f"id{int(identity * 100)}_var{variant_idx + 1}",
                    "sequence": variant,
                    "true_identity": identity,
                    "group": f"identity_{int(identity * 100)}",
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(data)

    return df


def create_edge_case_dataset() -> pd.DataFrame:
    """
    Create a dataset with edge cases for testing robustness.

    Returns:
        DataFrame with edge case sequences
    """
    data = []

    # 1. Very short sequence
    data.append(
        {"id": "short_seq", "sequence": "ACDEFG", "case_type": "short_sequence"}
    )

    # 2. Very long sequence
    data.append(
        {"id": "long_seq", "sequence": "A" * 1000, "case_type": "long_sequence"}
    )

    # 3. Biased amino acid composition (all hydrophobic)
    data.append(
        {
            "id": "hydrophobic_seq",
            "sequence": "VVVIIIILLLLFFFFFWWWWMMMM",
            "case_type": "biased_composition",
        }
    )

    # 4. Repeated motif
    data.append(
        {
            "id": "repeat_seq",
            "sequence": "ACDEFACDEFACDEFACDEFACDEF",
            "case_type": "repeated_motif",
        }
    )

    # 5. Nearly identical sequences (99% identity) - Using more realistic sequences
    base_seq = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKS"
    variant_seq = (
        base_seq[:98] + "E" + base_seq[99:]
    )  # 99% identity (1 substitution in 100aa)

    data.append(
        {"id": "near_ident_1", "sequence": base_seq, "case_type": "near_identical"}
    )
    data.append(
        {"id": "near_ident_2", "sequence": variant_seq, "case_type": "near_identical"}
    )

    # 6. Exactly identical sequences
    data.append(
        {"id": "ident_1", "sequence": "ACDEFGHIKLMNPQRSTVWY", "case_type": "identical"}
    )
    data.append(
        {"id": "ident_2", "sequence": "ACDEFGHIKLMNPQRSTVWY", "case_type": "identical"}
    )

    # Convert to DataFrame
    df = pd.DataFrame(data)

    return df
