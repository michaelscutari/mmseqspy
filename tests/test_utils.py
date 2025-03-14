"""Utilities for generating synthetic test data for MMseqsPy."""

import random
import pandas as pd
import numpy as np
from typing import List, Optional

# Standard amino acids used in protein sequences
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Realistic amino acid frequencies based on UniProt statistics
AA_FREQUENCIES = {
    "A": 0.083,
    "R": 0.056,
    "N": 0.038,
    "D": 0.053,
    "C": 0.017,
    "Q": 0.037,
    "E": 0.063,
    "G": 0.071,
    "H": 0.023,
    "I": 0.056,
    "L": 0.093,
    "K": 0.059,
    "M": 0.023,
    "F": 0.038,
    "P": 0.049,
    "S": 0.068,
    "T": 0.053,
    "W": 0.014,
    "Y": 0.030,
    "V": 0.065,
}

# List for random.choices which needs separate lists of items and weights
AA_LIST = list(AA_FREQUENCIES.keys())
AA_WEIGHTS = list(AA_FREQUENCIES.values())


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


def generate_realistic_sequence(length: int, seed: Optional[int] = None) -> str:
    """
    Generate a protein sequence with realistic amino acid frequencies.

    Parameters:
        length: Length of the sequence to generate
        seed: Random seed for reproducibility

    Returns:
        A random protein sequence with realistic amino acid composition
    """
    if seed is not None:
        random.seed(seed)

    return "".join(random.choices(AA_LIST, weights=AA_WEIGHTS, k=length))


def create_sequence_variant(
    sequence: str, identity: float, seed: Optional[int] = None, realistic: bool = True
) -> str:
    """
    Create a variant of a sequence with a specified sequence identity.

    Parameters:
        sequence: Original sequence
        identity: Target sequence identity (0.0-1.0)
        seed: Random seed for reproducibility
        realistic: Use realistic amino acid frequencies for mutations

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
        current_aa = seq_list[pos]

        if realistic:
            # Get all amino acids except the current one
            aa_options = list(AA_FREQUENCIES.keys())
            aa_options.remove(current_aa)

            # Choose a random amino acid with realistic frequencies
            # We need to recalculate weights after removing the current amino acid
            aa_weights = [AA_FREQUENCIES[aa] for aa in aa_options]
            # Normalize weights
            sum_weights = sum(aa_weights)
            aa_weights = [w / sum_weights for w in aa_weights]

            seq_list[pos] = random.choices(aa_options, weights=aa_weights, k=1)[0]
        else:
            # Use uniform distribution over all other amino acids
            alt_amino_acids = AMINO_ACIDS.replace(current_aa, "")
            seq_list[pos] = random.choice(alt_amino_acids)

    return "".join(seq_list)


def create_cluster_dataset(
    n_clusters: int = 5,
    seqs_per_cluster: int = 10,
    seq_length: int = 100,
    within_identity: float = 0.9,
    between_identity: float = 0.3,
    seed: Optional[int] = None,
    realistic: bool = True,
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
        realistic: Use realistic amino acid frequencies

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
        if realistic:
            cluster_seed = generate_realistic_sequence(
                seq_length, seed=None if seed is None else seed + cluster_idx
            )
        else:
            cluster_seed = generate_random_sequence(
                seq_length, seed=None if seed is None else seed + cluster_idx
            )

        # Create variants within the cluster
        for seq_idx in range(seqs_per_cluster):
            seq_id = f"cluster{cluster_idx + 1}_seq{seq_idx + 1}"

            # First sequence in cluster is the prototype (seed sequence)
            if seq_idx == 0:
                sequence = cluster_seed
                true_identity = 1.0
            else:
                # Create variant with high similarity to cluster seed
                # Add slight variation to identity to make it more realistic
                variation = random.uniform(-0.05, 0.0) if realistic else 0.0
                identity = max(0.7, within_identity + variation)

                sequence = create_sequence_variant(
                    cluster_seed,
                    identity,
                    seed=None if seed is None else seed + cluster_idx * 100 + seq_idx,
                    realistic=realistic,
                )
                true_identity = identity

            # Add some metadata for testing
            molecular_weight = (
                len(sequence) * 110 + random.normalvariate(0, 500)
                if realistic
                else len(sequence) * 100
            )
            isoelectric_point = (
                random.normalvariate(7.0, 1.5) if realistic else cluster_idx + 5
            )
            hydrophobicity = (
                random.normalvariate(cluster_idx - 2, 0.5) if realistic else cluster_idx
            )

            # Add entry
            data.append(
                {
                    "id": seq_id,
                    "sequence": sequence,
                    "cluster_id": cluster_idx + 1,
                    "seq_length": len(sequence),
                    "true_identity": true_identity,
                    "molecular_weight": molecular_weight,
                    "isoelectric_point": isoelectric_point,
                    "hydrophobicity": hydrophobicity,
                    "property_value": np.random.normal(
                        cluster_idx, 0.5
                    ),  # Cluster-correlated property
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(data)

    return df


def create_protein_family_dataset(
    n_families: int = 5,
    proteins_per_family: int = 10,
    avg_seq_length: int = 350,
    length_stddev: int = 50,
    within_identity: float = 0.9,
    between_identity: float = 0.3,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Create a dataset with realistic protein families.

    Parameters:
        n_families: Number of protein families to create
        proteins_per_family: Number of proteins in each family
        avg_seq_length: Average protein length in amino acids
        length_stddev: Standard deviation of protein length
        within_identity: Target identity within families (0.0-1.0)
        between_identity: Target identity between families (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with protein sequence data organized in protein families
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
            f"between_identity ({between_identity}) for clear family separation"
        )

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Initialize data storage
    data = []

    # Create each family
    for family_idx in range(n_families):
        # Determine seed sequence length for this family (varies by family)
        family_seed_length = int(np.random.normal(avg_seq_length, length_stddev / 2))
        family_seed_length = max(50, family_seed_length)  # Ensure minimum length

        # Create a seed sequence for this family
        family_seed = generate_realistic_sequence(
            family_seed_length, seed=None if seed is None else seed + family_idx
        )

        # Create variants within the family
        for protein_idx in range(proteins_per_family):
            protein_id = f"fam{family_idx + 1}_prot{protein_idx + 1}"

            # First protein in family is the prototype (seed sequence)
            if protein_idx == 0:
                sequence = family_seed
                true_identity = 1.0
            else:
                # Create variant with high similarity to family seed
                # Add slight variation to identity to make it more realistic
                variation = random.uniform(
                    -0.05, 0.0
                )  # Slightly reduce identity for variants
                identity = max(0.7, within_identity + variation)

                sequence = create_sequence_variant(
                    family_seed,
                    identity,
                    seed=None
                    if seed is None
                    else seed + family_idx * 100 + protein_idx,
                    realistic=True,
                )
                true_identity = identity

            # Generate realistic protein properties for testing
            molecular_weight = len(sequence) * 110 + random.normalvariate(
                0, 500
            )  # ~110 Da per residue
            isoelectric_point = random.normalvariate(
                7.0, 1.5
            )  # Most proteins around pH 7
            hydrophobicity = random.normalvariate(
                family_idx - 2, 0.5
            )  # Family-specific property

            # Add some categorical features
            domains = random.choice(
                ["kinase", "protease", "transferase", "transporter", "channel"]
            )
            organism = random.choice(
                ["human", "mouse", "yeast", "e_coli", "arabidopsis"]
            )

            # Add entry to dataset
            data.append(
                {
                    "id": protein_id,
                    "sequence": sequence,
                    "family_id": family_idx + 1,
                    "length": len(sequence),
                    "molecular_weight": molecular_weight,
                    "isoelectric_point": isoelectric_point,
                    "hydrophobicity": hydrophobicity,
                    "true_identity": true_identity,
                    "domains": domains,
                    "organism": organism,
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
    realistic: bool = True,
) -> pd.DataFrame:
    """
    Create a dataset with sequences at specific identity levels to a base sequence.
    Useful for testing clustering at different identity thresholds.

    Parameters:
        base_seq_length: Length of the base sequence
        identity_levels: List of identity levels to create variants for
        variants_per_level: Number of variants to create at each identity level
        seed: Random seed for reproducibility
        realistic: Use realistic amino acid frequencies

    Returns:
        DataFrame with sequences at controlled identity levels
    """
    if seed is not None:
        random.seed(seed)

    # Create base sequence
    if realistic:
        base_sequence = generate_realistic_sequence(base_seq_length, seed=seed)
    else:
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
                realistic=realistic,
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


def create_challenging_dataset() -> pd.DataFrame:
    """
    Create a challenging dataset with edge cases for robustness testing.

    Returns:
        DataFrame with protein sequences with characteristics that could challenge
        clustering and splitting algorithms.
    """
    data = []

    # 1. Very short sequence
    data.append(
        {
            "id": "short_seq",
            "sequence": "MKWVTFISLLFLFSSAY",
            "case_type": "short_sequence",
            "length": 17,
        }
    )

    # 2. Very long sequence (realistic large protein)
    long_seq = generate_realistic_sequence(2500, seed=42)
    data.append(
        {
            "id": "long_seq",
            "sequence": long_seq,
            "case_type": "long_sequence",
            "length": 2500,
        }
    )

    # 3. Highly hydrophobic sequence (membrane protein-like)
    hydrophobic = "".join(random.choices("VILMFWC", k=200))
    data.append(
        {
            "id": "hydrophobic_seq",
            "sequence": hydrophobic,
            "case_type": "hydrophobic",
            "length": 200,
        }
    )

    # 4. Highly charged sequence (disordered region-like)
    charged = "".join(random.choices("DEKR", k=200))
    data.append(
        {
            "id": "charged_seq",
            "sequence": charged,
            "case_type": "charged",
            "length": 200,
        }
    )

    # 5. Repeated motif (like collagen)
    repeat_unit = "GPPGPPGPP"
    repeats = repeat_unit * 25
    data.append(
        {
            "id": "repeat_seq",
            "sequence": repeats,
            "case_type": "repetitive",
            "length": len(repeats),
        }
    )

    # 6. Nearly identical sequences (99% identity)
    human_albumin = "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"
    variant = (
        human_albumin[:594] + "E" + human_albumin[595:]
    )  # 99.8% identity (1 substitution in 609aa)

    data.append(
        {
            "id": "near_ident_1",
            "sequence": human_albumin,
            "case_type": "near_identical",
            "length": len(human_albumin),
        }
    )
    data.append(
        {
            "id": "near_ident_2",
            "sequence": variant,
            "case_type": "near_identical",
            "length": len(variant),
        }
    )

    # 7. Exactly identical sequences (protein duplication)
    data.append(
        {
            "id": "ident_1",
            "sequence": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
            "case_type": "identical",
            "length": 238,
        }
    )
    data.append(
        {
            "id": "ident_2",
            "sequence": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
            "case_type": "identical",
            "length": 238,
        }
    )

    # 8. Multi-domain proteins with shared domains
    domain_a = "MKLQQTDLPGLKAIRDAIEVYVPKDEEVSLAALKKAL"
    domain_b = "SKTVEKTLDSLRSAAADMIAKYKITPAGTEAFDHLRQTLESDAAALSKHEGL"
    domain_c = "TSEQQIAELNQVLESLPKTLEDLRNEVIKTTFTVTE"

    protein1 = (
        domain_a + domain_b
    )  # Shares domain_a with protein3, domain_b with protein2
    protein2 = (
        domain_b + domain_c
    )  # Shares domain_b with protein1, domain_c with protein3
    protein3 = (
        domain_c + domain_a
    )  # Shares domain_c with protein2, domain_a with protein1

    data.append(
        {
            "id": "multidomain_1",
            "sequence": protein1,
            "case_type": "multidomain",
            "length": len(protein1),
        }
    )
    data.append(
        {
            "id": "multidomain_2",
            "sequence": protein2,
            "case_type": "multidomain",
            "length": len(protein2),
        }
    )
    data.append(
        {
            "id": "multidomain_3",
            "sequence": protein3,
            "case_type": "multidomain",
            "length": len(protein3),
        }
    )

    # Convert to DataFrame
    df = pd.DataFrame(data)

    return df


def create_edge_case_dataset() -> pd.DataFrame:
    """
    Create a legacy-compatible dataset with edge cases for testing robustness.

    Returns:
        DataFrame with edge case sequences.
    """
    # This function is maintained for backward compatibility
    # It calls the more comprehensive challenging_dataset function
    # and ensures it has the expected structure

    df = create_challenging_dataset()

    # Keep only the core test cases needed for backward compatibility
    core_cases = ["short_sequence", "long_sequence", "identical", "near_identical"]
    filtered_df = df[df["case_type"].isin(core_cases)].copy()

    # Ensure it has the expected columns for backward compatibility
    if "seq_length" not in filtered_df.columns and "length" in filtered_df.columns:
        filtered_df["seq_length"] = filtered_df["length"]

    return filtered_df
