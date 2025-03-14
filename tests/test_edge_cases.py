"""Tests for edge cases and challenging protein data scenarios."""

import pytest
import pandas as pd
import random
from mmseqspy import clean, cluster


def test_extreme_sequence_lengths():
    """Test handling of extremely short and long sequences."""
    # Create test data
    data = []

    # Very short sequences (8-10 amino acids)
    for i in range(5):
        data.append(
            {
                "id": f"short_{i}",
                "sequence": "".join(
                    random.choices("ACDEFGHIKLMNPQRSTVWY", k=random.randint(8, 10))
                ),
                "length": random.randint(8, 10),
                "type": "short",
            }
        )

    # Very long sequences (2000-3000 amino acids)
    for i in range(5):
        length = random.randint(2000, 3000)
        data.append(
            {
                "id": f"long_{i}",
                "sequence": "".join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=length)),
                "length": length,
                "type": "long",
            }
        )

    # Normal sequences (200-300 amino acids)
    for i in range(10):
        length = random.randint(200, 300)
        data.append(
            {
                "id": f"normal_{i}",
                "sequence": "".join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=length)),
                "length": length,
                "type": "normal",
            }
        )

    df = pd.DataFrame(data)

    # Test cleaning
    cleaned_df = clean(df, sequence_col="sequence")
    assert len(cleaned_df) == len(df), "Cleaning shouldn't remove valid sequences"

    # Test clustering
    try:
        clustered_df = cluster(
            cleaned_df,
            sequence_col="sequence",
            id_col="id",
            min_seq_id=0.9,
            coverage=0.8,
        )

        # Check all sequences were processed
        assert len(clustered_df) == len(df)
        assert "representative_sequence" in clustered_df.columns

        # Check extreme length sequences weren't all grouped together
        long_reps = clustered_df[clustered_df["type"] == "long"][
            "representative_sequence"
        ].nunique()

        # Short sequences might cluster together due to limited variation possibilities
        # but long sequences should definitely not all cluster
        assert long_reps > 1, "All long sequences incorrectly clustered together"

    except Exception as e:
        pytest.fail(f"Failed to cluster extreme length sequences: {e}")


def test_biased_composition_sequences():
    """Test handling of sequences with biased amino acid composition."""
    # Create test data with biased compositions
    data = []

    # Hydrophobic-biased (VILMFWY)
    for i in range(5):
        length = random.randint(100, 200)
        data.append(
            {
                "id": f"hydrophobic_{i}",
                "sequence": "".join(random.choices("VILMFWY", k=length)),
                "type": "hydrophobic",
            }
        )

    # Charged-biased (DEKR)
    for i in range(5):
        length = random.randint(100, 200)
        data.append(
            {
                "id": f"charged_{i}",
                "sequence": "".join(random.choices("DEKR", k=length)),
                "type": "charged",
            }
        )

    # Balanced sequences
    for i in range(10):
        length = random.randint(100, 200)
        data.append(
            {
                "id": f"balanced_{i}",
                "sequence": "".join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=length)),
                "type": "balanced",
            }
        )

    df = pd.DataFrame(data)

    # Test clustering
    clustered_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.5,
        coverage=0.5,
    )

    # Check that biased sequences were handled
    assert len(clustered_df) == len(df)

    # Similar bias should tend to cluster together, but not all
    hydrophobic_clusters = clustered_df[clustered_df["type"] == "hydrophobic"][
        "representative_sequence"
    ].nunique()
    charged_clusters = clustered_df[clustered_df["type"] == "charged"][
        "representative_sequence"
    ].nunique()

    # Some clustering of similar composition is expected, but not all in one cluster
    assert hydrophobic_clusters >= 2, (
        "All hydrophobic sequences incorrectly clustered together"
    )
    assert charged_clusters >= 2, "All charged sequences incorrectly clustered together"

    # Different bias types should not cluster together
    hydrophobic_reps = set(
        clustered_df[clustered_df["type"] == "hydrophobic"]["representative_sequence"]
    )
    charged_reps = set(
        clustered_df[clustered_df["type"] == "charged"]["representative_sequence"]
    )

    assert len(hydrophobic_reps.intersection(charged_reps)) == 0, (
        "Hydrophobic and charged sequences incorrectly clustered together"
    )


def test_repeating_motif_sequences():
    """Test handling of sequences with repeating motifs."""
    # Create data with different types of repetitive sequences
    data = []

    # Simple repeats
    repeat_motifs = ["A", "KE", "PG", "GPPGPP", "AAAGG"]
    for i, motif in enumerate(repeat_motifs):
        # Repeat the motif to get reasonable length
        repeats = 200 // len(motif)
        sequence = motif * repeats

        data.append(
            {
                "id": f"repeat_{i}",
                "sequence": sequence,
                "motif": motif,
                "type": "repetitive",
            }
        )

    # Normal sequences
    for i in range(10):
        length = random.randint(100, 200)
        data.append(
            {
                "id": f"normal_{i}",
                "sequence": "".join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=length)),
                "motif": "none",
                "type": "normal",
            }
        )

    df = pd.DataFrame(data)

    # Test clustering
    clustered_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.5,
        coverage=0.5,
    )

    # Check that repetitive sequences were handled
    assert len(clustered_df) == len(df)

    # Repetitive sequences shouldn't cluster with normal sequences
    rep_clusters = set(
        clustered_df[clustered_df["type"] == "repetitive"]["representative_sequence"]
    )
    normal_clusters = set(
        clustered_df[clustered_df["type"] == "normal"]["representative_sequence"]
    )

    # There should be very limited overlap if any
    overlap = len(rep_clusters.intersection(normal_clusters))
    assert overlap <= 1, (
        f"Too many repetitive sequences clustered with normal ones: {overlap}"
    )


def test_clustering_identical_sequences():
    """Test clustering behavior with identical sequences."""
    # Create data with identical sequences
    data = []

    # Create 5 identical sequences
    sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTF"
    for i in range(5):
        data.append({"id": f"identical_{i}", "sequence": sequence, "type": "identical"})

    # Create 5 nearly identical sequences (1 mutation)
    for i in range(5):
        # Introduce a single mutation
        seq_list = list(sequence)
        pos = random.randint(0, len(sequence) - 1)
        original_aa = seq_list[pos]
        available_aa = list("ACDEFGHIKLMNPQRSTVWY")
        available_aa.remove(original_aa)
        seq_list[pos] = random.choice(available_aa)

        data.append(
            {
                "id": f"nearly_identical_{i}",
                "sequence": "".join(seq_list),
                "type": "nearly_identical",
            }
        )

    # Add some different sequences
    for i in range(5):
        length = len(sequence)
        data.append(
            {
                "id": f"different_{i}",
                "sequence": "".join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=length)),
                "type": "different",
            }
        )

    df = pd.DataFrame(data)

    # Test with high identity threshold (>0.99)
    high_id_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.99,
        coverage=0.9,
    )

    # Identical sequences must cluster together
    identical_reps = high_id_df[high_id_df["type"] == "identical"][
        "representative_sequence"
    ].nunique()
    assert identical_reps == 1, "Identical sequences didn't cluster together"

    # Nearly identical may or may not cluster with high threshold
    # Different sequences should definitely not cluster with identical ones
    identical_rep = high_id_df[high_id_df["type"] == "identical"][
        "representative_sequence"
    ].iloc[0]
    different_in_identical_cluster = high_id_df[
        (high_id_df["type"] == "different")
        & (high_id_df["representative_sequence"] == identical_rep)
    ]

    assert len(different_in_identical_cluster) == 0, (
        "Different sequences incorrectly clustered with identical ones"
    )

    # Test with lower identity threshold (0.9)
    low_id_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.9,
        coverage=0.9,
    )

    # Nearly identical should now cluster with identical
    identical_rep = low_id_df[low_id_df["type"] == "identical"][
        "representative_sequence"
    ].iloc[0]
    nearly_in_identical_cluster = low_id_df[
        (low_id_df["type"] == "nearly_identical")
        & (low_id_df["representative_sequence"] == identical_rep)
    ]

    assert len(nearly_in_identical_cluster) > 0, (
        "Nearly identical sequences didn't cluster with identical ones at 90% threshold"
    )


def test_multidomain_proteins():
    """Test handling of multidomain proteins."""
    # Create data with multi-domain sequences
    data = []

    # Define synthetic domains
    domains = {
        "A": "MKLQQTDLPGLKAIRDAIEVYVPKDEEVSLAALKKAL",
        "B": "SKTVEKTLDSLRSAAADMIAKYKITPAGTEAFDHLRQTLESDAAALSKHEGL",
        "C": "TSEQQIAELNQVLESLPKTLEDLRNEVIKTTFTVTE",
        "D": "KNHVVIITGSNVGGLKGNPLGYFLRRAIGHEVEVYTALSP",
    }

    # Create proteins with different domain architectures
    architectures = [
        {"id": "AB_1", "domains": ["A", "B"]},
        {"id": "AB_2", "domains": ["A", "B"]},
        {"id": "BC_1", "domains": ["B", "C"]},
        {"id": "BC_2", "domains": ["B", "C"]},
        {"id": "CA_1", "domains": ["C", "A"]},
        {"id": "CA_2", "domains": ["C", "A"]},
        {"id": "ABC_1", "domains": ["A", "B", "C"]},
        {"id": "ABC_2", "domains": ["A", "B", "C"]},
        {"id": "BD_1", "domains": ["B", "D"]},
        {"id": "BD_2", "domains": ["B", "D"]},
        {"id": "AD_1", "domains": ["A", "D"]},
        {"id": "AD_2", "domains": ["A", "D"]},
    ]

    for arch in architectures:
        sequence = "".join(domains[d] for d in arch["domains"])
        data.append(
            {
                "id": arch["id"],
                "sequence": sequence,
                "architecture": "_".join(arch["domains"]),
                "type": "multidomain",
            }
        )

    df = pd.DataFrame(data)

    # Test with different clustering parameters
    # High identity (0.9) and coverage (0.9)
    high_params_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.9,
        coverage=0.9,
    )

    # Only identical architectures should cluster together
    high_param_groups = {}
    for _, row in high_params_df.iterrows():
        rep = row["representative_sequence"]
        arch = row["architecture"]
        if rep not in high_param_groups:
            high_param_groups[rep] = []
        high_param_groups[rep].append(arch)

    # Check that each cluster has only one architecture
    for rep, architectures in high_param_groups.items():
        assert len(set(architectures)) == 1, (
            f"Different architectures clustered together with high parameters: {architectures}"
        )

    # Lower identity (0.7) and coverage (0.7)
    # This should allow some shared domains to cluster
    low_params_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.7,
        coverage=0.7,
    )

    # Check that at least some different architectures with shared domains cluster together
    low_param_clusters = low_params_df.groupby("representative_sequence")[
        "architecture"
    ].unique()

    # At least one cluster should have multiple architectures
    multiple_arch_clusters = [archs for archs in low_param_clusters if len(archs) > 1]
    assert len(multiple_arch_clusters) > 0, (
        "No shared domain architectures clustered together with lower parameters"
    )

    # Verify that clustered architectures share domains
    for architectures in multiple_arch_clusters:
        domain_sets = [set(arch.split("_")) for arch in architectures]
        # Check for intersection - should have at least one domain in common
        common_domains = set.intersection(*domain_sets)
        assert len(common_domains) > 0, (
            f"Architectures clustered without common domains: {architectures}"
        )
