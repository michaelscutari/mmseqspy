"""Tests for constrained splitting functionality using realistic protein data."""

import pytest
from mmseqspy import cluster, constrained_split


def test_constrained_split_basic(realistic_protein_data, mmseqs_installed):
    """Test basic constrained split functionality with realistic protein data."""
    df = realistic_protein_data.copy()

    # First, cluster the data
    clustered_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.8,
        coverage=0.8,
    )

    # Select some IDs to force into train and test
    train_ids = clustered_df["id"].iloc[0:10].tolist()
    test_ids = clustered_df["id"].iloc[50:60].tolist()

    # Run constrained split
    train_df, test_df = constrained_split(
        clustered_df,
        group_col="representative_sequence",
        id_col="id",
        test_size=0.3,
        force_train_ids=train_ids,
        force_test_ids=test_ids,
        id_type="sequence",
        random_state=42,
    )

    # Check that all sequences are accounted for
    assert len(train_df) + len(test_df) == len(clustered_df)

    # Check that clusters are kept intact
    train_clusters = set(train_df["representative_sequence"])
    test_clusters = set(test_df["representative_sequence"])
    assert len(train_clusters.intersection(test_clusters)) == 0, (
        "Train and test sets share clusters"
    )

    # Check that forced IDs are in the correct sets
    assert all(id_val in train_df["id"].values for id_val in train_ids), (
        "Not all forced train IDs are in the train set"
    )
    assert all(id_val in test_df["id"].values for id_val in test_ids), (
        "Not all forced test IDs are in the test set"
    )


def test_constrained_split_cluster_level(realistic_protein_data, mmseqs_installed):
    """Test constrained split with cluster-level constraints."""
    df = realistic_protein_data.copy()

    # First, cluster the data
    clustered_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.8,
        coverage=0.8,
    )

    # Select some clusters to force into train and test
    unique_clusters = clustered_df["representative_sequence"].unique()
    train_clusters = unique_clusters[:2].tolist()
    test_clusters = unique_clusters[-2:].tolist()

    # Run constrained split at cluster level
    train_df, test_df = constrained_split(
        clustered_df,
        group_col="representative_sequence",
        id_col="id",
        test_size=0.3,
        force_train_ids=train_clusters,
        force_test_ids=test_clusters,
        id_type="cluster",
        random_state=42,
    )

    # Check that all sequences are accounted for
    assert len(train_df) + len(test_df) == len(clustered_df)

    # Check that forced clusters are in the correct sets
    train_result_clusters = set(train_df["representative_sequence"])
    test_result_clusters = set(test_df["representative_sequence"])

    assert all(cluster in train_result_clusters for cluster in train_clusters), (
        "Not all forced train clusters are in the train set"
    )
    assert all(cluster in test_result_clusters for cluster in test_clusters), (
        "Not all forced test clusters are in the test set"
    )


def test_constrained_split_conflicts(realistic_protein_data, mmseqs_installed):
    """Test that constrained split properly handles conflicts."""
    df = realistic_protein_data.copy()

    # First, cluster the data
    clustered_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.8,
        coverage=0.8,
    )

    # Create conflicting constraints (same ID in both train and test)
    conflict_id = clustered_df["id"].iloc[0]

    with pytest.raises(ValueError) as excinfo:
        constrained_split(
            clustered_df,
            group_col="representative_sequence",
            id_col="id",
            test_size=0.3,
            force_train_ids=[conflict_id],
            force_test_ids=[conflict_id],
            id_type="sequence",
        )
    assert "cannot force the same ids" in str(excinfo.value).lower()

    # Create cluster-level conflict by finding two sequences in the same cluster
    # First find a cluster with multiple sequences
    cluster_counts = clustered_df["representative_sequence"].value_counts()
    multi_clusters = cluster_counts[cluster_counts > 1].index
    if len(multi_clusters) > 0:
        rep = multi_clusters[0]
        # Get two sequences from the same cluster
        cluster_members = clustered_df[clustered_df["representative_sequence"] == rep][
            "id"
        ].tolist()

        # Try to force one to train and one to test (should detect conflict)
        with pytest.raises(ValueError) as excinfo:
            constrained_split(
                clustered_df,
                group_col="representative_sequence",
                id_col="id",
                test_size=0.3,
                force_train_ids=[cluster_members[0]],
                force_test_ids=[cluster_members[1]],
                id_type="sequence",
            )
        assert (
            "conflict" in str(excinfo.value).lower()
            or "both train and test" in str(excinfo.value).lower()
        )


def test_constrained_split_property_balance(realistic_protein_data, mmseqs_installed):
    """Test that constrained split preserves property distributions where possible."""
    df = realistic_protein_data.copy()

    # First, cluster the data
    clustered_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.8,
        coverage=0.8,
    )

    # Check properties we want to track
    properties = ["molecular_weight", "isoelectric_point", "hydrophobicity"]

    # Calculate property means for each cluster
    cluster_props = {}
    for cluster_id in clustered_df["representative_sequence"].unique():
        cluster_members = clustered_df[
            clustered_df["representative_sequence"] == cluster_id
        ]
        cluster_props[cluster_id] = {
            prop: cluster_members[prop].mean() for prop in properties
        }

    # Force clusters with low and high values for each property
    # This ensures diversity in the forced train and test sets
    force_train_clusters = []
    force_test_clusters = []

    for prop in properties:
        # Sort clusters by property value
        sorted_clusters = sorted(cluster_props.items(), key=lambda x: x[1][prop])
        # Add low and high extremes to train and test, respectively
        force_train_clusters.append(sorted_clusters[0][0])
        force_test_clusters.append(sorted_clusters[-1][0])

    # Remove duplicates
    force_train_clusters = list(set(force_train_clusters))
    force_test_clusters = list(set(force_test_clusters))

    # Run constrained split
    train_df, test_df = constrained_split(
        clustered_df,
        group_col="representative_sequence",
        id_col="id",
        test_size=0.3,
        force_train_ids=force_train_clusters,
        force_test_ids=force_test_clusters,
        id_type="cluster",
        random_state=42,
    )

    # Check that property ranges in each split include both low and high values
    for prop in properties:
        train_min, train_max = train_df[prop].min(), train_df[prop].max()
        test_min, test_max = test_df[prop].min(), test_df[prop].max()
        overall_min, overall_max = clustered_df[prop].min(), clustered_df[prop].max()

        # Calculate range coverages (how much of the overall range is covered by each split)
        train_coverage = (train_max - train_min) / (overall_max - overall_min)
        test_coverage = (test_max - test_min) / (overall_max - overall_min)

        # Both splits should cover a substantial portion of the range
        assert train_coverage >= 0.5, (
            f"Train set covers only {train_coverage:.2%} of the {prop} range"
        )
        assert test_coverage >= 0.5, (
            f"Test set covers only {test_coverage:.2%} of the {prop} range"
        )


def test_constrained_split_extreme_case(realistic_protein_data, mmseqs_installed):
    """Test constrained split with extreme constraints that force most data into one set."""
    df = realistic_protein_data.copy()

    # First, cluster the data
    clustered_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.8,
        coverage=0.8,
    )

    # Force almost all data into train set
    unique_clusters = clustered_df["representative_sequence"].unique()
    force_train_clusters = unique_clusters[:-1]  # All except one cluster

    train_df, test_df = constrained_split(
        clustered_df,
        group_col="representative_sequence",
        id_col="id",
        test_size=0.3,  # This will be overridden by constraints
        force_train_ids=force_train_clusters.tolist(),
        force_test_ids=[],
        id_type="cluster",
        random_state=42,
    )

    # Calculate actual split percentage
    actual_test_pct = len(test_df) / len(clustered_df)

    # The test percentage should be very low due to constraints
    assert actual_test_pct < 0.15, (
        f"Test percentage ({actual_test_pct:.2f}) is too high given constraints"
    )

    # Verify that majority of clusters are in train set as forced
    train_clusters = set(train_df["representative_sequence"])

    forced_pct = len(set(force_train_clusters) & train_clusters) / len(
        force_train_clusters
    )
    assert forced_pct == 1.0, (
        f"Only {forced_pct:.2%} of forced train clusters are in train set"
    )
