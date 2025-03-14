"""Tests for split integrity using synthetic data with predictable patterns."""

from mmseqspy import cluster, split, train_test_cluster_split


def test_basic_split_integrity(synthetic_cluster_data, mmseqs_installed):
    """Test that splits maintain cluster integrity."""
    df = synthetic_cluster_data.copy()

    # First, cluster the data
    clustered_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.8,
        coverage=0.8,
    )

    # Perform splitting with different test sizes
    test_sizes = [0.2, 0.3, 0.4]

    for test_size in test_sizes:
        # Split the data
        train_df, test_df = split(
            clustered_df,
            group_col="representative_sequence",
            test_size=test_size,
            random_state=42,
        )

        # Check that all rows are accounted for
        assert len(train_df) + len(test_df) == len(clustered_df)

        # Check that clusters are kept intact (no cluster spans both train and test)
        train_clusters = set(train_df["representative_sequence"])
        test_clusters = set(test_df["representative_sequence"])

        assert len(train_clusters.intersection(test_clusters)) == 0, (
            f"Split with test_size={test_size} has clusters that span both train and test sets"
        )

        # Check that the test size is reasonably close to target
        actual_test_size = len(test_df) / len(clustered_df)
        assert abs(actual_test_size - test_size) <= 0.1, (
            f"Target test_size={test_size}, but got {actual_test_size:.2f}"
        )


def test_split_reproducibility(synthetic_cluster_data, mmseqs_installed):
    """Test that splits are reproducible with the same random seed."""
    df = synthetic_cluster_data.copy()

    # First, cluster the data
    clustered_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.8,
        coverage=0.8,
    )

    # Perform the split twice with the same seed
    train1, test1 = split(
        clustered_df,
        group_col="representative_sequence",
        test_size=0.3,
        random_state=42,
    )

    train2, test2 = split(
        clustered_df,
        group_col="representative_sequence",
        test_size=0.3,
        random_state=42,
    )

    # Check that the splits are identical
    assert set(train1["id"]) == set(train2["id"]), (
        "Train sets differ despite same random seed"
    )
    assert set(test1["id"]) == set(test2["id"]), (
        "Test sets differ despite same random seed"
    )

    # Perform another split with a different seed
    train3, test3 = split(
        clustered_df,
        group_col="representative_sequence",
        test_size=0.3,
        random_state=24,
    )

    # Check that the splits are different
    assert set(train1["id"]) != set(train3["id"]), (
        "Train sets are identical despite different seeds"
    )
    assert set(test1["id"]) != set(test3["id"]), (
        "Test sets are identical despite different seeds"
    )


def test_combined_cluster_split(synthetic_cluster_data, mmseqs_installed):
    """Test the combined cluster and split functionality."""
    df = synthetic_cluster_data.copy()

    # Use combined function
    train_df, test_df = train_test_cluster_split(
        df,
        sequence_col="sequence",
        id_col="id",
        test_size=0.3,
        min_seq_id=0.8,
        coverage=0.8,
        random_state=42,
    )

    # Check that all rows are accounted for
    assert len(train_df) + len(test_df) == len(df)

    # Check that clusters are present and intact
    assert "representative_sequence" in train_df.columns
    assert "representative_sequence" in test_df.columns

    train_clusters = set(train_df["representative_sequence"])
    test_clusters = set(test_df["representative_sequence"])

    assert len(train_clusters.intersection(test_clusters)) == 0, (
        "Combined function produced splits with clusters spanning both train and test sets"
    )

    # Verify reproducibility by running again
    train2_df, test2_df = train_test_cluster_split(
        df,
        sequence_col="sequence",
        id_col="id",
        test_size=0.3,
        min_seq_id=0.8,
        coverage=0.8,
        random_state=42,
    )

    # Check that results are the same
    assert set(train_df["id"]) == set(train2_df["id"]), (
        "Combined function not reproducible with same random seed"
    )
    assert set(test_df["id"]) == set(test2_df["id"]), (
        "Combined function not reproducible with same random seed"
    )


def test_split_with_property_distribution(synthetic_cluster_data, mmseqs_installed):
    """Test that property distributions are maintained in splits."""
    df = synthetic_cluster_data.copy()

    # First, cluster the data
    clustered_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.8,
        coverage=0.8,
    )

    # Split the data
    train_df, test_df = split(
        clustered_df,
        group_col="representative_sequence",
        test_size=0.3,
        random_state=42,
    )

    # Check property value distributions
    # Since our synthetic data correlates property_value with cluster_id,
    # we can check if the property distribution is roughly maintained

    # Get overall stats
    overall_mean = df["property_value"].mean()
    overall_std = df["property_value"].std()

    # Get split stats
    train_mean = train_df["property_value"].mean()
    test_mean = test_df["property_value"].mean()

    # Check that means are reasonably close
    # Allow for some deviation due to cluster-based splitting
    assert abs(train_mean - overall_mean) < overall_std, (
        f"Train set property mean ({train_mean:.2f}) differs too much from overall ({overall_mean:.2f})"
    )
    assert abs(test_mean - overall_mean) < overall_std, (
        f"Test set property mean ({test_mean:.2f}) differs too much from overall ({overall_mean:.2f})"
    )
