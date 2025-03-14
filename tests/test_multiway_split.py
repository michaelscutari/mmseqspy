"""Tests for multi-way splitting functionality using realistic protein data."""

import pytest
from mmseqspy import train_test_val_cluster_split


def test_train_val_test_integrity(realistic_protein_data, mmseqs_installed):
    """Test that train/val/test splits maintain cluster integrity."""
    df = realistic_protein_data.copy()

    # Different split size combinations to test
    split_configs = [
        {"test_size": 0.2, "val_size": 0.1},  # Standard split
        {"test_size": 0.3, "val_size": 0.15},  # Larger test and validation
        {"test_size": 0.1, "val_size": 0.05},  # Small test and validation
    ]

    for config in split_configs:
        test_size = config["test_size"]
        val_size = config["val_size"]

        # Perform three-way split
        train_df, val_df, test_df = train_test_val_cluster_split(
            df,
            sequence_col="sequence",
            id_col="id",
            test_size=test_size,
            val_size=val_size,
            min_seq_id=0.8,
            coverage=0.8,
            random_state=42,
        )

        # Check that all sequences are accounted for
        assert len(train_df) + len(val_df) + len(test_df) == len(df)

        # Check that clusters are kept intact (no cluster spans multiple sets)
        train_clusters = set(train_df["representative_sequence"])
        val_clusters = set(val_df["representative_sequence"])
        test_clusters = set(test_df["representative_sequence"])

        assert len(train_clusters.intersection(val_clusters)) == 0, (
            "Train and validation sets share clusters"
        )
        assert len(train_clusters.intersection(test_clusters)) == 0, (
            "Train and test sets share clusters"
        )
        assert len(val_clusters.intersection(test_clusters)) == 0, (
            "Validation and test sets share clusters"
        )

        # Check that the split sizes are reasonably close to targets
        total = len(df)
        actual_train_frac = len(train_df) / total
        actual_val_frac = len(val_df) / total
        actual_test_frac = len(test_df) / total

        target_train_frac = 1.0 - (test_size + val_size)

        # Allow more tolerance due to cluster constraints
        tolerance = 0.15

        assert abs(actual_train_frac - target_train_frac) <= tolerance, (
            f"Train set size ({actual_train_frac:.2f}) too far from target ({target_train_frac:.2f})"
        )
        assert abs(actual_val_frac - val_size) <= tolerance, (
            f"Validation set size ({actual_val_frac:.2f}) too far from target ({val_size:.2f})"
        )
        assert abs(actual_test_frac - test_size) <= tolerance, (
            f"Test set size ({actual_test_frac:.2f}) too far from target ({test_size:.2f})"
        )


def test_multiway_split_property_distribution(realistic_protein_data, mmseqs_installed):
    """Test that property distributions are reasonably preserved in multi-way splits."""
    df = realistic_protein_data.copy()

    # Run the split
    train_df, val_df, test_df = train_test_val_cluster_split(
        df,
        sequence_col="sequence",
        id_col="id",
        test_size=0.2,
        val_size=0.1,
        min_seq_id=0.8,
        coverage=0.8,
        random_state=42,
    )

    # Check numerical property distributions
    for prop in ["molecular_weight", "isoelectric_point", "hydrophobicity"]:
        # Calculate overall statistics
        overall_mean = df[prop].mean()
        overall_std = df[prop].std()

        # Calculate split statistics
        train_mean = train_df[prop].mean()
        val_mean = val_df[prop].mean()
        test_mean = test_df[prop].mean()

        # Verify that means don't deviate too much (within 1.5 std deviations)
        # This is a reasonable expectation for random splits with sufficient data
        assert abs(train_mean - overall_mean) <= 1.5 * overall_std, (
            f"{prop} train mean ({train_mean:.2f}) deviates too much from overall ({overall_mean:.2f})"
        )
        assert abs(val_mean - overall_mean) <= 1.5 * overall_std, (
            f"{prop} validation mean ({val_mean:.2f}) deviates too much from overall ({overall_mean:.2f})"
        )
        assert abs(test_mean - overall_mean) <= 1.5 * overall_std, (
            f"{prop} test mean ({test_mean:.2f}) deviates too much from overall ({overall_mean:.2f})"
        )

    # Check categorical property distributions
    for cat_prop in ["domains", "organism"]:
        # Calculate overall category frequencies
        overall_freqs = df[cat_prop].value_counts(normalize=True)

        # Calculate split frequencies
        train_freqs = train_df[cat_prop].value_counts(normalize=True)
        val_freqs = val_df[cat_prop].value_counts(normalize=True)
        test_freqs = test_df[cat_prop].value_counts(normalize=True)

        # Ensure all categories are represented
        for category in overall_freqs.index:
            # Allow some categories to be missing in smaller splits (val/test)
            if overall_freqs[category] > 0.1:  # Only check for common categories
                assert category in train_freqs.index, (
                    f"Common category {category} missing from train set"
                )

                # For validation and test, only assert if they should have enough samples
                # based on the split ratio and original count
                expected_val_count = len(val_df) * overall_freqs[category]
                expected_test_count = len(test_df) * overall_freqs[category]

                if expected_val_count >= 2:  # At least 2 expected samples
                    assert category in val_freqs.index, (
                        f"Common category {category} missing from validation set"
                    )

                if expected_test_count >= 2:  # At least 2 expected samples
                    assert category in test_freqs.index, (
                        f"Common category {category} missing from test set"
                    )


def test_multiway_split_invalid_params(realistic_protein_data, mmseqs_installed):
    """Test error handling for invalid parameters in multi-way splits."""
    df = realistic_protein_data.copy()

    # Case 1: test_size + val_size > 1.0
    with pytest.raises(ValueError) as excinfo:
        train_test_val_cluster_split(
            df,
            sequence_col="sequence",
            id_col="id",
            test_size=0.6,
            val_size=0.5,  # Together > 1.0
            min_seq_id=0.8,
        )
    assert "less than 1.0" in str(excinfo.value).lower()

    # Case 2: Negative test_size
    with pytest.raises(ValueError) as excinfo:
        train_test_val_cluster_split(
            df,
            sequence_col="sequence",
            id_col="id",
            test_size=-0.2,
            val_size=0.1,
        )
    assert "must be non-negative" in str(excinfo.value).lower()

    # Case 3: Negative val_size
    with pytest.raises(ValueError) as excinfo:
        train_test_val_cluster_split(
            df,
            sequence_col="sequence",
            id_col="id",
            test_size=0.2,
            val_size=-0.1,
        )
    assert "must be non-negative" in str(excinfo.value).lower()

    # Case 4: Missing required columns
    with pytest.raises(ValueError) as excinfo:
        df_missing_seq = df.drop(columns=["sequence"])
        train_test_val_cluster_split(
            df_missing_seq,
            sequence_col="sequence",
            id_col="id",
            test_size=0.2,
            val_size=0.1,
        )
    assert (
        "column" in str(excinfo.value).lower()
        or "sequence" in str(excinfo.value).lower()
    )


def test_multiway_split_reproducibility(realistic_protein_data, mmseqs_installed):
    """Test that multi-way splits are reproducible with the same random seed."""
    df = realistic_protein_data.copy()

    # Perform the split twice with the same seed
    train1, val1, test1 = train_test_val_cluster_split(
        df,
        sequence_col="sequence",
        id_col="id",
        test_size=0.2,
        val_size=0.1,
        min_seq_id=0.8,
        coverage=0.8,
        random_state=42,
    )

    train2, val2, test2 = train_test_val_cluster_split(
        df,
        sequence_col="sequence",
        id_col="id",
        test_size=0.2,
        val_size=0.1,
        min_seq_id=0.8,
        coverage=0.8,
        random_state=42,
    )

    # Check that the splits are identical
    assert sorted(train1["id"].tolist()) == sorted(train2["id"].tolist()), (
        "Train sets differ despite same random seed"
    )
    assert sorted(val1["id"].tolist()) == sorted(val2["id"].tolist()), (
        "Validation sets differ despite same random seed"
    )
    assert sorted(test1["id"].tolist()) == sorted(test2["id"].tolist()), (
        "Test sets differ despite same random seed"
    )

    # Perform another split with a different seed
    train3, val3, test3 = train_test_val_cluster_split(
        df,
        sequence_col="sequence",
        id_col="id",
        test_size=0.2,
        val_size=0.1,
        min_seq_id=0.8,
        coverage=0.8,
        random_state=24,
    )

    # Check that the splits are different
    assert sorted(train1["id"].tolist()) != sorted(train3["id"].tolist()), (
        "Train sets identical despite different seeds"
    )
    assert sorted(val1["id"].tolist()) != sorted(val3["id"].tolist()), (
        "Validation sets identical despite different seeds"
    )
