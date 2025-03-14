"""Tests for cluster-aware k-fold cross-validation."""

import pandas as pd
from mmseqspy import cluster_kfold


def test_kfold_different_k_values(realistic_protein_data, mmseqs_installed):
    """Test k-fold cross-validation with different k values."""
    df = realistic_protein_data.copy()

    # Test with k=3 and k=5
    for k in [3, 5]:
        folds = cluster_kfold(
            df,
            sequence_col="sequence",
            id_col="id",
            n_splits=k,
            min_seq_id=0.8,
            random_state=42,
        )

        # Check we got the right number of folds
        assert len(folds) == k

        # Track IDs seen in test sets to ensure no overlap
        all_test_ids = set()
        all_ids = set(df["id"])

        for i, (train_df, test_df) in enumerate(folds):
            # Check fold sizes
            assert len(train_df) + len(test_df) == len(df)

            # Check no overlap between train and test
            assert len(set(train_df["id"]) & set(test_df["id"])) == 0

            # Check each ID appears in exactly one test fold
            current_test_ids = set(test_df["id"])
            assert len(current_test_ids & all_test_ids) == 0, (
                f"Test sets overlap in fold {i} with k={k}"
            )

            # Track seen test IDs
            all_test_ids.update(current_test_ids)

        # Every ID should appear in exactly one test fold
        assert all_test_ids == all_ids, (
            f"Not all IDs appeared in test sets with k={k}. "
            f"Missing: {all_ids - all_test_ids}, extra: {all_test_ids - all_ids}"
        )


def test_kfold_cluster_integrity(realistic_protein_data, mmseqs_installed):
    """Test that k-fold cross-validation preserves cluster integrity."""
    df = realistic_protein_data.copy()

    # Run k-fold with k=4
    k = 4
    folds = cluster_kfold(
        df,
        sequence_col="sequence",
        id_col="id",
        n_splits=k,
        min_seq_id=0.8,
        random_state=42,
    )

    for i, (train_df, test_df) in enumerate(folds):
        # For each fold, verify cluster integrity
        # Sequences from the same cluster should never appear in both train and test
        clusters_in_train = set(train_df["representative_sequence"])
        clusters_in_test = set(test_df["representative_sequence"])

        assert len(clusters_in_train & clusters_in_test) == 0, (
            f"Fold {i} has clusters that span both train and test sets"
        )


def test_kfold_property_distribution(realistic_protein_data, mmseqs_installed):
    """Test property distribution balance across k-fold splits."""
    df = realistic_protein_data.copy()

    # Run k-fold with k=3
    k = 3
    folds = cluster_kfold(
        df,
        sequence_col="sequence",
        id_col="id",
        n_splits=k,
        min_seq_id=0.8,
        random_state=42,
    )

    # Calculate overall statistics
    overall_mean = df["molecular_weight"].mean()
    overall_std = df["molecular_weight"].std()

    # Check distribution in each fold
    for i, (train_df, test_df) in enumerate(folds):
        # Calculate mean for train and test sets
        train_mean = train_df["molecular_weight"].mean()
        test_mean = test_df["molecular_weight"].mean()

        # Calculate deviations from overall mean in standard deviation units
        train_dev = abs(train_mean - overall_mean) / overall_std
        test_dev = abs(test_mean - overall_mean) / overall_std

        # Neither set should deviate too much (over 2 std devs would be unusual)
        assert train_dev < 1.5, (
            f"Fold {i} train set deviates too much: {train_dev:.2f} std devs"
        )
        assert test_dev < 1.5, (
            f"Fold {i} test set deviates too much: {test_dev:.2f} std devs"
        )


def test_kfold_reproducibility(realistic_protein_data, mmseqs_installed):
    """Test reproducibility of k-fold splits with the same random seed."""
    df = realistic_protein_data.copy()

    # Run k-fold twice with the same seed
    folds1 = cluster_kfold(
        df,
        sequence_col="sequence",
        id_col="id",
        n_splits=3,
        min_seq_id=0.8,
        random_state=42,
    )

    folds2 = cluster_kfold(
        df,
        sequence_col="sequence",
        id_col="id",
        n_splits=3,
        min_seq_id=0.8,
        random_state=42,
    )

    # Run once with a different seed
    folds3 = cluster_kfold(
        df,
        sequence_col="sequence",
        id_col="id",
        n_splits=3,
        min_seq_id=0.8,
        random_state=24,
    )

    # Check that folds1 and folds2 are identical
    for i in range(3):
        train_ids1 = set(folds1[i][0]["id"])
        train_ids2 = set(folds2[i][0]["id"])

        test_ids1 = set(folds1[i][1]["id"])
        test_ids2 = set(folds2[i][1]["id"])

        assert train_ids1 == train_ids2, f"Train sets differ in fold {i}"
        assert test_ids1 == test_ids2, f"Test sets differ in fold {i}"

    # Check that folds1 and folds3 are different
    different_folds = 0
    for i in range(3):
        train_ids1 = set(folds1[i][0]["id"])
        train_ids3 = set(folds3[i][0]["id"])

        if train_ids1 != train_ids3:
            different_folds += 1

    # At least one fold should be different with a different seed
    assert different_folds > 0, "Different seeds produced identical folds"


def test_kfold_return_indices(realistic_protein_data, mmseqs_installed):
    """Test cluster_kfold with indices return option."""
    df = realistic_protein_data.copy()

    # Run with return_indices=True
    k = 3
    index_folds = cluster_kfold(
        df,
        sequence_col="sequence",
        id_col="id",
        n_splits=k,
        min_seq_id=0.8,
        random_state=42,
        return_indices=True,
    )

    # Check return type
    assert isinstance(index_folds[0][0], pd.Index)
    assert isinstance(index_folds[0][1], pd.Index)

    # Verify indices can be used to access the dataframe
    for i, (train_idx, test_idx) in enumerate(index_folds):
        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        # All data accounted for
        assert len(train_df) + len(test_df) == len(df)

        # No overlap
        assert set(train_df.index).isdisjoint(set(test_df.index))
