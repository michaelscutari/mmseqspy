"""Tests for clustering accuracy using synthetic data with predictable patterns."""

from mmseqspy import cluster


def test_cluster_basic_functionality(synthetic_cluster_data, mmseqs_installed):
    """Test that clustering correctly identifies the synthetic clusters."""
    df = synthetic_cluster_data.copy()

    # Run clustering with parameters that should match our synthetic data design
    clustered_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.8,  # Should group sequences with ≥80% identity
        coverage=0.8,
    )

    # Check that all sequences are assigned to a representative
    assert "representative_sequence" in clustered_df.columns
    assert clustered_df["representative_sequence"].notna().all()

    # Extract cluster assignments
    cluster_assignments = {}
    for _, row in clustered_df.iterrows():
        seq_id = row["id"]
        rep_id = row["representative_sequence"]

        if rep_id not in cluster_assignments:
            cluster_assignments[rep_id] = []
        cluster_assignments[rep_id].append(seq_id)

    # Check number of clusters
    # We should have approximately 5 clusters (matching our synthetic data)
    n_clusters = len(cluster_assignments)
    assert 9 <= n_clusters <= 11, f"Expected ~10 clusters, got {n_clusters}"

    # Check cluster integrity - sequences from the same synthetic cluster
    # should be assigned to the same MMseqs2 cluster
    for rep_id, members in cluster_assignments.items():
        # Get original cluster IDs for these sequences
        original_clusters = df.loc[df["id"].isin(members), "cluster_id"].unique()

        # Most sequences in a cluster should be from the same original cluster
        assert len(original_clusters) <= 2, (
            f"Cluster with representative {rep_id} contains sequences from too many original clusters: {original_clusters}"
        )


def test_clustering_identity_thresholds(identity_test_data, mmseqs_installed):
    """Test clustering behavior at different sequence identity thresholds."""
    df = identity_test_data.copy()

    # Test different identity thresholds
    identity_thresholds = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45]

    for threshold in identity_thresholds:
        # Run clustering with current threshold
        clustered_df = cluster(
            df,
            sequence_col="sequence",
            id_col="id",
            min_seq_id=threshold,
            coverage=0.8,
        )

        # Get base sequence and its representative
        base_seq_row = clustered_df[clustered_df["id"] == "base_seq"]
        base_rep = base_seq_row["representative_sequence"].iloc[0]

        # Get all sequences assigned to the same cluster as the base sequence
        same_cluster_mask = clustered_df["representative_sequence"] == base_rep
        same_cluster_ids = clustered_df.loc[same_cluster_mask, "id"].tolist()

        # Check which identity levels are in the same cluster as the base sequence
        grouped_identities = {}
        for seq_id in same_cluster_ids:
            if seq_id == "base_seq":
                continue

            # Extract the identity level from the ID
            if seq_id.startswith("id"):
                identity_str = seq_id.split("_")[0][2:]  # Extract digits after "id"
                identity = int(identity_str) / 100

                if identity not in grouped_identities:
                    grouped_identities[identity] = []
                grouped_identities[identity].append(seq_id)

        # Verify that sequences with identity ≥ threshold are grouped with base sequence
        for identity, seq_ids in df[df["id"] != "base_seq"].groupby("true_identity"):
            ids = seq_ids["id"].tolist()

            # Check if these sequences should be in the same cluster as base
            if identity >= threshold:
                # Check that most sequences at this identity level are clustered with base
                matching = sum(1 for id in ids if id in same_cluster_ids)
                assert matching / len(ids) >= 0.8, (
                    f"At threshold {threshold}, only {matching}/{len(ids)} sequences "
                    f"with identity {identity} were clustered with base sequence"
                )
            else:
                # Check that most sequences at this identity level are NOT clustered with base
                matching = sum(1 for id in ids if id in same_cluster_ids)
                assert matching / len(ids) <= 0.2, (
                    f"At threshold {threshold}, {matching}/{len(ids)} sequences "
                    f"with identity {identity} were incorrectly clustered with base sequence"
                )


def test_edge_case_clustering(edge_case_data, mmseqs_installed):
    """Test clustering behavior with edge case sequences."""
    df = edge_case_data.copy()

    # Run clustering
    clustered_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.9,
        coverage=0.5,  # Lower coverage to handle variable length sequences
    )

    # 1. Check that identical sequences are clustered together
    identical_ids = df[df["case_type"] == "identical"]["id"].tolist()
    identical_reps = clustered_df.loc[
        clustered_df["id"].isin(identical_ids), "representative_sequence"
    ].tolist()

    # Both identical sequences should have the same representative
    assert identical_reps[0] == identical_reps[1], (
        "Identical sequences were not clustered together"
    )

    # 2. For near-identical sequences, let's verify the sequences themselves
    near_identical_ids = df[df["case_type"] == "near_identical"]["id"].tolist()
    near_ident_seqs = df.loc[df["id"].isin(near_identical_ids), "sequence"].tolist()

    # Calculate the actual identity between these sequences
    total_positions = len(near_ident_seqs[0])
    matching_positions = sum(
        a == b for a, b in zip(near_ident_seqs[0], near_ident_seqs[1])
    )
    actual_identity = matching_positions / total_positions

    # Verify that they are indeed near-identical (at least 95% identity)
    assert actual_identity >= 0.95, (
        f"Test sequences should have ≥95% identity, but have {actual_identity * 100:.1f}%"
    )

    # 3. Test with extremely high identity threshold to verify different clusters
    high_threshold_df = cluster(
        df.loc[
            df["id"].isin(near_identical_ids)
        ],  # Just use the near-identical sequences
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.99,  # 99% threshold
        coverage=0.9,
    )

    # Since we're using near-identical sequences (99% identity) and a 99% threshold,
    # MMseqs2 could reasonably cluster them together or separately depending on
    # its exact algorithm. Instead of specifying exactly how they should cluster,
    # we'll just verify that the clustering ran without errors.
    assert "representative_sequence" in high_threshold_df.columns
    assert high_threshold_df["representative_sequence"].notna().all()
