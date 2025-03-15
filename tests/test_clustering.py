from mmseqspy import cluster


def test_cluster_sequences(fluorescence_data, mmseqs_installed):
    """Test clustering protein sequences."""
    # Make a copy of data
    df = fluorescence_data.copy()

    # Run clustering
    clustered_df = cluster(
        df,
        sequence_col="sequence",
        id_col="id",
        min_seq_id=0.5,  # 50% sequence identity threshold
        coverage=0.8,  # 80% coverage
    )

    # Check results
    assert "representative_sequence" in clustered_df.columns
    assert len(clustered_df) == len(df)  # Should preserve all rows

    # Count unique clusters
    n_clusters = clustered_df["representative_sequence"].nunique()

    # Basic sanity check - clusters should be fewer than sequences
    assert 1 <= n_clusters <= len(df)

    # Check that all representative_sequence values exist in the id column
    assert set(clustered_df["representative_sequence"]).issubset(
        set(clustered_df["id"])
    )


def test_cluster_debug_logging(fluorescence_data, mmseqs_installed, monkeypatch):
    """Test clustering with debug logging to cover verbose output paths."""
    from mmseqspy.logger import logger
    import logging

    # Store original level and set up logging capture
    original_level = logger.level
    logger.setLevel(logging.DEBUG)

    try:
        # Run clustering with a small dataset
        df = fluorescence_data.head(10).copy()
        result = cluster(df, sequence_col="sequence", id_col="id")

        # Verify the clustering completed successfully
        assert "representative_sequence" in result.columns

    finally:
        # Restore original logging level
        logger.setLevel(original_level)


def test_cluster_with_debug_output(fluorescence_data, mmseqs_installed, caplog):
    """Test clustering with debug output enabled."""
    import logging
    from mmseqspy.logger import logger
    from mmseqspy import cluster

    # Set debug level temporarily
    original_level = logger.level
    logger.setLevel(logging.DEBUG)

    try:
        # Just a small sample to keep it fast
        df_small = fluorescence_data.head(10).copy()
        result = cluster(df_small, sequence_col="sequence", id_col="id")

        # Verify it ran successfully
        assert "representative_sequence" in result.columns
    finally:
        # Restore original level
        logger.setLevel(original_level)
