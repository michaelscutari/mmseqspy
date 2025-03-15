"""Tests for performance and scaling of MMseqsPy functions."""

import pytest
import pandas as pd
import time
import random
from mmseqspy import cluster, split, train_test_cluster_split
from .test_utils import create_protein_family_dataset


def test_clustering_scaling():
    """Test scaling of clustering with increasing dataset sizes."""
    sizes = [20, 40, 80]  # Sizes to test
    runtimes = []

    for size in sizes:
        # Generate dataset of appropriate size
        df = create_protein_family_dataset(
            n_families=size // 8,
            proteins_per_family=8,
            avg_seq_length=200,  # Medium-length sequences for faster tests
            seed=42,
        )

        # Time the clustering operation
        start_time = time.time()
        clustered_df = cluster(
            df, sequence_col="sequence", id_col="id", min_seq_id=0.8, coverage=0.8
        )
        elapsed = time.time() - start_time
        runtimes.append(elapsed)

        # Verify basic functionality
        assert len(clustered_df) == len(df)
        assert "representative_sequence" in clustered_df.columns

    # Calculate scaling factors
    scaling_20_to_40 = runtimes[1] / runtimes[0]
    scaling_40_to_80 = runtimes[2] / runtimes[1]

    # Expect roughly linear to quadratic scaling
    # Multiplying size by 2 should increase runtime by 2-4x
    assert scaling_20_to_40 < 8, (
        f"Clustering scales poorly: {scaling_20_to_40:.2f}x runtime for 2x data"
    )
    assert scaling_40_to_80 < 8, (
        f"Clustering scales poorly: {scaling_40_to_80:.2f}x runtime for 2x data"
    )


def test_splitting_performance():
    """Test performance of various splitting methods."""
    # Generate a larger dataset
    df = create_protein_family_dataset(
        n_families=25, proteins_per_family=8, avg_seq_length=200, seed=42
    )

    # Cluster the data first (common preprocessing step)
    clustered_df = cluster(df, sequence_col="sequence", id_col="id", min_seq_id=0.8)

    # Time basic split
    start_time = time.time()
    train_df, test_df = split(
        clustered_df,
        group_col="representative_sequence",
        test_size=0.3,
        random_state=42,
    )
    basic_split_time = time.time() - start_time

    # Verify the split
    assert len(train_df) + len(test_df) == len(clustered_df)

    # Skip importing and timing MILP split if it fails due to missing dependencies
    try:
        from mmseqspy import milp_split

        # Time MILP split with minimal properties
        start_time = time.time()
        milp_train_df, milp_test_df = milp_split(
            clustered_df,
            group_col="representative_sequence",
            test_size=0.3,
            balance_cols=["molecular_weight"],
            time_limit=10,
            random_state=42,
        )
        milp_split_time = time.time() - start_time

        # Verify the split
        assert len(milp_train_df) + len(milp_test_df) == len(clustered_df)

        # MILP should be slower but not prohibitively so
        assert milp_split_time > basic_split_time, "MILP split should be slower"

        # But it shouldn't be unreasonably slow
        # Use a minimum comparison time to handle cases where basic split is very fast
        minimum_compare_time = max(
            basic_split_time, 0.05
        )  # At least 50ms for baseline comparison
        assert milp_split_time < minimum_compare_time * 40, (
            f"MILP split too slow: {milp_split_time:.2f}s vs {basic_split_time:.2f}s"
        )

    except ImportError:
        # Skip test if PuLP is not available
        pass


def test_combined_function_efficiency():
    """Test efficiency of combined train_test_cluster_split function."""
    # Generate dataset
    df = create_protein_family_dataset(
        n_families=15, proteins_per_family=8, avg_seq_length=200, seed=42
    )

    # Time separate cluster + split
    start_time = time.time()
    clustered_df = cluster(df, sequence_col="sequence", id_col="id", min_seq_id=0.8)
    train1_df, test1_df = split(
        clustered_df,
        group_col="representative_sequence",
        test_size=0.3,
        random_state=42,
    )
    separate_time = time.time() - start_time

    # Time combined function
    start_time = time.time()
    train2_df, test2_df = train_test_cluster_split(
        df,
        sequence_col="sequence",
        id_col="id",
        test_size=0.3,
        min_seq_id=0.8,
        random_state=42,
    )
    combined_time = time.time() - start_time

    # Verify functionality
    assert len(train1_df) + len(test1_df) == len(df)
    assert len(train2_df) + len(test2_df) == len(df)

    # Combined function should be at least as efficient
    # Allow for some variation (within 20%)
    assert combined_time <= separate_time * 1.2, (
        f"Combined function less efficient: {combined_time:.2f}s vs {separate_time:.2f}s"
    )


def test_memory_usage_with_large_sequences():
    """Test memory efficiency with large protein sequences."""
    import sys

    # Create data with some very large sequences
    data = []

    # Add 5 huge proteins (50,000 aa)
    for i in range(5):
        huge_seq = "".join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=50000))
        data.append({"id": f"huge_{i}", "sequence": huge_seq})

    # Add 20 normal proteins
    for i in range(20):
        normal_seq = "".join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=500))
        data.append({"id": f"normal_{i}", "sequence": normal_seq})

    df = pd.DataFrame(data)

    # Monitor approximate memory usage
    before_size = 0
    try:
        # Track DataFrame size before
        before_size = sys.getsizeof(df) + sum(sys.getsizeof(obj) for obj in df.values)

        # Process data
        clustered_df = cluster(
            df,
            sequence_col="sequence",
            id_col="id",
            min_seq_id=0.8,
            coverage=0.5,  # Lower coverage for better handling of variable lengths
        )

        # Track size after
        after_size = sys.getsizeof(clustered_df) + sum(
            sys.getsizeof(obj) for obj in clustered_df.values
        )

        # Memory shouldn't explode (shouldn't grow by more than 10x)
        assert after_size < before_size * 10, (
            f"Memory usage grew too much: {after_size} bytes vs {before_size} bytes"
        )

        # Basic functionality check
        assert len(clustered_df) == len(df)
        assert "representative_sequence" in clustered_df.columns

    except MemoryError:
        pytest.skip("Not enough memory for large sequence test")
