"""Tests for MILP-based splitting functionality with detailed property balance verification."""

import numpy as np
from mmseqspy import cluster, milp_split


def test_milp_numeric_properties(realistic_protein_data, mmseqs_installed):
    """Test MILP balancing of multiple numeric properties."""
    df = realistic_protein_data.copy()

    # Cluster the data
    clustered_df = cluster(df, sequence_col="sequence", id_col="id", min_seq_id=0.8)

    # Define numeric properties to balance
    num_props = ["molecular_weight", "hydrophobicity", "isoelectric_point"]

    # Run MILP split balancing multiple numeric properties
    train_df, test_df = milp_split(
        clustered_df,
        group_col="representative_sequence",
        test_size=0.3,
        balance_cols=num_props,
        balance_weight=1.0,
        time_limit=15,  # Short time limit to keep tests fast
        random_state=42,
    )

    # Verify basic split integrity
    assert len(train_df) + len(test_df) == len(clustered_df)
    assert set(train_df["representative_sequence"]).isdisjoint(
        set(test_df["representative_sequence"])
    )

    # Run a baseline split for comparison
    from mmseqspy import split

    baseline_train, baseline_test = split(
        clustered_df,
        group_col="representative_sequence",
        test_size=0.3,
        random_state=42,
    )

    # Calculate and compare balance metrics
    milp_imbalances = []
    baseline_imbalances = []

    for prop in num_props:
        # MILP split balance
        milp_train_mean = train_df[prop].mean()
        milp_test_mean = test_df[prop].mean()
        overall_mean = clustered_df[prop].mean()
        milp_imbalance = abs(milp_train_mean - milp_test_mean) / overall_mean
        milp_imbalances.append(milp_imbalance)

        # Baseline split balance
        baseline_train_mean = baseline_train[prop].mean()
        baseline_test_mean = baseline_test[prop].mean()
        baseline_imbalance = (
            abs(baseline_train_mean - baseline_test_mean) / overall_mean
        )
        baseline_imbalances.append(baseline_imbalance)

        # MILP should achieve reasonable balance
        assert milp_imbalance < 0.25, (
            f"Property {prop} poorly balanced: diff ratio = {milp_imbalance:.2f}"
        )

    # MILP should be better than baseline for at least one property
    assert any(m < b for m, b in zip(milp_imbalances, baseline_imbalances)), (
        "MILP split didn't improve balance for any property compared to baseline"
    )


def test_milp_categorical_properties(realistic_protein_data, mmseqs_installed):
    """Test MILP balancing of categorical properties."""
    df = realistic_protein_data.copy()

    # Add an extra categorical column with imbalanced distribution
    # 70% of sequences in category A, 30% in category B
    np.random.seed(42)
    df["category"] = np.random.choice(["A", "B"], size=len(df), p=[0.7, 0.3])

    # Cluster the data
    clustered_df = cluster(df, sequence_col="sequence", id_col="id", min_seq_id=0.8)

    # Run MILP split with categorical balancing
    train_df, test_df = milp_split(
        clustered_df,
        group_col="representative_sequence",
        test_size=0.3,
        balance_cols=["molecular_weight"],  # Include one numeric property
        categorical_cols=["domains", "organism", "category"],
        balance_weight=1.0,
        time_limit=15,
        random_state=42,
    )

    # Verify split integrity
    assert len(train_df) + len(test_df) == len(clustered_df)

    # Calculate categorical distribution similarity
    for cat_col in ["domains", "organism", "category"]:
        # Get distributions
        train_dist = train_df[cat_col].value_counts(normalize=True)
        test_dist = test_df[cat_col].value_counts(normalize=True)
        overall_dist = clustered_df[cat_col].value_counts(normalize=True)

        # Calculate Jensen-Shannon divergence (simplified)
        # Lower values indicate more similar distributions
        train_test_div = 0
        all_cats = set(train_dist.index) | set(test_dist.index)
        for cat in all_cats:
            train_val = train_dist.get(cat, 0)
            test_val = test_dist.get(cat, 0)
            train_test_div += abs(train_val - test_val) / 2

        # For comparison, calculate divergence between test and overall
        random_div = 0
        for cat in all_cats:
            test_val = test_dist.get(cat, 0)
            overall_val = overall_dist.get(cat, 0)
            random_div += abs(test_val - overall_val) / 2

        # MILP should achieve good balance (JS div < 0.2 is good)
        assert train_test_div < 0.2, (
            f"Categorical variable {cat_col} poorly balanced: JS div = {train_test_div:.2f}"
        )

        # MILP should achieve reasonable balance, either relative to random
        # or below an absolute threshold for good balance
        assert train_test_div <= random_div * 2.5 or train_test_div < 0.15, (
            f"MILP split has poor {cat_col} balance. JS div = {train_test_div:.2f} vs random {random_div:.2f}"
        )


def test_milp_variance_balance(realistic_protein_data, mmseqs_installed):
    """Test MILP balancing of variance and range for properties."""
    df = realistic_protein_data.copy()

    # Add a property with very different variance in different groups
    np.random.seed(42)
    family_variances = {}
    for family in df["family_id"].unique():
        # Some families have high variance, some have low
        if family % 2 == 0:
            family_variances[family] = 0.5  # Low variance
        else:
            family_variances[family] = 2.0  # High variance

    df["variable_property"] = df.apply(
        lambda row: np.random.normal(
            row["family_id"], family_variances[row["family_id"]]
        ),
        axis=1,
    )

    # Cluster the data
    clustered_df = cluster(df, sequence_col="sequence", id_col="id", min_seq_id=0.8)

    # Run MILP split with variance balancing
    train_df, test_df = milp_split(
        clustered_df,
        group_col="representative_sequence",
        test_size=0.3,
        balance_cols=["variable_property"],
        balance_weight=0.5,
        variance_weight=1.0,  # Focus on variance
        range_weight=1.0,  # And range
        time_limit=15,
        random_state=42,
    )

    # Calculate variance metrics
    train_var = train_df["variable_property"].var()
    test_var = test_df["variable_property"].var()
    overall_var = clustered_df["variable_property"].var()

    # Calculate variance difference
    var_diff_ratio = abs(train_var - test_var) / overall_var

    # Calculate range metrics
    train_range = (
        train_df["variable_property"].max() - train_df["variable_property"].min()
    )
    test_range = test_df["variable_property"].max() - test_df["variable_property"].min()
    overall_range = (
        clustered_df["variable_property"].max()
        - clustered_df["variable_property"].min()
    )

    # Calculate range coverage
    train_coverage = train_range / overall_range
    test_coverage = test_range / overall_range

    # Variance should be reasonably balanced
    assert var_diff_ratio < 0.4, (
        f"Variance poorly balanced: diff ratio = {var_diff_ratio:.2f}"
    )

    # Both splits should cover a substantial portion of the range
    assert train_coverage > 0.7, f"Train set covers only {train_coverage:.2f} of range"
    assert test_coverage > 0.7, f"Test set covers only {test_coverage:.2f} of range"


def test_milp_time_limit(realistic_protein_data, mmseqs_installed):
    """Test MILP solver with different time limits."""
    df = realistic_protein_data.copy()

    # Cluster the data
    clustered_df = cluster(df, sequence_col="sequence", id_col="id", min_seq_id=0.8)

    # Run with very short time limit
    import time

    start_time = time.time()
    train_short, test_short = milp_split(
        clustered_df,
        group_col="representative_sequence",
        test_size=0.3,
        balance_cols=["molecular_weight", "hydrophobicity"],
        time_limit=1,  # Very short - might not find optimal
        random_state=42,
    )
    short_time = time.time() - start_time

    # Run with longer time limit
    start_time = time.time()
    train_long, test_long = milp_split(
        clustered_df,
        group_col="representative_sequence",
        test_size=0.3,
        balance_cols=["molecular_weight", "hydrophobicity"],
        time_limit=5,  # Still short but more time to optimize
        random_state=42,
    )
    long_time = time.time() - start_time

    # Verify both splits are valid
    assert len(train_short) + len(test_short) == len(clustered_df)
    assert len(train_long) + len(test_long) == len(clustered_df)

    # Longer time should lead to better balancing or the same result
    # Calculate imbalance for short run
    short_imb = (
        abs(
            train_short["molecular_weight"].mean()
            - test_short["molecular_weight"].mean()
        )
        / clustered_df["molecular_weight"].mean()
    )

    # Calculate imbalance for long run
    long_imb = (
        abs(
            train_long["molecular_weight"].mean() - test_long["molecular_weight"].mean()
        )
        / clustered_df["molecular_weight"].mean()
    )

    # Longer run should be at least as good (not significantly worse)
    assert long_imb <= short_imb * 1.1, (
        f"Longer optimization time produced worse balance: {long_imb:.3f} vs {short_imb:.3f}"
    )

    # The timing should reflect the limits
    assert short_time < long_time or abs(short_time - long_time) < 0.5, (
        f"Time limits not reflected in runtime: short={short_time:.2f}s, long={long_time:.2f}s"
    )
