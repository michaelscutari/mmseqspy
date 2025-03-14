import pandas as pd
import numpy as np
from .logger import logger
from .utils import check_random_state
from .clustering import (
    cluster as perform_clustering,
)


def split(
    df,
    group_col="representative_sequence",
    test_size=0.2,
    random_state=None,
    tolerance=0.05,
):
    """
    Splits DataFrame into train/test sets based on grouping in a specified column.

    Parameters:
        df (pd.DataFrame): DataFrame to split.
        group_col (str): Column by which to group before splitting.
        test_size (float): Desired fraction of data in test set (default 0.2).
        random_state (int): Random state for reproducibility in group selection.
        tolerance (float): Acceptable deviation from test_size (default 0.05).

    Returns:
        (pd.DataFrame, pd.DataFrame): (train_df, test_df)
    """
    logger.info(f"Splitting data by '{group_col}' with target test size {test_size}")

    # Get random state for reproducibility
    rng = check_random_state(random_state)

    total_sequences = len(df)
    target_test_count = int(round(test_size * total_sequences))

    logger.info(f"Total sequence count: {total_sequences}")
    logger.info(f"Target test count: {target_test_count}")

    # Get group sizes
    size_per_group = df.groupby(group_col).size()

    # Create tuples of (group, size) and sort them deterministically
    group_size_pairs = [(group, size) for group, size in size_per_group.items()]
    group_size_pairs.sort(key=lambda x: str(x[0]))  # Deterministic sorting

    # Shuffle using the global random state
    rng.shuffle(group_size_pairs)

    groups = [pair[0] for pair in group_size_pairs]
    sizes = [pair[1] for pair in group_size_pairs]

    logger.debug(f"Found {len(groups)} unique groups in '{group_col}'")
    logger.debug("Finding optimal subset-sum solution for test set")

    # Use dynamic programming to find subset of groups that gets closest to target test size
    dp = {0: []}
    for idx, group_size in enumerate(sizes):
        current_dp = dict(dp)
        for current_sum, idx_list in dp.items():
            new_sum = current_sum + group_size
            if new_sum not in current_dp:
                current_dp[new_sum] = idx_list + [idx]
        dp = current_dp

    # Find the sum closest to target_test_count
    all_sums = sorted(dp.keys())

    distances = [(s, abs(s - target_test_count)) for s in all_sums]
    min_distance = min(distances, key=lambda x: x[1])[1]

    # Find all sums that have this minimum distance
    candidate_sums = [s for s, d in distances if d == min_distance]

    # Ensure deterministic selection
    best_sum = min(candidate_sums)

    # Sort the indices for deterministic selection
    best_group_indices = sorted(dp[best_sum])
    chosen_groups = [groups[i] for i in best_group_indices]

    logger.debug(f"Best achievable test set size: {best_sum} sequences")
    logger.debug(f"Selected {len(chosen_groups)} groups for test set")

    test_df = df[df[group_col].isin(chosen_groups)]
    train_df = df[~df[group_col].isin(chosen_groups)]

    achieved_test_fraction = len(test_df) / total_sequences

    logger.info(
        f"Train set: {len(train_df)} sequences ({len(train_df) / total_sequences:.2%})"
    )
    logger.info(f"Test set: {len(test_df)} sequences ({achieved_test_fraction:.2%})")

    if abs(achieved_test_fraction - test_size) > tolerance:
        logger.warning(
            f"Desired test fraction = {test_size:.2f}, "
            f"achieved = {achieved_test_fraction:.2f}. "
            "This is the closest possible split given the constraint to keep groups together."
        )

    return train_df, test_df


def train_test_cluster_split(
    df,
    sequence_col,
    id_col=None,
    test_size=0.2,
    min_seq_id=0.3,
    coverage=0.5,
    cov_mode=0,
    alignment_mode=0,
    cluster_mode=0,
    cluster_steps=1,
    random_state=None,
    tolerance=0.05,
):
    """
    Clusters sequences and splits data into train/test sets by grouping entire clusters.
    """
    logger.info("Performing combined clustering and train/test split")
    logger.info(
        f"Parameters: sequence_col='{sequence_col}', id_col='{id_col}', test_size={test_size}"
    )
    logger.info(
        f"Clustering parameters: min_seq_id={min_seq_id}, coverage={coverage}, cov_mode={cov_mode}, "
        f"alignment_mode={alignment_mode}, cluster_mode={cluster_mode}, cluster_steps={cluster_steps}"
    )

    from .utils import _check_mmseqs, _validate_clustering_params

    _check_mmseqs()
    _validate_clustering_params(
        min_seq_id, coverage, cov_mode, alignment_mode, cluster_mode, cluster_steps
    )

    logger.info("Step 1: Clustering sequences")
    df_clustered = perform_clustering(  # Use the renamed import
        df=df,
        sequence_col=sequence_col,
        id_col=id_col,
        min_seq_id=min_seq_id,
        coverage=coverage,
        cov_mode=cov_mode,
        alignment_mode=alignment_mode,
        cluster_mode=cluster_mode,
        cluster_steps=cluster_steps,
    )

    logger.info("Step 2: Splitting data based on sequence clusters")
    return split(
        df=df_clustered,
        group_col="representative_sequence",
        test_size=test_size,
        random_state=random_state,
        tolerance=tolerance,
    )


def train_test_val_cluster_split(
    df,
    sequence_col,
    id_col=None,
    test_size=0.2,
    val_size=0.1,
    min_seq_id=0.3,
    coverage=0.5,
    cov_mode=0,
    alignment_mode=0,
    cluster_mode=0,
    cluster_steps=1,
    random_state=None,
    tolerance=0.05,
):
    """
    Clusters sequences and splits data into train, val, and test sets by grouping entire clusters.
    """
    logger.info("Performing 3-way train/validation/test split with clustering")
    logger.info(f"Parameters: sequence_col='{sequence_col}', id_col='{id_col}'")
    logger.info(f"Split sizes: test_size={test_size}, val_size={val_size}")
    logger.info(
        f"Clustering parameters: min_seq_id={min_seq_id}, coverage={coverage}, cov_mode={cov_mode}, "
        f"alignment_mode={alignment_mode}, cluster_mode={cluster_mode}, cluster_steps={cluster_steps}"
    )

    from .utils import _check_mmseqs, _validate_clustering_params

    _check_mmseqs()
    _validate_clustering_params(
        min_seq_id, coverage, cov_mode, alignment_mode, cluster_mode, cluster_steps
    )

    logger.info("Step 1: Clustering sequences")
    df_clustered = perform_clustering(  # Use the renamed import
        df=df,
        sequence_col=sequence_col,
        id_col=id_col,
        min_seq_id=min_seq_id,
        coverage=coverage,
        cov_mode=cov_mode,
        alignment_mode=alignment_mode,
        cluster_mode=cluster_mode,
        cluster_steps=cluster_steps,
    )

    logger.info("Step 2: Splitting into train+val vs test")
    train_val_df, test_df = split(
        df=df_clustered,
        group_col="representative_sequence",
        test_size=test_size,
        random_state=random_state,
        tolerance=tolerance,
    )

    logger.info("Step 3: Further splitting train+val into train vs val")
    adjusted_val_fraction = val_size / (1.0 - test_size)
    logger.debug(
        f"Adjusted validation fraction: {adjusted_val_fraction:.4f} of train+val set"
    )

    train_df, val_df = split(
        df=train_val_df,
        group_col="representative_sequence",
        test_size=adjusted_val_fraction,
        random_state=random_state,
        tolerance=tolerance,
    )

    total = len(df)
    logger.info("Final split results:")
    logger.info(f"  Train: {len(train_df)} sequences ({len(train_df) / total:.2%})")
    logger.info(f"  Validation: {len(val_df)} sequences ({len(val_df) / total:.2%})")
    logger.info(f"  Test: {len(test_df)} sequences ({len(test_df) / total:.2%})")

    return train_df, val_df, test_df


def constrained_split(
    df,
    group_col="representative_sequence",
    id_col=None,
    test_size=0.2,
    random_state=None,
    tolerance=0.05,
    force_train_ids=None,
    force_test_ids=None,
    id_type="sequence",
):
    """
    Splits data with constraints on which sequences or groups must be in the train or test set.
    """
    logger.info("Performing constrained train/test split")

    # Initialize forced IDs if not provided
    force_train_ids = [] if force_train_ids is None else force_train_ids
    force_test_ids = [] if force_test_ids is None else force_test_ids

    # Check for conflicts at the sequence level
    conflicts = set(force_train_ids).intersection(set(force_test_ids))
    if conflicts:
        logger.error(
            f"Found {len(conflicts)} IDs in both force_train_ids and force_test_ids"
        )
        raise ValueError("Cannot force the same IDs to both train and test sets")

    forced_train_groups = set()
    forced_test_groups = set()

    if id_type == "sequence":
        if id_col is None:
            raise ValueError("id_col must be provided when id_type is 'sequence'")
        # Determine forced groups based on forced sequence IDs
        if force_train_ids:
            train_mask = df[id_col].isin(force_train_ids)
            forced_train_groups.update(df.loc[train_mask, group_col].unique())
            logger.info(
                f"Forcing {len(forced_train_groups)} groups to train based on {len(force_train_ids)} sequence IDs"
            )
        if force_test_ids:
            test_mask = df[id_col].isin(force_test_ids)
            forced_test_groups.update(df.loc[test_mask, group_col].unique())
            logger.info(
                f"Forcing {len(forced_test_groups)} groups to test based on {len(force_test_ids)} sequence IDs"
            )
    elif id_type in ["cluster", "group"]:
        forced_train_groups.update(force_train_ids)
        forced_test_groups.update(force_test_ids)
        logger.info(
            f"Forcing {len(forced_train_groups)} groups to train and {len(forced_test_groups)} to test"
        )
    else:
        raise ValueError(f"Invalid id_type: {id_type}. Must be 'sequence' or 'cluster'")

    # Check for conflicts at the group level
    group_conflicts = forced_train_groups.intersection(forced_test_groups)
    if group_conflicts:
        logger.error(
            f"Found {len(group_conflicts)} groups forced to both train and test"
        )
        raise ValueError(
            "Constraint conflict: some groups are forced to both train and test sets"
        )

    # Create forced splits based on group_col membership
    train_forced_mask = df[group_col].isin(forced_train_groups)
    test_forced_mask = df[group_col].isin(forced_test_groups)

    train_forced = df[train_forced_mask]
    test_forced = df[test_forced_mask]
    remaining = df[~(train_forced_mask | test_forced_mask)]

    total_size = len(df)
    train_forced_size = len(train_forced)
    test_forced_size = len(test_forced)
    remaining_size = len(remaining)

    logger.info(
        f"Pre-assigned {train_forced_size} sequences to train ({train_forced_size / total_size:.2%})"
    )
    logger.info(
        f"Pre-assigned {test_forced_size} sequences to test ({test_forced_size / total_size:.2%})"
    )
    logger.info(
        f"Remaining {remaining_size} sequences to split ({remaining_size / total_size:.2%})"
    )

    if test_forced_size / total_size > test_size + tolerance:
        logger.warning(
            f"Forced test assignments ({test_forced_size / total_size:.2%}) exceed desired test size ({test_size:.2%}) by more than tolerance ({tolerance:.2%})"
        )

    # Calculate target test size for the remaining sequences
    target_test_from_remaining = max(0, (test_size * total_size) - test_forced_size)
    if remaining_size > 0:
        adjusted_test_size = target_test_from_remaining / remaining_size
        adjusted_test_size = min(1.0, max(0.0, adjusted_test_size))
        logger.info(f"Adjusted test size for remaining data: {adjusted_test_size:.2%}")

        # Split remaining data using the existing group-aware split function
        if 0 < adjusted_test_size < 1:
            train_remaining, test_remaining = split(
                df=remaining,
                group_col=group_col,
                test_size=adjusted_test_size,
                random_state=random_state,
                tolerance=tolerance,
            )
        elif adjusted_test_size <= 0:
            train_remaining, test_remaining = remaining.copy(), remaining.iloc[0:0]
            logger.info("All remaining sequences assigned to train set")
        else:  # adjusted_test_size >= 1
            train_remaining, test_remaining = remaining.iloc[0:0], remaining.copy()
            logger.info("All remaining sequences assigned to test set")
    else:
        train_remaining = pd.DataFrame(columns=df.columns)
        test_remaining = pd.DataFrame(columns=df.columns)

    # Combine forced and split remaining data
    train_df = pd.concat([train_forced, train_remaining])
    test_df = pd.concat([test_forced, test_remaining])

    final_train_pct = len(train_df) / total_size
    final_test_pct = len(test_df) / total_size

    logger.info("Final split results:")
    logger.info(f"  Train: {len(train_df)} sequences ({final_train_pct:.2%})")
    logger.info(f"  Test: {len(test_df)} sequences ({final_test_pct:.2%})")

    if abs(final_test_pct - test_size) > tolerance:
        logger.warning(
            f"Final test fraction ({final_test_pct:.2%}) differs from target ({test_size:.2%}) by more than tolerance ({tolerance:.2%})"
        )

    return train_df, test_df


def cluster_kfold(
    df,
    sequence_col,
    id_col=None,
    n_splits=5,
    min_seq_id=0.3,
    coverage=0.5,
    cov_mode=0,
    alignment_mode=0,
    cluster_mode=0,
    cluster_steps=1,
    random_state=None,
    shuffle=True,
    return_indices=False,
):
    """
    Performs k-fold cross-validation while respecting sequence clustering.
    """
    logger.info(
        f"Performing {n_splits}-fold cross-validation with cluster-aware splits"
    )

    # Get random state for reproducibility
    rng = check_random_state(random_state)

    from .utils import _validate_clustering_params

    _validate_clustering_params(
        min_seq_id, coverage, cov_mode, alignment_mode, cluster_mode, cluster_steps
    )

    # First, cluster the sequences
    df_clustered = perform_clustering(  # Use the renamed import
        df=df,
        sequence_col=sequence_col,
        id_col=id_col,
        min_seq_id=min_seq_id,
        coverage=coverage,
        cov_mode=cov_mode,
        alignment_mode=alignment_mode,
        cluster_mode=cluster_mode,
        cluster_steps=cluster_steps,
    )

    # Get unique clusters and their sizes
    cluster_sizes = df_clustered.groupby("representative_sequence").size()
    clusters_list = list(cluster_sizes.index)  # Renamed to avoid shadowing the import

    logger.info(f"Found {len(clusters_list)} unique sequence clusters")

    if len(clusters_list) < n_splits:
        logger.warning(
            f"Number of clusters ({len(clusters_list)}) is less than requested folds ({n_splits}). "
            f"Some folds will be empty."
        )

    # Sort clusters by size (descending) for better balancing
    clusters_with_sizes = [
        (cluster_name, cluster_sizes[cluster_name]) for cluster_name in clusters_list
    ]
    clusters_with_sizes.sort(key=lambda x: x[1], reverse=True)

    # Optional shuffling of similarly-sized clusters to ensure randomness
    if shuffle:
        # Group clusters by size
        size_groups = {}
        for cluster_name, size in clusters_with_sizes:
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(cluster_name)

        # Shuffle each size group
        for size in size_groups:
            rng.shuffle(size_groups[size])

        # Reconstruct clusters_with_sizes with preserved size ordering but shuffled within size
        clusters_with_sizes = []
        for size in sorted(size_groups.keys(), reverse=True):
            for cluster_name in size_groups[size]:
                clusters_with_sizes.append((cluster_name, size))

    # Initialize fold assignments
    fold_sizes = [0] * n_splits
    cluster_to_fold = {}

    # Assign each cluster to the smallest fold
    for cluster_name, size in clusters_with_sizes:
        smallest_fold = fold_sizes.index(min(fold_sizes))
        fold_sizes[smallest_fold] += size
        cluster_to_fold[cluster_name] = smallest_fold

    # Report fold balance
    total_size = sum(fold_sizes)
    fold_percentages = [size / total_size for size in fold_sizes]

    logger.info(f"Fold sizes: {fold_sizes}")
    logger.info(f"Fold percentages: {[f'{p:.1%}' for p in fold_percentages]}")

    # Check for imbalanced folds
    fold_imbalance = max(fold_percentages) - min(fold_percentages)
    if fold_imbalance > 0.1:  # More than 10% difference
        logger.warning(
            f"Folds are imbalanced (max={max(fold_percentages):.1%}, min={min(fold_percentages):.1%}, "
            f"diff={fold_imbalance:.1%}). Consider adjusting clustering parameters."
        )

    # For each fold, create train/test split
    result = []
    for fold_idx in range(n_splits):
        logger.info(f"Preparing fold {fold_idx + 1}/{n_splits}")

        # Create mask for test set (current fold)
        test_mask = df_clustered["representative_sequence"].apply(
            lambda x: cluster_to_fold.get(x) == fold_idx
        )

        if return_indices:
            train_indices = df_clustered[~test_mask].index
            test_indices = df_clustered[test_mask].index
            result.append((train_indices, test_indices))
        else:
            train_df = df_clustered[~test_mask].copy()
            test_df = df_clustered[test_mask].copy()
            result.append((train_df, test_df))

        n_train = len(train_indices if return_indices else train_df)
        n_test = len(test_indices if return_indices else test_df)
        logger.info(
            f"Fold {fold_idx + 1}: train={n_train} ({n_train / (n_train + n_test):.1%}), "
            f"test={n_test} ({n_test / (n_train + n_test):.1%})"
        )

    return result


def milp_split(
    df,
    group_col="representative_sequence",
    test_size=0.2,
    balance_cols=None,
    categorical_cols=None,  # New parameter for categorical features
    residue_cols=None,  # New parameter for residue-level features
    balance_weight=1.0,
    variance_weight=0.5,  # Weight for variance balancing
    range_weight=0.5,  # Weight for min/max range balancing
    time_limit=60,
    random_state=None,
):
    """
    Splits DataFrame into train/test sets using Mixed Integer Linear Programming (MILP)
    to balance multiple properties across the splits, including distribution similarity.

    Parameters:
        df (pd.DataFrame): DataFrame to split (already clustered).
        group_col (str): Column indicating cluster membership.
        test_size (float): Target fraction of data in test set (0-1).
        balance_cols (list): Columns to balance across splits.
        categorical_cols (list): Categorical columns to balance using one-hot encoding.
        residue_cols (list): Columns containing per-residue data (arrays/lists) to process
                             into scalar metrics for balancing.
        balance_weight (float): Weight of balance terms in objective (higher = more important).
        variance_weight (float): Weight for variance balancing (higher = more important).
        range_weight (float): Weight for min/max range balancing (higher = more important).
        time_limit (int): Maximum time in seconds to spend solving.
        random_state (int): Random seed for reproducibility.

    Returns:
        (pd.DataFrame, pd.DataFrame): (train_df, test_df)
    """
    try:
        from pulp import (
            LpProblem,
            LpVariable,
            LpMinimize,
            lpSum,
            LpStatus,
            LpBinary,
            value,
            PULP_CBC_CMD,
        )
    except ImportError:
        logger.error(
            "PuLP is required for MILP-based splitting. Install with 'pip install pulp'"
        )
        raise ImportError("PuLP is required for MILP-based splitting")

    logger.info(f"Performing MILP-based splitting with target test size {test_size}")

    if balance_cols is None:
        balance_cols = []

    # Create a working copy of the dataframe for feature processing
    working_df = df.copy()

    # Keep track of derived columns for balancing
    derived_cols = []

    # Process categorical columns (one-hot encoding approach)
    if categorical_cols:
        for cat_col in categorical_cols:
            if cat_col not in df.columns:
                logger.warning(f"Categorical column '{cat_col}' not found, skipping")
                continue

            logger.info(f"Processing categorical column: {cat_col}")

            # Get unique categories
            categories = df[cat_col].unique()

            # Create one-hot encoded columns
            for cat in categories:
                col_name = f"{cat_col}_{cat}"
                working_df[col_name] = (df[cat_col] == cat).astype(float)
                derived_cols.append(col_name)

            # Replace original categorical column in balance_cols with derived columns
            if cat_col in balance_cols:
                balance_cols.remove(cat_col)

        # Add all categorical derived columns to balance_cols
        balance_cols.extend([col for col in derived_cols if col not in balance_cols])

    # Process residue-level columns (summary statistics approach)
    if residue_cols:
        residue_derived_cols = []

        for res_col in residue_cols:
            if res_col not in df.columns:
                logger.warning(f"Residue-level column '{res_col}' not found, skipping")
                continue

            logger.info(f"Processing residue-level column: {res_col}")

            # Basic summary statistics
            try:
                # Mean value across residues
                working_df[f"{res_col}_mean"] = df[res_col].apply(
                    lambda x: np.mean(x)
                    if isinstance(x, (list, np.ndarray))
                    else np.nan
                )

                # Min and max values
                working_df[f"{res_col}_min"] = df[res_col].apply(
                    lambda x: np.min(x) if isinstance(x, (list, np.ndarray)) else np.nan
                )
                working_df[f"{res_col}_max"] = df[res_col].apply(
                    lambda x: np.max(x) if isinstance(x, (list, np.ndarray)) else np.nan
                )

                # Variance across residues
                working_df[f"{res_col}_var"] = df[res_col].apply(
                    lambda x: np.var(x)
                    if isinstance(x, (list, np.ndarray)) and len(x) > 1
                    else np.nan
                )

                # Distribution percentiles
                working_df[f"{res_col}_p25"] = df[res_col].apply(
                    lambda x: np.percentile(x, 25)
                    if isinstance(x, (list, np.ndarray)) and len(x) > 0
                    else np.nan
                )
                working_df[f"{res_col}_p50"] = df[res_col].apply(
                    lambda x: np.percentile(x, 50)
                    if isinstance(x, (list, np.ndarray)) and len(x) > 0
                    else np.nan
                )
                working_df[f"{res_col}_p75"] = df[res_col].apply(
                    lambda x: np.percentile(x, 75)
                    if isinstance(x, (list, np.ndarray)) and len(x) > 0
                    else np.nan
                )

                # Add derived columns to the list
                new_cols = [
                    f"{res_col}_mean",
                    f"{res_col}_min",
                    f"{res_col}_max",
                    f"{res_col}_var",
                    f"{res_col}_p25",
                    f"{res_col}_p50",
                    f"{res_col}_p75",
                ]
                residue_derived_cols.extend(new_cols)

                # Replace NaN values with column means to avoid issues in optimization
                for col in new_cols:
                    if working_df[col].isna().any():
                        col_mean = working_df[col].mean()
                        working_df[col].fillna(col_mean, inplace=True)
                        logger.warning(
                            f"Filled NaN values in {col} with mean: {col_mean:.4f}"
                        )

            except Exception as e:
                logger.error(f"Error processing residue column {res_col}: {str(e)}")
                continue

        # Add residue-derived columns to balance_cols
        balance_cols.extend(
            [col for col in residue_derived_cols if col not in balance_cols]
        )

    # Get unique clusters and their sizes
    group_sizes = df.groupby(group_col).size()
    groups = list(group_sizes.index)
    total_size = len(df)

    # For each property to balance, calculate the sum and variance components per group
    group_properties = {}
    group_variances = {}  # Store variance components per group
    group_min_values = {}  # Store min values per property
    group_max_values = {}  # Store max values per property

    # Process all columns to balance (original + derived)
    for col in balance_cols:
        if col not in working_df.columns:
            logger.warning(
                f"Column '{col}' not found in working DataFrame, ignoring for balance"
            )
            continue

        # Calculate sum of values per group (for mean balancing)
        group_properties[col] = working_df.groupby(group_col)[col].sum()

        # Calculate values needed for variance balancing
        global_mean = working_df[col].mean()

        # For each group, calculate its contribution to variance
        # Variance component = sum((x - global_mean)²)
        variance_components = working_df.groupby(group_col)[col].apply(
            lambda x: ((x - global_mean) ** 2).sum()
        )
        group_variances[col] = variance_components

        # Find groups with min/max values for range balancing
        group_min_values[col] = {}
        group_max_values[col] = {}

        # For each group, check if it contains min or max values
        group_min = working_df.groupby(group_col)[col].min()
        group_max = working_df.groupby(group_col)[col].max()

        # Identify groups with values close to global min/max
        global_min = working_df[col].min()
        global_max = working_df[col].max()

        # Flag groups having values within 5% of the global range to min/max
        if global_max > global_min:
            range_threshold = 0.05 * (global_max - global_min)
            for group in groups:
                if (
                    group in group_min
                    and (group_min[group] - global_min) <= range_threshold
                ):
                    group_min_values[col][group] = 1
                else:
                    group_min_values[col][group] = 0

                if (
                    group in group_max
                    and (global_max - group_max[group]) <= range_threshold
                ):
                    group_max_values[col][group] = 1
                else:
                    group_max_values[col][group] = 0
        else:
            # Handle case where all values are the same (global_max == global_min)
            for group in groups:
                group_min_values[col][group] = 1
                group_max_values[col][group] = 1

    # Create the MILP problem
    prob = LpProblem("ClusterSplit", LpMinimize)

    # Decision variables (1 if group is in test set, 0 if in train set)
    group_vars = {
        group: LpVariable(f"group_{i}", cat=LpBinary) for i, group in enumerate(groups)
    }

    # Variable to represent absolute difference from target test size
    size_deviation_var = LpVariable("size_deviation", lowBound=0)

    # Calculate total for each property
    property_totals = {
        col: group_prop.sum() for col, group_prop in group_properties.items()
    }
    variance_totals = {
        col: variance_comp.sum() for col, variance_comp in group_variances.items()
    }

    # Variables to represent absolute differences for property balances
    balance_vars = {
        col: LpVariable(f"balance_{col}", lowBound=0) for col in group_properties.keys()
    }

    # Variables to represent absolute differences for variance balances
    variance_vars = {
        col: LpVariable(f"variance_{col}", lowBound=0) for col in group_variances.keys()
    }

    # Variables to represent min/max range differences
    min_range_vars = {
        col: LpVariable(f"min_range_{col}", lowBound=0)
        for col in group_min_values.keys()
    }
    max_range_vars = {
        col: LpVariable(f"max_range_{col}", lowBound=0)
        for col in group_max_values.keys()
    }

    # Objective: minimize all deviations with their respective weights
    objective_terms = [
        size_deviation_var,  # Size deviation (weight 1.0)
        balance_weight * lpSum(balance_vars.values()),  # Mean balance
        variance_weight * lpSum(variance_vars.values()),  # Variance balance
        range_weight
        * lpSum(
            list(min_range_vars.values()) + list(max_range_vars.values())
        ),  # Range balance
    ]

    prob += lpSum(objective_terms)

    # Constraints for size deviation
    target_test_count = test_size * total_size
    test_count = lpSum([group_sizes[group] * group_vars[group] for group in groups])
    prob += size_deviation_var >= test_count - target_test_count
    prob += size_deviation_var >= target_test_count - test_count

    # Constraints for property balance (mean)
    for col, group_prop in group_properties.items():
        test_prop_sum = lpSum(
            [group_vars[group] * group_prop[group] for group in groups]
        )
        target_test_prop_sum = test_size * property_totals[col]
        prob += balance_vars[col] >= test_prop_sum - target_test_prop_sum
        prob += balance_vars[col] >= target_test_prop_sum - test_prop_sum

    # Constraints for variance balance
    for col, variance_comp in group_variances.items():
        test_variance_sum = lpSum(
            [
                group_vars[group] * variance_comp[group]
                for group in groups
                if group in variance_comp
            ]
        )
        target_test_variance_sum = test_size * variance_totals[col]
        prob += variance_vars[col] >= test_variance_sum - target_test_variance_sum
        prob += variance_vars[col] >= target_test_variance_sum - test_variance_sum

    # Constraints for min/max range balance
    for col in group_min_values.keys():
        # Count of min-range groups in test set
        min_range_test_count = lpSum(
            [
                group_vars[group] * group_min_values[col][group]
                for group in groups
                if group in group_min_values[col]
            ]
        )
        # Target proportion based on test_size
        min_range_total = sum(group_min_values[col].values())
        if min_range_total > 0:
            target_min_range_test = test_size * min_range_total
            prob += min_range_vars[col] >= min_range_test_count - target_min_range_test
            prob += min_range_vars[col] >= target_min_range_test - min_range_test_count

        # Count of max-range groups in test set
        max_range_test_count = lpSum(
            [
                group_vars[group] * group_max_values[col][group]
                for group in groups
                if group in group_max_values[col]
            ]
        )
        # Target proportion based on test_size
        max_range_total = sum(group_max_values[col].values())
        if max_range_total > 0:
            target_max_range_test = test_size * max_range_total
            prob += max_range_vars[col] >= max_range_test_count - target_max_range_test
            prob += max_range_vars[col] >= target_max_range_test - max_range_test_count

    # Try to configure solver with time limit
    from pulp import PulpSolverError, COIN_CMD

    logger.info(f"Attempting to set MILP solver time limit to {time_limit} seconds")

    # Try different solver options in order of preference
    solver = None

    # Option 1: Try PuLP's CBC CMD interface
    try:
        solver = PULP_CBC_CMD(timeLimit=time_limit)
        prob.solve(solver)
    except PulpSolverError as e:
        logger.warning(f"CBC solver not available: {e}")
        solver = None

    # Option 2: Try COIN_CMD if available (needs coinor-cbc installed)
    if solver is None:
        try:
            logger.info("Trying COIN_CMD solver")
            solver = COIN_CMD(timeLimit=time_limit)
            prob.solve(solver)
        except (PulpSolverError, Exception) as e:
            logger.warning(f"COIN_CMD solver not available: {e}")
            solver = None

    # Option 3: Fall back to default solver with no time limit
    if solver is None:
        logger.warning(
            "No time-limited solver available. Using default solver without time limit."
        )
        logger.warning(
            "To enable time limits, install CBC solver: pip install pulp[cbc]"
        )
        prob.solve()

    logger.info(f"MILP solution status: {LpStatus[prob.status]}")

    if prob.status != 1:  # Not optimal
        logger.warning("MILP solution not optimal, using best found solution")

    # Extract results
    test_groups = [group for group, var in group_vars.items() if value(var) > 0.5]

    # Create train and test DataFrames (using original dataframe)
    test_df = df[df[group_col].isin(test_groups)]
    train_df = df[~df[group_col].isin(test_groups)]

    # Report results
    achieved_test_fraction = len(test_df) / total_size
    logger.info(
        f"Train set: {len(train_df)} sequences ({len(train_df) / total_size:.2%})"
    )
    logger.info(f"Test set: {len(test_df)} sequences ({achieved_test_fraction:.2%})")

    # Report balance of properties (original columns first)
    original_cols = [col for col in balance_cols if col in df.columns]
    for col in original_cols:
        # Report mean balance
        train_mean = train_df[col].mean() if len(train_df) > 0 else 0
        test_mean = test_df[col].mean() if len(test_df) > 0 else 0
        mean_diff_pct = (
            abs(test_mean - train_mean)
            / (abs(train_mean) if train_mean != 0 else 1)
            * 100
        )

        # Report variance balance
        train_var = train_df[col].var() if len(train_df) > 1 else 0
        test_var = test_df[col].var() if len(test_df) > 1 else 0
        var_diff_pct = (
            abs(test_var - train_var) / (abs(train_var) if train_var != 0 else 1) * 100
        )

        # Report min/max balance
        train_min = train_df[col].min() if len(train_df) > 0 else 0
        test_min = test_df[col].min() if len(test_df) > 0 else 0
        train_max = train_df[col].max() if len(train_df) > 0 else 0
        test_max = test_df[col].max() if len(test_df) > 0 else 0

        logger.info(f"Property '{col}':")
        logger.info(
            f"  Mean: train={train_mean:.4f}, test={test_mean:.4f}, diff={mean_diff_pct:.2f}%"
        )
        logger.info(
            f"  Variance: train={train_var:.4f}, test={test_var:.4f}, diff={var_diff_pct:.2f}%"
        )
        logger.info(
            f"  Range: train=[{train_min:.4f}, {train_max:.4f}], test=[{test_min:.4f}, {test_max:.4f}]"
        )

    # Report balance of categorical variables
    if categorical_cols:
        for cat_col in categorical_cols:
            if cat_col not in df.columns:
                continue

            # Get category distributions
            train_dist = train_df[cat_col].value_counts(normalize=True)
            test_dist = test_df[cat_col].value_counts(normalize=True)

            # Ensure both distributions have the same categories
            all_cats = set(train_dist.index).union(set(test_dist.index))
            for cat in all_cats:
                if cat not in train_dist:
                    train_dist[cat] = 0
                if cat not in test_dist:
                    test_dist[cat] = 0

            # Sort by category name
            train_dist = train_dist.sort_index()
            test_dist = test_dist.sort_index()

            # Calculate distribution difference
            dist_diff = (
                np.sum(np.abs(train_dist.values - test_dist.values)) / 2
            )  # Jensen-Shannon divergence simplification

            logger.info(f"Categorical property '{cat_col}':")
            logger.info(f"  Train distribution: {dict(train_dist)}")
            logger.info(f"  Test distribution: {dict(test_dist)}")
            logger.info(
                f"  Distribution difference: {dist_diff:.4f} (0=identical, 1=completely different)"
            )

    # Report balance of residue-level derived metrics
    if residue_cols:
        for res_col in residue_cols:
            if res_col in df.columns:
                logger.info(f"Residue-level property '{res_col}' derived metrics:")

                # Report all derived metrics
                for metric in ["mean", "min", "max", "var", "p25", "p50", "p75"]:
                    derived_col = f"{res_col}_{metric}"
                    if derived_col in working_df.columns:
                        # Get the metric values for each split
                        train_val = working_df.loc[train_df.index, derived_col].mean()
                        test_val = working_df.loc[test_df.index, derived_col].mean()
                        val_diff_pct = (
                            abs(test_val - train_val)
                            / (abs(train_val) if train_val != 0 else 1)
                            * 100
                        )

                        logger.info(
                            f"  {metric}: train={train_val:.4f}, test={test_val:.4f}, diff={val_diff_pct:.2f}%"
                        )

    return train_df, test_df
