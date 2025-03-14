# MMseqsPy

A Python library for working with biological sequence data, providing clustering capabilities via MMseqs2 and utilities for train/test splits based on sequence similarity.

---

## Requirements

This library requires [MMseqs2](https://github.com/soedinglab/MMseqs2), which must be installed and accessible via the command line. MMseqs2 can be installed using one of the following methods:

### Installation Options

- **Homebrew**:
    ```bash
    brew install mmseqs2
    ```

- **Conda**:
    ```bash
    conda install -c conda-forge -c bioconda mmseqs2
    ```

- **Docker**:
    ```bash
    docker pull ghcr.io/soedinglab/mmseqs2
    ```

- **Static Build (AVX2, SSE4.1, or SSE2)**:
    ```bash
    wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
    tar xvfz mmseqs-linux-avx2.tar.gz
    export PATH=$(pwd)/mmseqs/bin/:$PATH
    ```

MMseqs2 must be accessible via the `mmseqs` command in your system's PATH. If the library cannot detect MMseqs2, it will raise an error.

## Installation

You can install MMseqsPy using pip:

```bash
pip install mmseqspy
```

Or if installing from source, clone the repository and run the following command in the project root:

```bash
pip install .
```

## Features

This library provides the following functions:

1. **clean**: Preprocesses sequence data by removing invalid characters and sequences.
2. **cluster**: Clusters biological sequences using MMseqs2 and adds cluster information to a pandas DataFrame.
3. **split**: Splits data into train/test sets while respecting cluster groupings.
4. **train_test_cluster_split**: Performs clustering and train/test splitting in one step.
5. **train_test_val_cluster_split**: Creates train/validation/test splits with cluster awareness.
6. **constrained_split**: Creates data splits with constraints on which sequences must be in train or test sets.
7. **cluster_kfold**: Performs k-fold cross-validation while respecting sequence clusters.

## Quick Start

Here's an example of how to use this library:

```python
import pandas as pd
from mmseqspy.cluster_utils import clean, cluster, split, set_verbosity

# Enable more detailed logging (optional)
set_verbosity(verbose=True)

# Example data
df = pd.DataFrame({
    "id": ["seq1", "seq2", "seq3", "seq4"],
    "sequence": ["ACDEFGHIKL", "ACDEFGHIKL", "MNPdQRSTVWY", "MNPQRSTVWY"]
})

# Clean data
clean_df = clean(df, sequence_col="sequence")

# Cluster sequences
clustered_df = cluster(clean_df, sequence_col="sequence", id_col="id")

# Split data into train and test sets
train_df, test_df = split(clustered_df, group_col="representative_sequence", test_size=0.3)

print("Train set:\n", train_df)
print("Test set:\n", test_df)

# Or use the combined function
from mmseqspy.cluster_utils import train_test_cluster_split
train_df, test_df = train_test_cluster_split(df, sequence_col="sequence", id_col="id", test_size=0.3)
```

## Parameters

Common parameters for clustering functions:

- `df`: Pandas DataFrame containing sequence data
- `sequence_col`: Column name containing sequences
- `id_col`: Column name containing unique identifiers
- `min_seq_id`: Minimum sequence identity threshold (0.0-1.0, default 0.3)
- `coverage`: Minimum alignment coverage (0.0-1.0, default 0.5)
- `cov_mode`: Coverage mode (0-3, default 0)
- `test_size`: Desired fraction of data in test set (default 0.2)
- `random_state`: Random seed for reproducibility
- `tolerance`: Acceptable deviation from desired split sizes (default 0.05)

## Advanced Features

### Three-way Splits

```python
from mmseqspy.cluster_utils import train_test_val_cluster_split

train_df, val_df, test_df = train_test_val_cluster_split(
    df, sequence_col="sequence", test_size=0.2, val_size=0.1
)
```

### Cross-validation with Cluster Awareness

```python
from mmseqspy.cluster_utils import cluster_kfold

folds = cluster_kfold(
    df, sequence_col="sequence", n_splits=5, random_state=42
)

for i, (train_df, test_df) in enumerate(folds):
    print(f"Fold {i+1}: {len(train_df)} train, {len(test_df)} test")
```

### Constrained Splits

```python
from mmseqspy.cluster_utils import constrained_split

train_df, test_df = constrained_split(
    clustered_df,
    group_col="representative_sequence",
    force_train_ids=["seq1", "seq2"],
    force_test_ids=["seq3"],
    id_type="sequence"
)
```

## Notes

- Ensure MMseqs2 is installed and accessible via the command line before using the library.
- Temporary files are automatically cleaned up after processing.
- The library provides detailed logging that can be controlled with `set_verbosity()`.
- Set reproducible random seeds with `set_seed()`.
