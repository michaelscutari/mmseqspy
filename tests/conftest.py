import os
import json
import pytest
import pandas as pd
import logging
from .test_utils import (
    create_cluster_dataset,
    create_identity_test_dataset,
    create_edge_case_dataset,
)

# Define the path to the JSON file - you can place it in this location
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TAPE_FLUORESCENCE_PATH = os.path.join(DATA_DIR, "fluorescence.json")


@pytest.fixture
def fluorescence_data():
    """
    Load the first 100 proteins from the TAPE fluorescence dataset.
    The dataset should be in JSON format with at least 'sequence' and 'fluorescence' fields.
    """
    if not os.path.exists(TAPE_FLUORESCENCE_PATH):
        pytest.skip(f"TAPE fluorescence data not found at {TAPE_FLUORESCENCE_PATH}")

    with open(TAPE_FLUORESCENCE_PATH, "r") as f:
        data = json.load(f)

    # Take only the first 400 proteins
    truncated_data = data[:400]

    # Convert to DataFrame
    df = pd.DataFrame(truncated_data)

    # Ensure required columns exist
    required_cols = ["sequence", "fluorescence"]
    for col in required_cols:
        if col not in df.columns:
            pytest.fail(f"Required column '{col}' not found in dataset")

    # Add a unique ID column if not present
    if "id" not in df.columns:
        df["id"] = [f"protein_{i}" for i in range(len(df))]

    return df


@pytest.fixture
def synthetic_cluster_data():
    """
    Generate synthetic data with well-defined clusters for testing.

    Returns:
        DataFrame with synthetic sequences organized in 5 clusters,
        with high within-cluster similarity and low between-cluster similarity.
    """
    return create_cluster_dataset(
        n_clusters=10,
        seqs_per_cluster=6,
        seq_length=100,
        within_identity=0.9,
        between_identity=0.3,
        seed=42,
    )


@pytest.fixture
def identity_test_data():
    """
    Generate synthetic data with sequences at specific identity levels.

    Returns:
        DataFrame with sequences at controlled identity levels to test
        clustering thresholds.
    """
    return create_identity_test_dataset(
        base_seq_length=100,
        identity_levels=[1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5],
        variants_per_level=3,
        seed=42,
    )


@pytest.fixture
def edge_case_data():
    """
    Generate synthetic data with edge cases for robustness testing.

    Returns:
        DataFrame with edge case sequences.
    """
    return create_edge_case_dataset()


@pytest.fixture
def mmseqs_installed():
    """Skip tests if MMseqs2 is not installed."""
    import shutil

    if shutil.which("mmseqs") is None:
        pytest.skip("MMseqs2 not installed, skipping tests that require it")
    return True


@pytest.fixture(autouse=True)
def test_logger():
    # Set up a dedicated logger for tests.
    test_log = logging.getLogger("mmseqspy")
    test_log.setLevel(logging.DEBUG)

    # Clear any existing handlers to avoid duplicates.
    if test_log.hasHandlers():
        test_log.handlers.clear()

    # Create and add a stream handler with a simple formatter.
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    test_log.addHandler(handler)

    yield test_log

    # Cleanup: clear handlers after tests run.
    test_log.handlers.clear()
