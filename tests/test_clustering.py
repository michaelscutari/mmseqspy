import pytest
import pandas as pd
from mmseqspy import cluster

def test_cluster_sequences(fluorescence_data, mmseqs_installed):
    """Test clustering protein sequences."""
    # Make a copy of data
    df = fluorescence_data.copy()
    
    # Run clustering
    clustered_df = cluster(
        df, 
        sequence_col='sequence', 
        id_col='id',
        min_seq_id=0.5,  # 50% sequence identity threshold
        coverage=0.8     # 80% coverage
    )
    
    # Check results
    assert 'representative_sequence' in clustered_df.columns
    assert len(clustered_df) == len(df)  # Should preserve all rows
    
    # Count unique clusters
    n_clusters = clustered_df['representative_sequence'].nunique()
    
    # Basic sanity check - clusters should be fewer than sequences
    assert 1 <= n_clusters <= len(df)
    
    # Check that all representative_sequence values exist in the id column
    assert set(clustered_df['representative_sequence']).issubset(set(clustered_df['id']))