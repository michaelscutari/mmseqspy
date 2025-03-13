import os
import tempfile
import pandas as pd
import subprocess
import shutil
import logging

from .logger import logger
from .utils import _check_mmseqs, _validate_clustering_params

def cluster(
    df,
    sequence_col,
    id_col=None,
    min_seq_id=0.3,
    coverage=0.5,
    cov_mode=0,
    alignment_mode=0,
    cluster_mode=0,
    cluster_steps=1,
):
    """
    Clusters sequences with MMseqs2 and adds a 'representative_sequence' column.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns for IDs and sequences.
        sequence_col (str): Name of the column containing sequences.
        id_col (str): Unique ID column (default "id").
        min_seq_id (float): Minimum sequence identity for clustering (0-1, default 0.3).
        coverage (float): Minimum alignment coverage (0-1, default 0.5).
        cov_mode (int): Coverage mode for MMseqs2 (0-2, default 0):
            0: coverage of query and target
            1: coverage of target
            2: coverage of query
        alignment_mode (int): Alignment mode for MMseqs2 (0-4, default 0):
            0: automatic
            1: only score
            2: only ungapped alignment
            3: score and end_pos
            4: ungapped alignment and end_pos
        cluster_mode (int): Cluster mode for MMseqs2 (0-2, default 0):
            0: greedy set cover (default)
            1: connected component
            2: greedy incremental
        cluster_steps (int): Number of clustering steps (default 1)

    Returns:
        pd.DataFrame: Original DataFrame with a new 'representative_sequence' column.
    """
    logger.info("Starting sequence clustering with MMseqs2")
    logger.info(
        f"Parameters: min_seq_id={min_seq_id}, coverage={coverage}, cov_mode={cov_mode}, "
        f"alignment_mode={alignment_mode}, cluster_mode={cluster_mode}, cluster_steps={cluster_steps}"
    )

    _check_mmseqs()
    _validate_clustering_params(min_seq_id, coverage, cov_mode, alignment_mode, cluster_mode, cluster_steps)

    if id_col is None:
        df = df.reset_index()
        id_col = "index"
        logger.debug(f"No id_col provided, using '{id_col}' as identifier")

    if sequence_col not in df or id_col not in df:
        logger.error(f"Required columns missing: {sequence_col} or {id_col}")
        raise ValueError(f"The DataFrame must have '{id_col}' and '{sequence_col}'.")

    logger.info(f"Clustering {len(df)} sequences")

    df["sanitized_id"] = df[id_col].str.replace(" ", "_")
    tmp_dir = tempfile.mkdtemp()
    logger.debug(f"Created temporary directory: {tmp_dir}")

    try:
        input_fasta = os.path.join(tmp_dir, "input.fasta")
        with open(input_fasta, "w") as fasta_file:
            for _, row in df.iterrows():
                fasta_file.write(f">{row['sanitized_id']}\n{row[sequence_col]}\n")

        logger.debug(f"Wrote {len(df)} sequences to FASTA file")

        output_dir = os.path.join(tmp_dir, "output")
        tmp_mmseqs = os.path.join(tmp_dir, "tmp_mmseqs")

        mmseqs_cmd = [
            "mmseqs",
            "easy-cluster",
            input_fasta,
            output_dir,
            tmp_mmseqs,
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            str(cov_mode),
            "--alignment-mode",
            str(alignment_mode),
            "--cluster-mode",
            str(cluster_mode),
            "--cluster-steps",
            str(cluster_steps),
        ]

        logger.debug(f"Running MMseqs2 command: {' '.join(mmseqs_cmd)}")

        if logger.level <= logging.DEBUG:
            subprocess.run(mmseqs_cmd, check=True)
        else:
            subprocess.run(
                mmseqs_cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        clusters_file = os.path.join(output_dir + "_cluster.tsv")
        if not os.path.exists(clusters_file):
            logger.error("MMseqs2 clustering results file not found")
            raise FileNotFoundError("MMseqs2 clustering results not found.")

        logger.debug(f"Reading clustering results from {clusters_file}")

        cluster_map = {}
        cluster_sizes = {}
        with open(clusters_file, "r") as f:
            for line in f:
                rep, seq = line.strip().split("\t")
                cluster_map[seq] = rep
                cluster_sizes[rep] = cluster_sizes.get(rep, 0) + 1

        logger.info(f"Found {len(cluster_sizes)} clusters")

        if logger.level <= logging.DEBUG:
            # Report cluster distribution statistics
            cluster_size_counts = {}
            for size in cluster_sizes.values():
                cluster_size_counts[size] = cluster_size_counts.get(size, 0) + 1

            logger.debug("Cluster size distribution:")
            for size in sorted(cluster_size_counts.keys()):
                logger.debug(f"  Size {size}: {cluster_size_counts[size]} clusters")

        reverse_map = dict(zip(df["sanitized_id"], df[id_col]))
        df["representative_sequence"] = df["sanitized_id"].apply(
            lambda x: reverse_map.get(cluster_map.get(x, x), x)
        )

        logger.info(
            "Clustering complete, added 'representative_sequence' column to DataFrame"
        )

    finally:
        logger.debug(f"Cleaning up temporary directory: {tmp_dir}")
        shutil.rmtree(tmp_dir)

    df.drop(columns=["sanitized_id"], inplace=True)
    return df