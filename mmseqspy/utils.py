import os
import random
import shutil
import subprocess
import numpy as np
import logging

from .logger import logger


# default seed
_DEFAULT_SEED = 42
_current_seed = _DEFAULT_SEED


def set_seed(seed=_DEFAULT_SEED):
    """
    Set random seeds for reproducibility.

    Parameters:
        seed (int): Random seed value (default 42)
    """
    global _current_seed
    if seed != _current_seed:
        logger.info(f"Setting random seed to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        _current_seed = seed


def set_verbosity(verbose=False):
    """
    Set the verbosity level for the package.

    Parameters:
        verbose (bool or int): If True or 1, sets to INFO level.
                               If 2, sets to DEBUG level.
                               If False or 0, sets to WARNING level.
    """
    if verbose == 2:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbosity set to DEBUG level")
    elif verbose:
        logger.setLevel(logging.INFO)
        logger.info("Verbosity set to INFO level")
    else:
        logger.setLevel(logging.WARNING)


def _check_mmseqs():
    """
    Ensures 'mmseqs' command is in PATH.
    """
    logger.debug("Checking if MMseqs2 is installed")
    if shutil.which("mmseqs") is None:
        logger.error("MMseqs2 not found in PATH")
        raise EnvironmentError(
            "MMseqs2 is not installed or not found in PATH. "
            "See the README for installation instructions."
        )
    logger.debug("MMseqs2 found in PATH")


def _validate_clustering_params(min_seq_id, coverage, cov_mode, alignment_mode, cluster_mode, cluster_steps):
    """
    Validates the clustering parameters are within acceptable ranges.
    
    Parameters:
        min_seq_id (float): Minimum sequence identity (0-1)
        coverage (float): Minimum coverage (0-1)
        cov_mode (int): Coverage mode (0-2)
        alignment_mode (int): Alignment mode (0-4)
        cluster_mode (int): Cluster mode (0-2)
        cluster_steps (int): Number of clustering steps (>0)
        
    Raises:
        ValueError: If any parameter is outside its valid range
    """
    if not 0 <= min_seq_id <= 1:
        raise ValueError(f"min_seq_id must be between 0 and 1, got {min_seq_id}")
    if not 0 <= coverage <= 1:
        raise ValueError(f"coverage must be between 0 and 1, got {coverage}")
    if cov_mode not in [0, 1, 2]:
        raise ValueError(f"cov_mode must be 0, 1, or 2, got {cov_mode}")
    if alignment_mode not in [0, 1, 2, 3, 4]:
        raise ValueError(f"alignment_mode must be 0, 1, 2, 3, or 4, got {alignment_mode}")
    if cluster_mode not in [0, 1, 2]:
        raise ValueError(f"cluster_mode must be 0, 1, or 2, got {cluster_mode}")
    if not isinstance(cluster_steps, int) or cluster_steps <= 0:
        raise ValueError(f"cluster_steps must be a positive integer, got {cluster_steps}")