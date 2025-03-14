# setup.py
from setuptools import find_packages, setup

setup(
    name="mmseqspy",
    version="0.3.0",
    description="Python utilities for protein sequence clustering and dataset splitting with MMseqs2",
    long_description="""
    mmseqspy provides utilities for clustering protein sequences and creating train-test splits
    that respect sequence similarity. It requires MMseqs2 to be installed and in your PATH.
    Features include sequence clustering, cluster-aware train/test splits, k-fold cross-validation,
    and constrained dataset splitting.
    """,
    author="Michael Scutari",
    author_email="michael.scutari@duke.edu",
    url="https://github.com/michaelscutari/mmseqspy",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
    ],
    keywords="bioinformatics, protein, sequence, clustering, mmseqs2",
)
