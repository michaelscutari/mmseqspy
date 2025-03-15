"""Baseline protein sequence embedders."""

import numpy as np
from typing import Dict, List, Optional

# Import matrix data
from .matrices import BLOSUM62, BLOSUM90, PROPERTY_SCALES

# Register embedders at module level
from .api import register_embedder


class BaseEmbedder:
    """Base class for all sequence embedders."""

    # Class property to indicate default pooling strategy
    default_pooling = "none"

    def generate(
        self, sequence: str, pooling: str = "auto", max_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate embeddings for a sequence.

        Args:
            sequence: Amino acid sequence
            pooling: Pooling method ('none', 'mean', 'max', 'sum', 'auto')
            max_length: Maximum sequence length to consider

        Returns:
            Embedding array
        """
        raise NotImplementedError("Embedder must implement generate method")

    def _apply_pooling(self, embedding: np.ndarray, method: str) -> np.ndarray:
        """Apply pooling to convert per-residue to sequence-level."""
        if method == "auto":
            method = self.default_pooling

        if method == "none" or len(embedding.shape) == 1:
            return embedding

        if method == "mean":
            return np.mean(embedding, axis=0)
        elif method == "max":
            return np.max(embedding, axis=0)
        elif method == "sum":
            return np.sum(embedding, axis=0)
        else:
            raise ValueError(f"Unknown pooling method: {method}")

    def _apply_length_constraint(self, sequence: str, max_length: Optional[int]) -> str:
        """Apply length constraint to sequence."""
        if max_length is None:
            return sequence

        if len(sequence) > max_length:
            return sequence[:max_length]  # Truncate
        return sequence  # No padding needed at sequence level


class BLOSUMEmbedder(BaseEmbedder):
    """BLOSUM matrix-based sequence embeddings."""

    default_pooling = "none"  # Default to per-residue

    def __init__(self, matrix_type: str = "BLOSUM62"):
        """
        Initialize with specified BLOSUM matrix.

        Args:
            matrix_type: Type of BLOSUM matrix (BLOSUM62 or BLOSUM90)
        """
        self.matrix_type = matrix_type
        self.matrix = self._load_matrix(matrix_type)

    def generate(
        self, sequence: str, pooling: str = "auto", max_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate BLOSUM embedding for sequence.

        Args:
            sequence: Amino acid sequence
            pooling: Pooling method
            max_length: Maximum sequence length

        Returns:
            Embedding array of shape (seq_len, 20) or (20,) depending on pooling
        """
        # Apply length constraint
        sequence = self._apply_length_constraint(sequence, max_length)

        # Handle empty sequences - return empty 2D array with correct shape
        if not sequence:
            return np.empty((0, 20))

        # Generate per-residue embeddings
        embedding = np.array([self.matrix.get(aa, self.matrix["X"]) for aa in sequence])

        # Apply pooling if requested
        return self._apply_pooling(embedding, pooling)

    def _load_matrix(self, matrix_type: str) -> Dict[str, np.ndarray]:
        """
        Load the specified BLOSUM matrix.

        Args:
            matrix_type: BLOSUM matrix type

        Returns:
            Dictionary mapping amino acids to embedding vectors
        """
        if matrix_type == "BLOSUM62":
            return BLOSUM62
        elif matrix_type == "BLOSUM90":
            return BLOSUM90
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")


class BLOSUM90Embedder(BLOSUMEmbedder):
    """BLOSUM90 matrix-based sequence embeddings."""

    def __init__(self):
        """Initialize with BLOSUM90 matrix."""
        super().__init__(matrix_type="BLOSUM90")


class AACompositionEmbedder(BaseEmbedder):
    """Amino acid composition embeddings."""

    default_pooling = "mean"  # Already sequence-level by nature

    def __init__(self, k: int = 1):
        """
        Initialize with k-mer size.

        Args:
            k: Size of k-mer (1 for amino acids, 2 for dipeptides, etc.)
        """
        self.k = k
        self.aa_list = list("ACDEFGHIKLMNPQRSTVWY")

    def generate(
        self, sequence: str, pooling: str = "auto", max_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate AAC embedding for sequence.

        Args:
            sequence: Amino acid sequence
            pooling: Pooling method (ignored, always produces sequence-level)
            max_length: Maximum sequence length

        Returns:
            Embedding array of shape (20^k,) for k-mer composition
        """
        # Apply length constraint
        sequence = self._apply_length_constraint(sequence, max_length)

        # For k=1, count individual amino acids
        if self.k == 1:
            counts = np.zeros(20)
            for i, aa in enumerate(self.aa_list):
                counts[i] = sequence.count(aa) / max(1, len(sequence))
            return counts

        # For k>1, count k-mers
        if self.k == 2:
            # Dipeptide composition (20×20 = 400 features)
            counts = np.zeros(400)
            for i, aa1 in enumerate(self.aa_list):
                for j, aa2 in enumerate(self.aa_list):
                    dipeptide = aa1 + aa2
                    idx = i * 20 + j
                    counts[idx] = sequence.count(dipeptide) / max(1, len(sequence) - 1)
            return counts

        # For k=3, calculate tripeptide composition
        if self.k == 3:
            # Tripeptide composition (20×20×20 = 8000 features)
            # This would be very high-dimensional, so we'll simplify by focusing
            # on common tripeptides only
            # Implementation details...
            pass

        raise ValueError(f"k value {self.k} not implemented")


class DiAACompositionEmbedder(AACompositionEmbedder):
    """Dipeptide composition embeddings."""

    def __init__(self):
        """Initialize with k=2 for dipeptides."""
        super().__init__(k=2)


class PropertyEmbedder(BaseEmbedder):
    """Physicochemical property-based embeddings."""

    default_pooling = "none"  # Default to per-residue

    def __init__(self, properties: List[str] = None):
        """
        Initialize with specified properties.

        Args:
            properties: List of properties to include, or None for all available
        """
        # Default properties if none specified
        self.properties = properties or [
            "hydrophobicity",
            "charge",
            "volume",
            "polarity",
        ]

        # Load property scales
        self.scales = self._load_scales()

    def generate(
        self, sequence: str, pooling: str = "auto", max_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate physicochemical property embedding for sequence.

        Args:
            sequence: Amino acid sequence
            pooling: Pooling method
            max_length: Maximum sequence length

        Returns:
            Embedding array of shape (seq_len, n_properties) or (n_properties,)
        """
        # Apply length constraint
        sequence = self._apply_length_constraint(sequence, max_length)

        # Generate per-residue embeddings
        embedding = np.array(
            [
                [self.scales[prop].get(aa, 0) for prop in self.properties]
                for aa in sequence
            ]
        )

        # Apply pooling if requested
        return self._apply_pooling(embedding, pooling)

    def _load_scales(self) -> Dict[str, Dict[str, float]]:
        """
        Load amino acid property scales from matrices module.

        Returns:
            Dictionary mapping property names to AA-value dictionaries
        """
        # Validate requested properties exist
        for prop in self.properties:
            if prop not in PROPERTY_SCALES:
                raise ValueError(
                    f"Unknown property: {prop}. Available properties: {list(PROPERTY_SCALES.keys())}"
                )

        # Return only requested properties
        return {prop: PROPERTY_SCALES[prop] for prop in self.properties}


class OneHotEmbedder(BaseEmbedder):
    """One-hot encoding of amino acid sequences."""

    default_pooling = "none"  # Default to per-residue

    def __init__(self):
        """Initialize the embedder."""
        self.aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        self.aa_dict = {aa: i for i, aa in enumerate(self.aa_list)}

    def generate(
        self, sequence: str, pooling: str = "auto", max_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate one-hot encoding for sequence.

        Args:
            sequence: Amino acid sequence
            pooling: Pooling method
            max_length: Maximum sequence length

        Returns:
            Embedding array of shape (seq_len, 20) or (20,)
        """
        # Apply length constraint
        sequence = self._apply_length_constraint(sequence, max_length)

        # Handle empty sequences
        if not sequence:
            return np.empty((0, len(self.aa_list)))

        # Generate one-hot encoding
        embedding = np.zeros((len(sequence), len(self.aa_list)))
        for i, aa in enumerate(sequence):
            if aa in self.aa_dict:
                embedding[i, self.aa_dict[aa]] = 1
            # If unknown amino acid, leave as all zeros

        # Apply pooling if requested
        return self._apply_pooling(embedding, pooling)


# Register embedders using proper subclasses
register_embedder("blosum62", BLOSUMEmbedder)
register_embedder("blosum90", BLOSUM90Embedder)
register_embedder("aac", AACompositionEmbedder)
register_embedder("di-aac", DiAACompositionEmbedder)
register_embedder("property", PropertyEmbedder)
register_embedder("onehot", OneHotEmbedder)
