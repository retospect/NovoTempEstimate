"""
Protein sequence encoding module.

This module provides functionality to encode protein sequences into numerical
representations suitable for machine learning models.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class ProteinEncoder:
    """
    A class for encoding protein sequences into various numerical representations.

    This encoder supports multiple encoding strategies:
    - One-hot encoding
    - Amino acid property-based encoding
    - Learned embeddings
    - Physicochemical property encoding
    """

    def __init__(self, encoding_type: str = "onehot", embedding_dim: int = 128):
        """
        Initialize the protein encoder.

        Args:
            encoding_type: Type of encoding ('onehot', 'properties', 'embedding')
            embedding_dim: Dimension for learned embeddings
        """
        self.encoding_type = encoding_type
        self.embedding_dim = embedding_dim

        # Standard amino acids
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        self.idx_to_aa = {idx: aa for aa, idx in self.aa_to_idx.items()}

        # Physicochemical properties (placeholder values)
        self.aa_properties = self._initialize_aa_properties()

        # Embedding layer for learned representations
        if encoding_type == "embedding":
            self.embedding = torch.nn.Embedding(len(self.amino_acids), embedding_dim)

    def _initialize_aa_properties(self) -> Dict[str, List[float]]:
        """
        Initialize amino acid physicochemical properties.

        Returns:
            Dictionary mapping amino acids to property vectors
        """
        # TODO: Implement actual physicochemical properties
        # For now, using placeholder random values
        properties = {}
        np.random.seed(42)  # For reproducibility

        for aa in self.amino_acids:
            # Placeholder: hydrophobicity, charge, size, polarity, aromaticity
            properties[aa] = np.random.randn(5).tolist()

        return properties

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        Encode a single protein sequence.

        Args:
            sequence: Protein sequence string

        Returns:
            Encoded sequence tensor
        """
        if self.encoding_type == "onehot":
            return self._onehot_encode(sequence)
        elif self.encoding_type == "properties":
            return self._property_encode(sequence)
        elif self.encoding_type == "embedding":
            return self._embedding_encode(sequence)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

    def encode_batch(self, sequences: List[str]) -> torch.Tensor:
        """
        Encode a batch of protein sequences.

        Args:
            sequences: List of protein sequence strings

        Returns:
            Batch of encoded sequences
        """
        encoded_sequences = []
        max_length = max(len(seq) for seq in sequences)

        for seq in sequences:
            encoded = self.encode_sequence(seq)
            # Pad sequence to max length
            if len(seq) < max_length:
                padding_size = max_length - len(seq)
                if self.encoding_type == "onehot":
                    padding = torch.zeros(padding_size, len(self.amino_acids))
                elif self.encoding_type == "properties":
                    padding = torch.zeros(padding_size, 5)
                elif self.encoding_type == "embedding":
                    padding = torch.zeros(padding_size, self.embedding_dim)
                encoded = torch.cat([encoded, padding], dim=0)

            encoded_sequences.append(encoded)

        return torch.stack(encoded_sequences)

    def _onehot_encode(self, sequence: str) -> torch.Tensor:
        """One-hot encode a protein sequence."""
        encoding = torch.zeros(len(sequence), len(self.amino_acids))

        for i, aa in enumerate(sequence.upper()):
            if aa in self.aa_to_idx:
                encoding[i, self.aa_to_idx[aa]] = 1.0

        return encoding

    def _property_encode(self, sequence: str) -> torch.Tensor:
        """Encode using physicochemical properties."""
        encoding = torch.zeros(len(sequence), 5)  # 5 properties

        for i, aa in enumerate(sequence.upper()):
            if aa in self.aa_properties:
                encoding[i] = torch.tensor(self.aa_properties[aa])

        return encoding

    def _embedding_encode(self, sequence: str) -> torch.Tensor:
        """Encode using learned embeddings."""
        indices = []
        for aa in sequence.upper():
            if aa in self.aa_to_idx:
                indices.append(self.aa_to_idx[aa])
            else:
                indices.append(0)  # Unknown amino acid

        indices_tensor = torch.tensor(indices, dtype=torch.long)
        return self.embedding(indices_tensor)

    def calculate_sequence_features(self, sequence: str) -> Dict[str, float]:
        """
        Calculate additional sequence-level features.

        Args:
            sequence: Protein sequence string

        Returns:
            Dictionary of calculated features
        """
        try:
            analysis = ProteinAnalysis(sequence)

            features = {
                "length": len(sequence),
                "molecular_weight": analysis.molecular_weight(),
                "aromaticity": analysis.aromaticity(),
                "instability_index": analysis.instability_index(),
                "isoelectric_point": analysis.isoelectric_point(),
                "gravy": analysis.gravy(),  # Grand average of hydropathy
            }

            # Secondary structure fractions
            sec_struct = analysis.secondary_structure_fraction()
            features.update(
                {
                    "helix_fraction": sec_struct[0],
                    "turn_fraction": sec_struct[1],
                    "sheet_fraction": sec_struct[2],
                }
            )

            return features

        except Exception as e:
            print(f"Error calculating features for sequence: {e}")
            return {"length": len(sequence)}

    def get_encoding_dim(self) -> int:
        """Get the dimension of the encoding."""
        if self.encoding_type == "onehot":
            return len(self.amino_acids)
        elif self.encoding_type == "properties":
            return 5
        elif self.encoding_type == "embedding":
            return self.embedding_dim
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")


# Example usage and testing functions
def test_encoder():
    """Test the protein encoder with sample sequences."""
    test_sequences = [
        "MKFLVLLFNILCLFPVLAADNHGVGPQGASGILKTLLKQIGDLQAGLQGVQAGVWPAAVRESVPSLL",
        "MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGD",
        "MKKLLFAIPLVVPFNYEAPVQVGVQNLKYEQPSLNELNSALQKGVQLLQGLQAGLQGVQAGVWPAAV",
    ]

    # Test different encoding types
    for encoding_type in ["onehot", "properties", "embedding"]:
        print(f"\nTesting {encoding_type} encoding:")
        encoder = ProteinEncoder(encoding_type=encoding_type)

        # Test single sequence
        encoded = encoder.encode_sequence(test_sequences[0])
        print(f"Single sequence shape: {encoded.shape}")

        # Test batch encoding
        batch_encoded = encoder.encode_batch(test_sequences)
        print(f"Batch shape: {batch_encoded.shape}")

        # Test features
        features = encoder.calculate_sequence_features(test_sequences[0])
        print(f"Features: {list(features.keys())}")


if __name__ == "__main__":
    test_encoder()
