"""
Positional encoding system for protein sequences.

This module implements a positional encoding approach where each amino acid
gets a vector representation based on both its identity and position in the sequence.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import math


class ProteinPositionalEncoder:
    """
    Positional encoder for protein sequences.

    Each amino acid is represented in an n-dimensional space with:
    - One axis for the amino acid type (20 standard amino acids)
    - One axis for the position in the sequence
    - Additional dimensions for enhanced representation
    """

    def __init__(self, embedding_dim: int = 64, max_sequence_length: int = 2000):
        """
        Initialize the positional encoder.

        Args:
            embedding_dim: Dimension of the embedding space
            max_sequence_length: Maximum sequence length to support
        """
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length

        # Standard amino acids
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        self.vocab_size = len(self.amino_acids)

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        # Extended vocabulary with special tokens
        self.extended_vocab = list(self.amino_acids) + [self.pad_token, self.unk_token]
        self.extended_vocab_size = len(self.extended_vocab)

        # Create embedding matrices
        self.aa_embedding = nn.Embedding(self.extended_vocab_size, embedding_dim // 2)
        self.pos_embedding = nn.Embedding(max_sequence_length, embedding_dim // 2)

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embedding weights."""
        # Xavier initialization for amino acid embeddings
        nn.init.xavier_uniform_(self.aa_embedding.weight)

        # Sinusoidal positional embeddings (like in Transformer)
        pos_embed_dim = self.embedding_dim // 2
        position = torch.arange(0, self.max_sequence_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, pos_embed_dim, 2).float()
            * -(math.log(10000.0) / pos_embed_dim)
        )

        pos_embeddings = torch.zeros(self.max_sequence_length, pos_embed_dim)
        pos_embeddings[:, 0::2] = torch.sin(position * div_term)
        pos_embeddings[:, 1::2] = torch.cos(position * div_term)

        self.pos_embedding.weight.data = pos_embeddings
        self.pos_embedding.weight.requires_grad = False  # Fixed positional embeddings

    def sequence_to_indices(self, sequence: str) -> List[int]:
        """Convert amino acid sequence to indices."""
        indices = []
        for aa in sequence.upper():
            if aa in self.aa_to_idx:
                indices.append(self.aa_to_idx[aa])
            else:
                # Unknown amino acid
                indices.append(self.vocab_size + 1)  # UNK token index
        return indices

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        Encode a single protein sequence with positional information.

        Args:
            sequence: Protein sequence string

        Returns:
            Tensor of shape (seq_len, embedding_dim)
        """
        # Convert to indices
        aa_indices = self.sequence_to_indices(sequence)
        seq_len = len(aa_indices)

        if seq_len > self.max_sequence_length:
            # Truncate if too long
            aa_indices = aa_indices[: self.max_sequence_length]
            seq_len = self.max_sequence_length

        # Create tensors
        aa_tensor = torch.tensor(aa_indices, dtype=torch.long)
        pos_tensor = torch.arange(seq_len, dtype=torch.long)

        # Get embeddings
        aa_embed = self.aa_embedding(aa_tensor)  # (seq_len, embed_dim//2)
        pos_embed = self.pos_embedding(pos_tensor)  # (seq_len, embed_dim//2)

        # Concatenate amino acid and positional embeddings
        combined_embed = torch.cat([aa_embed, pos_embed], dim=1)  # (seq_len, embed_dim)

        return combined_embed

    def encode_batch(
        self, sequences: List[str], pad_to_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode a batch of protein sequences with padding.

        Args:
            sequences: List of protein sequences
            pad_to_length: Length to pad sequences to (if None, use max length in batch)

        Returns:
            Tensor of shape (batch_size, max_seq_len, embedding_dim)
        """
        batch_size = len(sequences)

        # Determine padding length
        if pad_to_length is None:
            max_len = min(max(len(seq) for seq in sequences), self.max_sequence_length)
        else:
            max_len = min(pad_to_length, self.max_sequence_length)

        # Initialize batch tensor
        batch_tensor = torch.zeros(batch_size, max_len, self.embedding_dim)
        lengths = []

        for i, sequence in enumerate(sequences):
            # Encode individual sequence
            encoded = self.encode_sequence(sequence)
            seq_len = min(encoded.size(0), max_len)

            # Place in batch tensor
            batch_tensor[i, :seq_len] = encoded[:seq_len]
            lengths.append(seq_len)

        return batch_tensor, torch.tensor(lengths)

    def create_2d_representation(
        self, sequence: str, grid_size: int = 32
    ) -> torch.Tensor:
        """
        Create a 2D grid representation of the sequence for CNN processing.

        Args:
            sequence: Protein sequence
            grid_size: Size of the square grid (grid_size x grid_size)

        Returns:
            Tensor of shape (embedding_dim, grid_size, grid_size)
        """
        # Get 1D encoding
        encoded_1d = self.encode_sequence(sequence)  # (seq_len, embedding_dim)
        seq_len, embed_dim = encoded_1d.shape

        # Calculate how to arrange sequence in 2D grid
        total_positions = grid_size * grid_size

        if seq_len <= total_positions:
            # Pad sequence to fill grid
            padding_needed = total_positions - seq_len
            padding = torch.zeros(padding_needed, embed_dim)
            padded_sequence = torch.cat([encoded_1d, padding], dim=0)
        else:
            # Truncate sequence to fit grid
            padded_sequence = encoded_1d[:total_positions]

        # Reshape to 2D grid: (grid_size, grid_size, embedding_dim)
        grid_2d = padded_sequence.view(grid_size, grid_size, embed_dim)

        # Transpose to (embedding_dim, grid_size, grid_size) for CNN
        grid_2d = grid_2d.permute(2, 0, 1)

        return grid_2d

    def create_batch_2d_representation(
        self, sequences: List[str], grid_size: int = 32
    ) -> torch.Tensor:
        """
        Create batch of 2D representations for CNN processing.

        Args:
            sequences: List of protein sequences
            grid_size: Size of the square grid

        Returns:
            Tensor of shape (batch_size, embedding_dim, grid_size, grid_size)
        """
        batch_size = len(sequences)
        batch_tensor = torch.zeros(batch_size, self.embedding_dim, grid_size, grid_size)

        for i, sequence in enumerate(sequences):
            batch_tensor[i] = self.create_2d_representation(sequence, grid_size)

        return batch_tensor

    def get_sequence_statistics(self, sequences: List[str]) -> Dict:
        """Get statistics about sequence lengths and amino acid usage."""
        lengths = [len(seq) for seq in sequences]
        aa_counts = {aa: 0 for aa in self.amino_acids}
        unknown_count = 0

        for seq in sequences:
            for aa in seq.upper():
                if aa in aa_counts:
                    aa_counts[aa] += 1
                else:
                    unknown_count += 1

        return {
            "num_sequences": len(sequences),
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "amino_acid_counts": aa_counts,
            "unknown_amino_acids": unknown_count,
            "total_amino_acids": sum(aa_counts.values()) + unknown_count,
        }


def create_positional_encoder(
    embedding_dim: int = 64, max_length: int = 2000
) -> ProteinPositionalEncoder:
    """Factory function to create a positional encoder."""
    return ProteinPositionalEncoder(embedding_dim, max_length)
