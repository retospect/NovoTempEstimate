"""
Peptide sequence encoder with one-hot vector representation.

This module provides functionality to encode peptide sequences as one-hot vectors,
with support for standard amino acids and various modifications.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Set
import re
from dataclasses import dataclass
from enum import Enum


class ModificationType(Enum):
    """Types of peptide modifications."""

    PHOSPHORYLATION = "phosphorylation"
    METHYLATION = "methylation"
    ACETYLATION = "acetylation"
    UBIQUITINATION = "ubiquitination"
    OXIDATION = "oxidation"
    DEAMIDATION = "deamidation"
    CITRULLINATION = "citrullination"
    NITROSYLATION = "nitrosylation"
    HYDROXYLATION = "hydroxylation"
    CUSTOM = "custom"


@dataclass
class ModifiedResidue:
    """Represents a modified amino acid residue."""

    base_aa: str
    modification: str
    modification_type: ModificationType = ModificationType.CUSTOM
    mass_shift: Optional[float] = None

    def __str__(self):
        return f"{self.base_aa}[{self.modification}]"

    def __hash__(self):
        return hash((self.base_aa, self.modification))


class PeptideEncoder:
    """
    One-hot encoder for peptide sequences with modification support.

    This encoder handles:
    - Standard 20 amino acids
    - Common modifications (phosphorylation, methylation, etc.)
    - Custom modifications
    - Unknown residues
    - Variable sequence lengths
    """

    def __init__(self, include_modifications: bool = True, max_custom_mods: int = 50):
        """
        Initialize the peptide encoder.

        Args:
            include_modifications: Whether to support modifications
            max_custom_mods: Maximum number of custom modifications to support
        """
        self.include_modifications = include_modifications
        self.max_custom_mods = max_custom_mods

        # Standard amino acids
        self.standard_aa = set("ACDEFGHIKLMNPQRSTVWY")

        # Common modifications mapping
        self.common_modifications = {
            # Phosphorylation
            "pS": ModifiedResidue(
                "S", "phospho", ModificationType.PHOSPHORYLATION, 79.966331
            ),
            "pT": ModifiedResidue(
                "T", "phospho", ModificationType.PHOSPHORYLATION, 79.966331
            ),
            "pY": ModifiedResidue(
                "Y", "phospho", ModificationType.PHOSPHORYLATION, 79.966331
            ),
            # Methylation
            "mK": ModifiedResidue(
                "K", "methyl", ModificationType.METHYLATION, 14.015650
            ),
            "mR": ModifiedResidue(
                "R", "methyl", ModificationType.METHYLATION, 14.015650
            ),
            "me2K": ModifiedResidue(
                "K", "dimethyl", ModificationType.METHYLATION, 28.031300
            ),
            "me3K": ModifiedResidue(
                "K", "trimethyl", ModificationType.METHYLATION, 42.046950
            ),
            # Acetylation
            "acK": ModifiedResidue(
                "K", "acetyl", ModificationType.ACETYLATION, 42.010565
            ),
            # Oxidation
            "oxM": ModifiedResidue(
                "M", "oxidation", ModificationType.OXIDATION, 15.994915
            ),
            "oxW": ModifiedResidue(
                "W", "oxidation", ModificationType.OXIDATION, 15.994915
            ),
            # Deamidation
            "dN": ModifiedResidue(
                "N", "deamidation", ModificationType.DEAMIDATION, 0.984016
            ),
            "dQ": ModifiedResidue(
                "Q", "deamidation", ModificationType.DEAMIDATION, 0.984016
            ),
            # Citrullination
            "citR": ModifiedResidue(
                "R", "citrullination", ModificationType.CITRULLINATION, 0.984016
            ),
        }

        # Build vocabulary
        self._build_vocabulary()

    def _build_vocabulary(self):
        """Build the complete vocabulary including modifications."""
        self.vocabulary = {}
        self.reverse_vocabulary = {}

        # Add standard amino acids
        for i, aa in enumerate(sorted(self.standard_aa)):
            self.vocabulary[aa] = i
            self.reverse_vocabulary[i] = aa

        current_idx = len(self.standard_aa)

        if self.include_modifications:
            # Add common modifications
            for mod_key, mod_residue in self.common_modifications.items():
                self.vocabulary[str(mod_residue)] = current_idx
                self.reverse_vocabulary[current_idx] = str(mod_residue)
                current_idx += 1

            # Reserve space for custom modifications
            for i in range(self.max_custom_mods):
                custom_key = f"CUSTOM_{i}"
                self.vocabulary[custom_key] = current_idx
                self.reverse_vocabulary[current_idx] = custom_key
                current_idx += 1

        # Add unknown residue
        self.vocabulary["X"] = current_idx
        self.reverse_vocabulary[current_idx] = "X"
        self.unknown_idx = current_idx
        current_idx += 1

        # Add padding token
        self.vocabulary["<PAD>"] = current_idx
        self.reverse_vocabulary[current_idx] = "<PAD>"
        self.pad_idx = current_idx

        self.vocab_size = len(self.vocabulary)
        self.custom_modifications = {}  # Track custom modifications

    def parse_sequence(self, sequence: str) -> List[str]:
        """
        Parse a peptide sequence into individual residues/modifications.

        Supports formats:
        - Standard: "PEPTIDE"
        - Bracketed modifications: "PEP[phospho]TIDE"
        - Short notation: "pSpTIDE"
        - Mixed: "PEP[custom_mod]pSIDE"

        Args:
            sequence: Peptide sequence string

        Returns:
            List of residue strings
        """
        residues = []
        i = 0

        while i < len(sequence):
            if (
                self.include_modifications
                and i < len(sequence) - 1
                and sequence[i : i + 2] in self.common_modifications
            ):
                # Handle short notation modifications (e.g., pS, mK)
                residues.append(str(self.common_modifications[sequence[i : i + 2]]))
                i += 2
            elif (
                self.include_modifications
                and i < len(sequence) - 2
                and sequence[i : i + 3] in self.common_modifications
            ):
                # Handle 3-character modifications (e.g., me2K, me3K)
                residues.append(str(self.common_modifications[sequence[i : i + 3]]))
                i += 3
            elif (
                self.include_modifications
                and i < len(sequence) - 3
                and sequence[i : i + 4] in self.common_modifications
            ):
                # Handle 4-character modifications (e.g., citR)
                residues.append(str(self.common_modifications[sequence[i : i + 4]]))
                i += 4
            elif sequence[i] in self.standard_aa:
                # Check for bracketed modification
                if i + 1 < len(sequence) and sequence[i + 1] == "[":
                    # Find closing bracket
                    close_bracket = sequence.find("]", i + 2)
                    if close_bracket != -1:
                        base_aa = sequence[i]
                        modification = sequence[i + 2 : close_bracket]
                        mod_residue = ModifiedResidue(base_aa, modification)
                        residues.append(str(mod_residue))
                        i = close_bracket + 1
                    else:
                        # Malformed bracket, treat as standard AA
                        residues.append(sequence[i])
                        i += 1
                else:
                    # Standard amino acid
                    residues.append(sequence[i])
                    i += 1
            else:
                # Unknown character, treat as X
                residues.append("X")
                i += 1

        return residues

    def _get_residue_index(self, residue: str) -> int:
        """Get vocabulary index for a residue, handling custom modifications."""
        if residue in self.vocabulary:
            return self.vocabulary[residue]

        # Check if it's a custom modification
        if "[" in residue and "]" in residue:
            if self.include_modifications:
                if residue not in self.custom_modifications:
                    # Assign a custom modification slot
                    custom_slots_used = len(self.custom_modifications)
                    if custom_slots_used < self.max_custom_mods:
                        custom_key = f"CUSTOM_{custom_slots_used}"
                        self.custom_modifications[residue] = custom_key
                        return self.vocabulary[custom_key]
                    else:
                        # No more custom slots, treat as unknown
                        return self.unknown_idx
                else:
                    return self.vocabulary[self.custom_modifications[residue]]
            else:
                # Modifications not supported, treat as unknown
                return self.unknown_idx

        # Unknown residue
        return self.unknown_idx

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        Encode a single peptide sequence as one-hot vectors.

        Args:
            sequence: Peptide sequence string

        Returns:
            Tensor of shape (sequence_length, vocab_size)
        """
        residues = self.parse_sequence(sequence)

        # Create one-hot encoding
        encoding = torch.zeros(len(residues), self.vocab_size)

        for i, residue in enumerate(residues):
            idx = self._get_residue_index(residue)
            encoding[i, idx] = 1.0

        return encoding

    def encode_batch(
        self, sequences: List[str], pad_to_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode a batch of peptide sequences with padding.

        Args:
            sequences: List of peptide sequence strings
            pad_to_length: Length to pad sequences to (uses max length if None)

        Returns:
            Tensor of shape (batch_size, max_length, vocab_size)
        """
        if not sequences:
            return torch.empty(0, 0, self.vocab_size)

        # Encode all sequences
        encoded_sequences = [self.encode_sequence(seq) for seq in sequences]

        # Determine padding length
        if pad_to_length is None:
            pad_to_length = max(seq.shape[0] for seq in encoded_sequences)

        # Pad sequences
        batch_encoding = torch.zeros(len(sequences), pad_to_length, self.vocab_size)

        for i, encoded_seq in enumerate(encoded_sequences):
            seq_len = min(encoded_seq.shape[0], pad_to_length)
            batch_encoding[i, :seq_len] = encoded_seq[:seq_len]

            # Add padding tokens for remaining positions
            if seq_len < pad_to_length:
                batch_encoding[i, seq_len:, self.pad_idx] = 1.0

        return batch_encoding

    def decode_sequence(self, encoding: torch.Tensor) -> str:
        """
        Decode one-hot encoded sequence back to string.

        Args:
            encoding: Tensor of shape (sequence_length, vocab_size)

        Returns:
            Decoded peptide sequence string
        """
        indices = torch.argmax(encoding, dim=1)
        residues = []

        for idx in indices:
            idx_val = idx.item()
            if idx_val == self.pad_idx:
                break  # Stop at padding

            residue = self.reverse_vocabulary.get(idx_val, "X")

            # Handle custom modifications
            if residue.startswith("CUSTOM_"):
                # Find the actual custom modification
                for custom_mod, custom_key in self.custom_modifications.items():
                    if custom_key == residue:
                        residue = custom_mod
                        break
                else:
                    residue = "X"  # Fallback

            residues.append(residue)

        return "".join(residues)

    def get_modification_info(self, sequence: str) -> Dict:
        """
        Get detailed information about modifications in a sequence.

        Args:
            sequence: Peptide sequence string

        Returns:
            Dictionary with modification information
        """
        residues = self.parse_sequence(sequence)

        info = {
            "total_residues": len(residues),
            "modified_residues": 0,
            "modifications": {},
            "modification_types": set(),
            "unknown_residues": 0,
        }

        for residue in residues:
            if "[" in residue and "]" in residue:
                info["modified_residues"] += 1
                base_aa = residue.split("[")[0]
                modification = residue.split("[")[1].rstrip("]")

                if modification not in info["modifications"]:
                    info["modifications"][modification] = []
                info["modifications"][modification].append(base_aa)

                # Determine modification type
                for mod_residue in self.common_modifications.values():
                    if str(mod_residue) == residue:
                        info["modification_types"].add(
                            mod_residue.modification_type.value
                        )
                        break
                else:
                    info["modification_types"].add("custom")

            elif residue == "X":
                info["unknown_residues"] += 1

        info["modification_types"] = list(info["modification_types"])
        return info

    def get_vocabulary_info(self) -> Dict:
        """Get information about the encoder vocabulary."""
        return {
            "vocab_size": self.vocab_size,
            "standard_aa_count": len(self.standard_aa),
            "common_modifications_count": len(self.common_modifications),
            "custom_modifications_used": len(self.custom_modifications),
            "max_custom_mods": self.max_custom_mods,
            "includes_modifications": self.include_modifications,
            "vocabulary": dict(self.vocabulary),
        }

    def save_vocabulary(self, filepath: str):
        """Save vocabulary and custom modifications to file."""
        import json

        data = {
            "vocabulary": self.vocabulary,
            "reverse_vocabulary": {
                str(k): v for k, v in self.reverse_vocabulary.items()
            },
            "custom_modifications": self.custom_modifications,
            "vocab_size": self.vocab_size,
            "include_modifications": self.include_modifications,
            "max_custom_mods": self.max_custom_mods,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_vocabulary(cls, filepath: str) -> "PeptideEncoder":
        """Load vocabulary and create encoder from file."""
        import json

        with open(filepath, "r") as f:
            data = json.load(f)

        encoder = cls(
            include_modifications=data["include_modifications"],
            max_custom_mods=data["max_custom_mods"],
        )

        encoder.vocabulary = data["vocabulary"]
        encoder.reverse_vocabulary = {
            int(k): v for k, v in data["reverse_vocabulary"].items()
        }
        encoder.custom_modifications = data["custom_modifications"]
        encoder.vocab_size = data["vocab_size"]

        return encoder


# Convenience functions
def encode_peptide(sequence: str, include_modifications: bool = True) -> torch.Tensor:
    """Quick function to encode a single peptide sequence."""
    encoder = PeptideEncoder(include_modifications=include_modifications)
    return encoder.encode_sequence(sequence)


def encode_peptide_batch(
    sequences: List[str], include_modifications: bool = True
) -> torch.Tensor:
    """Quick function to encode a batch of peptide sequences."""
    encoder = PeptideEncoder(include_modifications=include_modifications)
    return encoder.encode_batch(sequences)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    encoder = PeptideEncoder()

    # Test sequences with various modifications
    test_sequences = [
        "PEPTIDE",  # Standard sequence
        "PEP[phospho]TIDE",  # Bracketed modification
        "pSpTIDE",  # Short notation
        "PEP[custom_mod]TIDE",  # Custom modification
        "me3KPEPTIDE",  # Trimethyl lysine
        "PEPTIDEoxM",  # Oxidized methionine
        "PEPTIDEX",  # Unknown residue
    ]

    print("Testing PeptideEncoder:")
    print(f"Vocabulary size: {encoder.vocab_size}")

    for seq in test_sequences:
        print(f"\nSequence: {seq}")

        # Parse sequence
        residues = encoder.parse_sequence(seq)
        print(f"Parsed residues: {residues}")

        # Encode
        encoded = encoder.encode_sequence(seq)
        print(f"Encoded shape: {encoded.shape}")

        # Decode
        decoded = encoder.decode_sequence(encoded)
        print(f"Decoded: {decoded}")

        # Get modification info
        mod_info = encoder.get_modification_info(seq)
        print(f"Modifications: {mod_info}")

    # Test batch encoding
    print(f"\nBatch encoding {len(test_sequences)} sequences:")
    batch_encoded = encoder.encode_batch(test_sequences)
    print(f"Batch shape: {batch_encoded.shape}")

    # Vocabulary info
    vocab_info = encoder.get_vocabulary_info()
    print(f"\nVocabulary info: {vocab_info}")
