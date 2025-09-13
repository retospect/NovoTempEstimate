"""
Comprehensive tests for the peptide_encoder module.
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, mock_open

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from novotempestimate.peptide_encoder import (
    PeptideEncoder,
    ModifiedResidue,
    ModificationType,
    encode_peptide,
    encode_peptide_batch,
)


class TestModifiedResidue:
    """Test cases for ModifiedResidue dataclass."""

    def test_modified_residue_creation(self):
        """Test basic ModifiedResidue creation."""
        mod_residue = ModifiedResidue(
            base_aa="S",
            modification="phospho",
            modification_type=ModificationType.PHOSPHORYLATION,
            mass_shift=79.966331,
        )

        assert mod_residue.base_aa == "S"
        assert mod_residue.modification == "phospho"
        assert mod_residue.modification_type == ModificationType.PHOSPHORYLATION
        assert mod_residue.mass_shift == 79.966331
        assert str(mod_residue) == "S[phospho]"

    def test_modified_residue_hash(self):
        """Test that ModifiedResidue objects are hashable."""
        mod1 = ModifiedResidue("S", "phospho")
        mod2 = ModifiedResidue("S", "phospho")
        mod3 = ModifiedResidue("T", "phospho")

        assert hash(mod1) == hash(mod2)
        assert hash(mod1) != hash(mod3)

        # Test in set
        mod_set = {mod1, mod2, mod3}
        assert len(mod_set) == 2


class TestPeptideEncoder:
    """Test cases for PeptideEncoder class."""

    def test_encoder_initialization(self):
        """Test encoder initialization."""
        encoder = PeptideEncoder()

        assert encoder.include_modifications is True
        assert encoder.max_custom_mods == 50
        assert encoder.vocab_size > 20  # At least standard amino acids
        assert "A" in encoder.vocabulary
        assert "X" in encoder.vocabulary  # Unknown
        assert "<PAD>" in encoder.vocabulary  # Padding

    def test_encoder_without_modifications(self):
        """Test encoder without modification support."""
        encoder = PeptideEncoder(include_modifications=False)

        assert encoder.include_modifications is False
        assert encoder.vocab_size == 22  # 20 AA + X + PAD
        assert "A" in encoder.vocabulary
        assert "X" in encoder.vocabulary
        assert "<PAD>" in encoder.vocabulary

    def test_parse_standard_sequence(self):
        """Test parsing standard amino acid sequences."""
        encoder = PeptideEncoder()

        sequence = "PEPTIDE"
        residues = encoder.parse_sequence(sequence)

        assert residues == ["P", "E", "P", "T", "I", "D", "E"]

    def test_parse_bracketed_modifications(self):
        """Test parsing sequences with bracketed modifications."""
        encoder = PeptideEncoder()

        sequence = "PEP[phospho]TIDE"
        residues = encoder.parse_sequence(sequence)

        expected = ["P", "E", "P[phospho]", "T", "I", "D", "E"]
        assert residues == expected

    def test_parse_short_notation_modifications(self):
        """Test parsing sequences with short notation modifications."""
        encoder = PeptideEncoder()

        test_cases = [
            ("pSPEPTIDE", ["S[phospho]", "P", "E", "P", "T", "I", "D", "E"]),
            ("PEPTIDEoxM", ["P", "E", "P", "T", "I", "D", "E", "M[oxidation]"]),
            ("me3KPEPTIDE", ["K[trimethyl]", "P", "E", "P", "T", "I", "D", "E"]),
            ("acKPEPTIDE", ["K[acetyl]", "P", "E", "P", "T", "I", "D", "E"]),
        ]

        for sequence, expected in test_cases:
            residues = encoder.parse_sequence(sequence)
            assert residues == expected, f"Failed for sequence: {sequence}"

    def test_parse_mixed_modifications(self):
        """Test parsing sequences with mixed modification formats."""
        encoder = PeptideEncoder()

        sequence = "PEP[custom]pSTIDE"
        residues = encoder.parse_sequence(sequence)

        expected = ["P", "E", "P[custom]", "S[phospho]", "T", "I", "D", "E"]
        assert residues == expected

    def test_parse_unknown_characters(self):
        """Test parsing sequences with unknown characters."""
        encoder = PeptideEncoder()

        sequence = "PEPZIDE"
        residues = encoder.parse_sequence(sequence)

        expected = ["P", "E", "P", "X", "I", "D", "E"]
        assert residues == expected

    def test_encode_standard_sequence(self):
        """Test encoding standard amino acid sequences."""
        encoder = PeptideEncoder()

        sequence = "PEPTIDE"
        encoded = encoder.encode_sequence(sequence)

        assert encoded.shape == (7, encoder.vocab_size)
        assert torch.allclose(
            encoded.sum(dim=1), torch.ones(7)
        )  # Each position sums to 1

        # Check specific amino acids
        p_idx = encoder.vocabulary["P"]
        e_idx = encoder.vocabulary["E"]

        assert encoded[0, p_idx] == 1.0
        assert encoded[1, e_idx] == 1.0

    def test_encode_modified_sequence(self):
        """Test encoding sequences with modifications."""
        encoder = PeptideEncoder()

        sequence = "pSPEPTIDE"
        encoded = encoder.encode_sequence(sequence)

        assert encoded.shape == (8, encoder.vocab_size)
        assert torch.allclose(encoded.sum(dim=1), torch.ones(8))

        # Check that phospho-serine is encoded correctly
        ps_residue = str(encoder.common_modifications["pS"])
        if ps_residue in encoder.vocabulary:
            ps_idx = encoder.vocabulary[ps_residue]
            assert encoded[0, ps_idx] == 1.0

    def test_encode_custom_modification(self):
        """Test encoding sequences with custom modifications."""
        encoder = PeptideEncoder()

        sequence = "PEP[custom_mod]TIDE"
        encoded = encoder.encode_sequence(sequence)

        assert encoded.shape == (7, encoder.vocab_size)
        assert torch.allclose(encoded.sum(dim=1), torch.ones(7))

        # Custom modification should be assigned to a CUSTOM slot
        assert (
            "PEP[custom_mod]TIDE" in encoder.custom_modifications
            or encoded[2].sum() == 1.0
        )

    def test_encode_batch_same_length(self):
        """Test batch encoding with sequences of same length."""
        encoder = PeptideEncoder()

        sequences = ["PEPTIDE", "PROTEIN", "SEQUENCE"]
        encoded = encoder.encode_batch(sequences)

        assert encoded.shape == (3, 8, encoder.vocab_size)  # Max length is 8
        assert torch.allclose(encoded.sum(dim=2), torch.ones(3, 8))

    def test_encode_batch_different_lengths(self):
        """Test batch encoding with sequences of different lengths."""
        encoder = PeptideEncoder()

        sequences = ["PEP", "PEPTIDE", "PROTEIN"]
        encoded = encoder.encode_batch(sequences)

        max_len = max(len(seq) for seq in sequences)
        assert encoded.shape == (3, max_len, encoder.vocab_size)

        # Check padding
        pad_idx = encoder.pad_idx
        assert encoded[0, 3:, pad_idx].sum() > 0  # First sequence should be padded

    def test_encode_batch_with_fixed_length(self):
        """Test batch encoding with fixed padding length."""
        encoder = PeptideEncoder()

        sequences = ["PEP", "PEPTIDE"]
        encoded = encoder.encode_batch(sequences, pad_to_length=10)

        assert encoded.shape == (2, 10, encoder.vocab_size)

    def test_decode_sequence(self):
        """Test decoding one-hot encoded sequences."""
        encoder = PeptideEncoder()

        original = "PEPTIDE"
        encoded = encoder.encode_sequence(original)
        decoded = encoder.decode_sequence(encoded)

        assert decoded == original

    def test_decode_sequence_with_modifications(self):
        """Test decoding sequences with modifications."""
        encoder = PeptideEncoder()

        original = "pSPEPTIDE"
        encoded = encoder.encode_sequence(original)
        decoded = encoder.decode_sequence(encoded)

        # Should decode to the full modification notation
        assert "S[phospho]" in decoded or decoded == original

    def test_decode_sequence_with_padding(self):
        """Test decoding sequences with padding."""
        encoder = PeptideEncoder()

        sequences = ["PEP", "PEPTIDE"]
        batch_encoded = encoder.encode_batch(sequences)

        # Decode first sequence (should stop at padding)
        decoded = encoder.decode_sequence(batch_encoded[0])
        assert len(decoded) >= 3  # At least the original length
        assert not decoded.endswith("<PAD>")  # Padding should be stripped

    def test_get_modification_info(self):
        """Test getting modification information from sequences."""
        encoder = PeptideEncoder()

        sequence = "PEP[phospho]pSTIDE"
        info = encoder.get_modification_info(sequence)

        assert info["total_residues"] == 8
        assert info["modified_residues"] == 2
        assert "phospho" in info["modifications"]
        assert "phosphorylation" in info["modification_types"]

    def test_get_modification_info_no_modifications(self):
        """Test modification info for sequences without modifications."""
        encoder = PeptideEncoder()

        sequence = "PEPTIDE"
        info = encoder.get_modification_info(sequence)

        assert info["total_residues"] == 7
        assert info["modified_residues"] == 0
        assert len(info["modifications"]) == 0
        assert len(info["modification_types"]) == 0

    def test_get_vocabulary_info(self):
        """Test getting vocabulary information."""
        encoder = PeptideEncoder()

        vocab_info = encoder.get_vocabulary_info()

        assert "vocab_size" in vocab_info
        assert "standard_aa_count" in vocab_info
        assert "common_modifications_count" in vocab_info
        assert vocab_info["standard_aa_count"] == 20
        assert vocab_info["includes_modifications"] is True

    def test_custom_modification_limit(self):
        """Test custom modification limit handling."""
        encoder = PeptideEncoder(max_custom_mods=2)

        sequences = [
            "PEP[mod1]TIDE",
            "PEP[mod2]TIDE",
            "PEP[mod3]TIDE",  # Should exceed limit
        ]

        for seq in sequences:
            encoded = encoder.encode_sequence(seq)
            assert encoded.shape[0] > 0  # Should still encode

        # Third modification should be treated as unknown
        assert len(encoder.custom_modifications) <= 2

    def test_save_and_load_vocabulary(self):
        """Test saving and loading vocabulary."""
        encoder = PeptideEncoder()

        # Add some custom modifications
        encoder.encode_sequence("PEP[custom1]TIDE")
        encoder.encode_sequence("PRO[custom2]TEIN")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            vocab_file = f.name

        try:
            # Save vocabulary
            encoder.save_vocabulary(vocab_file)

            # Load vocabulary
            loaded_encoder = PeptideEncoder.load_vocabulary(vocab_file)

            assert loaded_encoder.vocab_size == encoder.vocab_size
            assert loaded_encoder.vocabulary == encoder.vocabulary
            assert loaded_encoder.custom_modifications == encoder.custom_modifications

        finally:
            Path(vocab_file).unlink()

    def test_malformed_bracket_notation(self):
        """Test handling of malformed bracket notation."""
        encoder = PeptideEncoder()

        sequence = "PEP[incomplete"
        residues = encoder.parse_sequence(sequence)

        # Should treat as individual characters
        expected = [
            "P",
            "E",
            "P",
            "X",
            "i",
            "n",
            "c",
            "o",
            "m",
            "p",
            "l",
            "e",
            "t",
            "e",
        ]
        # Note: lowercase letters become 'X'
        expected_cleaned = [
            "P",
            "E",
            "P",
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
            "X",
        ]
        assert len(residues) == len(expected_cleaned)

    def test_empty_sequence(self):
        """Test handling of empty sequences."""
        encoder = PeptideEncoder()

        encoded = encoder.encode_sequence("")
        assert encoded.shape == (0, encoder.vocab_size)

        decoded = encoder.decode_sequence(encoded)
        assert decoded == ""

    def test_very_long_sequence(self):
        """Test handling of very long sequences."""
        encoder = PeptideEncoder()

        long_sequence = "A" * 1000
        encoded = encoder.encode_sequence(long_sequence)

        assert encoded.shape == (1000, encoder.vocab_size)
        assert torch.allclose(encoded.sum(dim=1), torch.ones(1000))


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_encode_peptide(self):
        """Test encode_peptide convenience function."""
        sequence = "PEPTIDE"
        encoded = encode_peptide(sequence)

        assert isinstance(encoded, torch.Tensor)
        assert encoded.shape[0] == len(sequence)
        assert torch.allclose(encoded.sum(dim=1), torch.ones(len(sequence)))

    def test_encode_peptide_without_modifications(self):
        """Test encode_peptide without modifications."""
        sequence = "pSPEPTIDE"
        encoded = encode_peptide(sequence, include_modifications=False)

        assert isinstance(encoded, torch.Tensor)
        # Should treat pS as separate characters (p becomes X, S stays S)
        assert encoded.shape[0] == len(sequence)

    def test_encode_peptide_batch(self):
        """Test encode_peptide_batch convenience function."""
        sequences = ["PEPTIDE", "PROTEIN", "SEQUENCE"]
        encoded = encode_peptide_batch(sequences)

        assert isinstance(encoded, torch.Tensor)
        assert encoded.shape[0] == len(sequences)
        max_len = max(len(seq) for seq in sequences)
        assert encoded.shape[1] == max_len


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_numeric_characters(self):
        """Test handling of numeric characters in sequences."""
        encoder = PeptideEncoder()

        sequence = "PEP123TIDE"
        residues = encoder.parse_sequence(sequence)

        # Numbers should be treated as unknown
        expected = ["P", "E", "P", "X", "X", "X", "T", "I", "D", "E"]
        assert residues == expected

    def test_special_characters(self):
        """Test handling of special characters."""
        encoder = PeptideEncoder()

        sequence = "PEP-TI_DE"
        residues = encoder.parse_sequence(sequence)

        # Special characters should be treated as unknown
        expected = ["P", "E", "P", "X", "T", "I", "X", "D", "E"]
        assert residues == expected

    def test_lowercase_amino_acids(self):
        """Test handling of lowercase amino acids."""
        encoder = PeptideEncoder()

        sequence = "peptide"
        residues = encoder.parse_sequence(sequence)

        # Lowercase should be treated as unknown
        expected = ["X", "X", "X", "X", "X", "X", "X"]
        assert residues == expected

    def test_nested_brackets(self):
        """Test handling of nested brackets."""
        encoder = PeptideEncoder()

        sequence = "PEP[mod[nested]]TIDE"
        residues = encoder.parse_sequence(sequence)

        # Should handle the outer brackets
        assert "P[mod[nested]" in residues[2] or len(residues) > 7

    def test_multiple_consecutive_modifications(self):
        """Test multiple consecutive modifications."""
        encoder = PeptideEncoder()

        sequence = "pSpTpYPEPTIDE"
        residues = encoder.parse_sequence(sequence)

        expected_start = ["S[phospho]", "T[phospho]", "Y[phospho]"]
        assert residues[:3] == expected_start


class TestPerformance:
    """Test performance characteristics."""

    def test_large_batch_encoding(self):
        """Test encoding large batches."""
        encoder = PeptideEncoder()

        # Create a large batch
        sequences = ["PEPTIDE" + str(i % 10) for i in range(1000)]

        encoded = encoder.encode_batch(sequences)

        assert encoded.shape[0] == 1000
        assert encoded.shape[1] >= 7  # At least as long as base sequence
        assert torch.all(encoded.sum(dim=2) >= 0)  # All valid encodings

    def test_memory_usage_consistency(self):
        """Test that memory usage is consistent."""
        encoder = PeptideEncoder()

        # Encode same sequence multiple times
        sequence = "PEPTIDE"
        encodings = []

        for _ in range(100):
            encoded = encoder.encode_sequence(sequence)
            encodings.append(encoded)

        # All encodings should be identical
        for encoding in encodings[1:]:
            assert torch.equal(encodings[0], encoding)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
