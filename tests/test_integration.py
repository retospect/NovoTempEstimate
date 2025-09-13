"""
Integration tests for data reader and peptide encoder.

This module tests the complete workflow from reading protein data
to encoding sequences with the peptide encoder.
"""

import pytest
import torch
from pathlib import Path
import time
from typing import List, Dict, Any

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from novotempestimate.data_reader import TemStaProReader, ProteinRecord
from novotempestimate.peptide_encoder import PeptideEncoder


class TestDataReaderPeptideEncoderIntegration:
    """Integration tests between data reader and peptide encoder."""

    def test_encode_sample_sequences(self):
        """Test encoding a sample of sequences from the dataset."""
        with TemStaProReader() as reader:
            # Load a small sample from testing data (smaller files)
            sample_records = []
            count = 0

            for record in reader.read_dataset("*sample2k*"):
                sample_records.append(record)
                count += 1
                if count >= 100:  # Test with first 100 sequences
                    break

            if not sample_records:
                pytest.skip("No sample data available for testing")

            print(f"\nTesting with {len(sample_records)} sequences")

            # Create peptide encoder
            encoder = PeptideEncoder(include_modifications=True)

            # Test encoding individual sequences
            successful_encodings = 0
            failed_encodings = 0
            encoding_stats = {
                "total_sequences": len(sample_records),
                "min_length": float("inf"),
                "max_length": 0,
                "avg_length": 0,
                "vocab_size": encoder.vocab_size,
            }

            sequences = []
            for record in sample_records:
                try:
                    encoded = encoder.encode_sequence(record.sequence)

                    # Verify encoding properties
                    assert encoded.shape[0] == len(record.sequence)
                    assert encoded.shape[1] == encoder.vocab_size
                    assert torch.allclose(
                        encoded.sum(dim=1), torch.ones(len(record.sequence))
                    )

                    sequences.append(record.sequence)
                    successful_encodings += 1

                    # Update stats
                    seq_len = len(record.sequence)
                    encoding_stats["min_length"] = min(
                        encoding_stats["min_length"], seq_len
                    )
                    encoding_stats["max_length"] = max(
                        encoding_stats["max_length"], seq_len
                    )

                except Exception as e:
                    failed_encodings += 1
                    print(f"Failed to encode sequence {record.id}: {e}")

            encoding_stats["avg_length"] = (
                sum(len(seq) for seq in sequences) / len(sequences) if sequences else 0
            )
            encoding_stats["successful_encodings"] = successful_encodings
            encoding_stats["failed_encodings"] = failed_encodings

            print(f"Encoding statistics: {encoding_stats}")

            # Should have high success rate
            success_rate = successful_encodings / len(sample_records)
            assert success_rate > 0.95, f"Success rate too low: {success_rate:.2%}"

            # Test batch encoding
            if sequences:
                batch_size = min(10, len(sequences))
                batch_sequences = sequences[:batch_size]

                batch_encoded = encoder.encode_batch(batch_sequences)

                assert batch_encoded.shape[0] == batch_size
                assert batch_encoded.shape[1] >= max(
                    len(seq) for seq in batch_sequences
                )
                assert batch_encoded.shape[2] == encoder.vocab_size

                print(f"Batch encoding successful: {batch_encoded.shape}")

    def test_encode_training_sequences_sample(self):
        """Test encoding a sample of training sequences."""
        with TemStaProReader() as reader:
            # Load a very small sample from training data
            training_records = []
            count = 0

            for record in reader.read_dataset("*training*"):
                training_records.append(record)
                count += 1
                if count >= 50:  # Small sample due to large dataset
                    break

            if not training_records:
                pytest.skip("No training data available for testing")

            print(f"\nTesting training data with {len(training_records)} sequences")

            encoder = PeptideEncoder(include_modifications=True)

            # Test encoding and collect statistics
            lengths = []
            unique_chars = set()

            for record in training_records:
                try:
                    encoded = encoder.encode_sequence(record.sequence)

                    # Basic validation
                    assert encoded.shape[0] == len(record.sequence)
                    assert encoded.shape[1] == encoder.vocab_size

                    lengths.append(len(record.sequence))
                    unique_chars.update(record.sequence)

                except Exception as e:
                    print(f"Failed to encode training sequence {record.id}: {e}")

            stats = {
                "sequences_tested": len(training_records),
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "avg_length": sum(lengths) / len(lengths) if lengths else 0,
                "unique_characters": len(unique_chars),
                "vocab_size": encoder.vocab_size,
            }

            print(f"Training data statistics: {stats}")

            # Verify we can handle the character diversity
            assert len(unique_chars) <= 25  # Should be mostly standard amino acids

    def test_encoder_with_different_sequence_types(self):
        """Test encoder with different types of sequences from the dataset."""
        with TemStaProReader() as reader:
            # Collect sequences of different lengths and characteristics
            short_sequences = []
            medium_sequences = []
            long_sequences = []

            count = 0
            for record in reader.read_dataset("*sample2k*"):
                seq_len = len(record.sequence)

                if seq_len < 50:
                    short_sequences.append(record.sequence)
                elif seq_len < 200:
                    medium_sequences.append(record.sequence)
                else:
                    long_sequences.append(record.sequence)

                count += 1
                if count >= 150:  # Limit total sequences
                    break

            if not (short_sequences or medium_sequences or long_sequences):
                pytest.skip("No sequences available for testing")

            encoder = PeptideEncoder(include_modifications=True)

            # Test different sequence types
            test_cases = [
                ("short", short_sequences[:10]),
                ("medium", medium_sequences[:10]),
                ("long", long_sequences[:5]),
            ]

            for seq_type, sequences in test_cases:
                if not sequences:
                    continue

                print(f"\nTesting {seq_type} sequences: {len(sequences)} sequences")

                # Individual encoding
                for seq in sequences:
                    encoded = encoder.encode_sequence(seq)
                    assert encoded.shape == (len(seq), encoder.vocab_size)

                # Batch encoding
                if len(sequences) > 1:
                    batch_encoded = encoder.encode_batch(sequences)
                    max_len = max(len(seq) for seq in sequences)
                    assert batch_encoded.shape == (
                        len(sequences),
                        max_len,
                        encoder.vocab_size,
                    )

                print(f"{seq_type.capitalize()} sequences encoded successfully")

    def test_encoder_performance_with_real_data(self):
        """Test encoder performance with real dataset sequences."""
        with TemStaProReader() as reader:
            # Load sequences for performance testing
            sequences = []
            count = 0

            for record in reader.read_dataset("*sample2k*"):
                sequences.append(record.sequence)
                count += 1
                if count >= 200:  # Performance test with 200 sequences
                    break

            if not sequences:
                pytest.skip("No sequences available for performance testing")

            encoder = PeptideEncoder(include_modifications=True)

            # Time individual encoding
            start_time = time.time()
            individual_encodings = []

            for seq in sequences:
                encoded = encoder.encode_sequence(seq)
                individual_encodings.append(encoded)

            individual_time = time.time() - start_time

            # Time batch encoding
            start_time = time.time()
            batch_encoded = encoder.encode_batch(sequences)
            batch_time = time.time() - start_time

            performance_stats = {
                "sequences_count": len(sequences),
                "individual_encoding_time": individual_time,
                "batch_encoding_time": batch_time,
                "individual_time_per_seq": individual_time / len(sequences),
                "batch_time_per_seq": batch_time / len(sequences),
                "speedup_factor": individual_time / batch_time if batch_time > 0 else 0,
            }

            print(f"\nPerformance statistics: {performance_stats}")

            # Verify batch encoding performance is reasonable (allow some overhead)
            if len(sequences) > 50:
                # Batch encoding should be within 50% of individual encoding time
                assert (
                    batch_time < individual_time * 1.5
                ), f"Batch encoding too slow: {batch_time:.3f}s vs {individual_time:.3f}s"

            # Verify encodings are consistent
            for i, seq in enumerate(sequences):
                individual_encoded = individual_encodings[i]
                batch_slice = batch_encoded[i, : individual_encoded.shape[0]]

                # Should be identical (within floating point precision)
                assert torch.allclose(individual_encoded, batch_slice, atol=1e-6)

    def test_encoder_vocabulary_coverage(self):
        """Test that encoder vocabulary covers characters in real data."""
        with TemStaProReader() as reader:
            # Collect character statistics from real data
            all_chars = set()
            char_counts = {}
            sequence_count = 0

            for record in reader.read_dataset("*sample2k*"):
                for char in record.sequence:
                    all_chars.add(char)
                    char_counts[char] = char_counts.get(char, 0) + 1

                sequence_count += 1
                if sequence_count >= 1000:  # Sample 1000 sequences
                    break

            if not all_chars:
                pytest.skip("No character data available for testing")

            encoder = PeptideEncoder(include_modifications=True)

            # Check vocabulary coverage
            covered_chars = set()
            uncovered_chars = set()

            for char in all_chars:
                if char in encoder.vocabulary:
                    covered_chars.add(char)
                else:
                    uncovered_chars.add(char)

            coverage_stats = {
                "total_unique_chars": len(all_chars),
                "covered_chars": len(covered_chars),
                "uncovered_chars": len(uncovered_chars),
                "coverage_percentage": len(covered_chars) / len(all_chars) * 100,
                "vocab_size": encoder.vocab_size,
                "most_common_uncovered": sorted(
                    [(char, char_counts.get(char, 0)) for char in uncovered_chars],
                    key=lambda x: x[1],
                    reverse=True,
                )[:10],
            }

            print(f"\nVocabulary coverage statistics: {coverage_stats}")

            # Should cover most standard amino acids
            standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
            covered_standard = standard_aa.intersection(covered_chars)
            assert (
                len(covered_standard) == 20
            ), f"Should cover all 20 standard amino acids, got {len(covered_standard)}"

            # Coverage should be reasonable (most characters should be standard AAs)
            assert (
                coverage_stats["coverage_percentage"] > 80
            ), f"Coverage too low: {coverage_stats['coverage_percentage']:.1f}%"

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow from data loading to encoding."""
        print("\n=== End-to-End Integration Test ===")

        # Step 1: Load data
        with TemStaProReader() as reader:
            records = []
            for record in reader.read_dataset("*sample2k*"):
                records.append(record)
                if len(records) >= 20:  # Small sample for end-to-end test
                    break

            if not records:
                pytest.skip("No data available for end-to-end test")

            print(f"Step 1: Loaded {len(records)} protein records")

        # Step 2: Create encoder
        encoder = PeptideEncoder(include_modifications=True)
        print(f"Step 2: Created encoder with vocabulary size {encoder.vocab_size}")

        # Step 3: Process sequences
        processed_data = []

        for record in records:
            # Parse sequence
            parsed_residues = encoder.parse_sequence(record.sequence)

            # Encode sequence
            encoded = encoder.encode_sequence(record.sequence)

            # Get modification info
            mod_info = encoder.get_modification_info(record.sequence)

            processed_data.append(
                {
                    "record_id": record.id,
                    "original_sequence": record.sequence,
                    "parsed_residues": parsed_residues,
                    "encoded_shape": encoded.shape,
                    "modification_info": mod_info,
                }
            )

        print(f"Step 3: Processed {len(processed_data)} sequences")

        # Step 4: Batch processing
        sequences = [record.sequence for record in records]
        batch_encoded = encoder.encode_batch(sequences)

        print(f"Step 4: Batch encoded shape: {batch_encoded.shape}")

        # Step 5: Validation
        for i, data in enumerate(processed_data):
            # Verify individual vs batch encoding consistency
            individual_encoded = encoder.encode_sequence(data["original_sequence"])
            batch_slice = batch_encoded[i, : individual_encoded.shape[0]]

            assert torch.allclose(individual_encoded, batch_slice, atol=1e-6)

            # Verify decoding works
            decoded = encoder.decode_sequence(individual_encoded)
            assert (
                len(decoded) >= len(data["original_sequence"]) * 0.8
            )  # Allow for some modification differences

        print("Step 5: Validation completed successfully")

        # Summary statistics
        total_residues = sum(len(data["parsed_residues"]) for data in processed_data)
        total_modifications = sum(
            data["modification_info"]["modified_residues"] for data in processed_data
        )

        summary = {
            "total_sequences": len(processed_data),
            "total_residues": total_residues,
            "total_modifications": total_modifications,
            "avg_sequence_length": total_residues / len(processed_data),
            "modification_rate": (
                total_modifications / total_residues * 100 if total_residues > 0 else 0
            ),
        }

        print(f"\n=== End-to-End Summary ===")
        print(f"Successfully processed {summary['total_sequences']} sequences")
        print(f"Total residues: {summary['total_residues']}")
        print(f"Average sequence length: {summary['avg_sequence_length']:.1f}")
        print(f"Modification rate: {summary['modification_rate']:.2f}%")

        # Final assertions
        assert summary["total_sequences"] > 0
        assert summary["avg_sequence_length"] > 0
        assert batch_encoded.shape[0] == summary["total_sequences"]


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])
