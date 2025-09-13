"""
Comprehensive tests for the data_reader module.
"""

import pytest
import tempfile
import gzip
import tarfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from novotempestimate.data_reader import (
    ProteinRecord,
    TemStaProReader,
    load_supplementary_predictions,
    quick_load_training,
    quick_load_testing,
    quick_load_validation,
)


class TestProteinRecord:
    """Test cases for ProteinRecord dataclass."""

    def test_protein_record_creation(self):
        """Test basic ProteinRecord creation."""
        record = ProteinRecord(
            id="test_id", sequence="MKFLVLL", description="Test protein"
        )

        assert record.id == "test_id"
        assert record.sequence == "MKFLVLL"
        assert record.description == "Test protein"
        assert record.length == 7
        assert record.temperature is None
        assert record.organism is None

    def test_temperature_extraction_from_description(self):
        """Test temperature extraction from description."""
        descriptions = [
            "Protein temp: 85.5 degrees",
            "Temperature 70.2",
            "temp=95.0",
            "TEMP: 60",
        ]

        expected_temps = [85.5, 70.2, 95.0, 60.0]

        for desc, expected_temp in zip(descriptions, expected_temps):
            record = ProteinRecord(id="test", sequence="MKFLVLL", description=desc)
            assert record.temperature == expected_temp

    def test_organism_extraction_from_description(self):
        """Test organism extraction from description."""
        description = (
            "sp|P12345|TEST_HUMAN Test protein OS=Homo sapiens OX=9606 GN=TEST"
        )

        record = ProteinRecord(id="test", sequence="MKFLVLL", description=description)

        assert record.organism == "Homo sapiens"

    def test_manual_temperature_override(self):
        """Test that manually set temperature is not overridden."""
        record = ProteinRecord(
            id="test", sequence="MKFLVLL", description="temp: 85.5", temperature=100.0
        )

        assert record.temperature == 100.0


class TestTemStaProReader:
    """Test cases for TemStaProReader class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)

            # Create sample FASTA content
            fasta_content = """>protein1|temp:75.5
MKFLVLLFNILCLFPVLAADNHGVGPQGASGILKTLLKQIGDLQAGLQGVQAGVWPAAVRESVPSLL
>protein2|temp:85.0
MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGD
"""

            # Create sample files
            fasta_file = data_dir / "test.fasta"
            fasta_file.write_text(fasta_content)

            # Create gzipped version
            gz_file = data_dir / "test.fasta.gz"
            with gzip.open(gz_file, "wt") as f:
                f.write(fasta_content)

            # Create tar.gz version
            tar_gz_file = data_dir / "TemStaPro-test-training.fasta.tar.gz"
            with tarfile.open(tar_gz_file, "w:gz") as tar:
                tar.add(fasta_file, arcname="test.fasta")

            yield data_dir

    def test_reader_initialization(self, temp_data_dir):
        """Test reader initialization."""
        reader = TemStaProReader(temp_data_dir)
        assert reader.data_dir == temp_data_dir
        assert reader.temp_dir is None

    def test_context_manager(self, temp_data_dir):
        """Test context manager functionality."""
        with TemStaProReader(temp_data_dir) as reader:
            assert reader.temp_dir is not None
            assert Path(reader.temp_dir).exists()
            temp_dir_path = reader.temp_dir

        # After exiting context, temp dir should be cleaned up
        assert not Path(temp_dir_path).exists()

    def test_list_available_files(self, temp_data_dir):
        """Test listing available files."""
        reader = TemStaProReader(temp_data_dir)
        files = reader.list_available_files()

        assert len(files) >= 2  # At least .gz and .tar.gz files
        file_names = [f.name for f in files]
        assert any("test.fasta.gz" in name for name in file_names)
        assert any(
            "TemStaPro-test-training.fasta.tar.gz" in name for name in file_names
        )

    def test_extract_gz_archive(self, temp_data_dir):
        """Test extracting .gz files."""
        with TemStaProReader(temp_data_dir) as reader:
            gz_file = temp_data_dir / "test.fasta.gz"
            extracted_files = reader.extract_archive(gz_file)

            assert len(extracted_files) == 1
            assert extracted_files[0].exists()
            assert extracted_files[0].name == "test.fasta"

    def test_extract_tar_gz_archive(self, temp_data_dir):
        """Test extracting .tar.gz files."""
        with TemStaProReader(temp_data_dir) as reader:
            tar_gz_file = temp_data_dir / "TemStaPro-test-training.fasta.tar.gz"
            extracted_files = reader.extract_archive(tar_gz_file)

            assert len(extracted_files) == 1
            assert extracted_files[0].exists()
            assert extracted_files[0].name == "test.fasta"

    def test_read_fasta_file(self, temp_data_dir):
        """Test reading FASTA files."""
        reader = TemStaProReader(temp_data_dir)
        fasta_file = temp_data_dir / "test.fasta"

        records = list(reader.read_fasta_file(fasta_file))

        assert len(records) == 2
        assert records[0].id == "protein1|temp:75.5"
        assert records[0].temperature == 75.5
        assert records[1].id == "protein2|temp:85.0"
        assert records[1].temperature == 85.0

    def test_read_dataset_with_pattern(self, temp_data_dir):
        """Test reading dataset with file pattern."""
        with TemStaProReader(temp_data_dir) as reader:
            records = list(reader.read_dataset("*training*"))

            assert len(records) == 2  # Two proteins in the training file
            assert all(isinstance(r, ProteinRecord) for r in records)

    def test_to_dataframe(self, temp_data_dir):
        """Test converting records to DataFrame."""
        with TemStaProReader(temp_data_dir) as reader:
            records = list(reader.read_dataset("*training*"))
            df = reader.to_dataframe(records)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "id" in df.columns
            assert "sequence" in df.columns
            assert "temperature" in df.columns
            assert "length" in df.columns

    def test_get_dataset_statistics(self, temp_data_dir):
        """Test getting dataset statistics."""
        with TemStaProReader(temp_data_dir) as reader:
            records = list(reader.read_dataset("*training*"))
            stats = reader.get_dataset_statistics(records)

            assert stats["total_records"] == 2
            assert stats["records_with_temperature"] == 2
            assert stats["unique_sequences"] == 2
            assert "avg_sequence_length" in stats
            assert "avg_temperature" in stats

    def test_filter_by_length(self, temp_data_dir):
        """Test filtering records by sequence length."""
        with TemStaProReader(temp_data_dir) as reader:
            records = list(reader.read_dataset("*training*"))

            # Filter to very short sequences (should return empty)
            short_records = reader.filter_by_length(
                records, min_length=1, max_length=10
            )
            assert len(short_records) == 0

            # Filter to include all sequences
            all_records = reader.filter_by_length(
                records, min_length=1, max_length=1000
            )
            assert len(all_records) == 2

    def test_filter_by_temperature(self, temp_data_dir):
        """Test filtering records by temperature."""
        with TemStaProReader(temp_data_dir) as reader:
            records = list(reader.read_dataset("*training*"))

            # Filter to temperatures above 80
            hot_records = reader.filter_by_temperature(records, min_temp=80.0)
            assert len(hot_records) == 1
            assert hot_records[0].temperature == 85.0

            # Filter to temperatures below 80
            cool_records = reader.filter_by_temperature(records, max_temp=80.0)
            assert len(cool_records) == 1
            assert cool_records[0].temperature == 75.5

    def test_load_training_data(self, temp_data_dir):
        """Test loading training data specifically."""
        with TemStaProReader(temp_data_dir) as reader:
            records = reader.load_training_data()
            assert len(records) == 2
            assert all(isinstance(r, ProteinRecord) for r in records)

    def test_empty_dataset_statistics(self):
        """Test statistics with empty dataset."""
        reader = TemStaProReader()
        stats = reader.get_dataset_statistics([])
        assert stats == {}


class TestSupplementaryFunctions:
    """Test supplementary functions."""

    @patch("novotempestimate.data_reader.pd.read_csv")
    def test_load_supplementary_predictions(self, mock_read_csv):
        """Test loading supplementary predictions file."""
        # Mock the CSV reading
        mock_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        mock_read_csv.return_value = mock_df

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            supp_file = data_dir / "SupplementaryFileC2EPsPredictions.tsv"
            supp_file.touch()  # Create empty file

            result = load_supplementary_predictions(data_dir)

            mock_read_csv.assert_called_once_with(supp_file, sep="\t")
            assert result.equals(mock_df)

    def test_load_supplementary_predictions_file_not_found(self):
        """Test loading supplementary predictions when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError):
                load_supplementary_predictions(temp_dir)

    @patch("novotempestimate.data_reader.TemStaProReader")
    def test_quick_load_functions(self, mock_reader_class):
        """Test quick load convenience functions."""
        # Mock the reader instance
        mock_reader = MagicMock()
        mock_reader.__enter__ = MagicMock(return_value=mock_reader)
        mock_reader.__exit__ = MagicMock(return_value=None)
        mock_reader.load_training_data.return_value = ["training_data"]
        mock_reader.load_testing_data.return_value = ["testing_data"]
        mock_reader.load_validation_data.return_value = ["validation_data"]

        mock_reader_class.return_value = mock_reader

        # Test quick load functions
        training_result = quick_load_training()
        testing_result = quick_load_testing()
        validation_result = quick_load_validation()

        assert training_result == ["training_data"]
        assert testing_result == ["testing_data"]
        assert validation_result == ["validation_data"]

        # Verify the reader was called correctly
        assert mock_reader_class.call_count == 3
        mock_reader.load_training_data.assert_called_once()
        mock_reader.load_testing_data.assert_called_once()
        mock_reader.load_validation_data.assert_called_once()


class TestIntegration:
    """Integration tests."""

    def test_full_workflow_with_mock_data(self):
        """Test complete workflow with mock data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)

            # Create mock FASTA content with realistic protein data
            fasta_content = """>sp|P12345|TEST1_HUMAN Test protein 1 OS=Homo sapiens temp:75.5
MKFLVLLFNILCLFPVLAADNHGVGPQGASGILKTLLKQIGDLQAGLQGVQAGVWPAAVRESVPSLL
>sp|P67890|TEST2_MOUSE Test protein 2 OS=Mus musculus temp:85.0
MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGD
>sp|P11111|TEST3_YEAST Test protein 3 OS=Saccharomyces cerevisiae temp:65.0
MKLVLLFNILCLFPVLAADNHGVGPQGASGILKTLLKQIGDLQAGLQGVQAGVWPAAVRESVPSLL
"""

            # Create training file
            training_file = data_dir / "TemStaPro-Major-30-imbal-training.fasta.tar.gz"
            fasta_file = data_dir / "training.fasta"
            fasta_file.write_text(fasta_content)

            with tarfile.open(training_file, "w:gz") as tar:
                tar.add(fasta_file, arcname="training.fasta")

            # Test complete workflow
            with TemStaProReader(data_dir) as reader:
                # Load data
                records = reader.load_training_data()
                assert len(records) == 3

                # Check organism extraction
                organisms = [r.organism for r in records if r.organism]
                assert "Homo sapiens" in organisms
                assert "Mus musculus" in organisms
                assert "Saccharomyces cerevisiae" in organisms

                # Filter by temperature
                hot_proteins = reader.filter_by_temperature(records, min_temp=80.0)
                assert len(hot_proteins) == 1
                assert hot_proteins[0].temperature == 85.0

                # Convert to DataFrame
                df = reader.to_dataframe(records)
                assert len(df) == 3
                assert df["temperature"].mean() == (75.5 + 85.0 + 65.0) / 3

                # Get statistics
                stats = reader.get_dataset_statistics(records)
                assert stats["total_records"] == 3
                assert stats["records_with_temperature"] == 3
                assert stats["avg_temperature"] == (75.5 + 85.0 + 65.0) / 3


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
