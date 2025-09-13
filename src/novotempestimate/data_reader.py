"""
Data reader library for zipped FASTA files from the TemStaPro dataset.

This module provides functionality to read and parse compressed FASTA files
containing protein sequences with temperature annotations.
"""

import gzip
import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Iterator, Optional, Union
from dataclasses import dataclass
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pandas as pd
import re


@dataclass
class ProteinRecord:
    """
    Data class representing a protein record with temperature information.
    """
    
    id: str
    sequence: str
    description: str
    temperature: Optional[float] = None
    organism: Optional[str] = None
    length: int = 0
    
    def __post_init__(self):
        """Calculate sequence length after initialization."""
        self.length = len(self.sequence)
        
        # Extract temperature from description if available
        if self.temperature is None and self.description:
            temp_match = re.search(r'temp[erature]*[:\s=]*([0-9.]+)', self.description.lower())
            if temp_match:
                self.temperature = float(temp_match.group(1))
        
        # Extract organism from description if available
        if self.organism is None and self.description:
            org_match = re.search(r'OS=([^=]+?)(?:\s+[A-Z]{2}=|$)', self.description)
            if org_match:
                organism_part = org_match.group(1).strip()
                # Remove temperature info if it got included
                organism_clean = re.sub(r'\s+temp[:\s]*[0-9.]+.*$', '', organism_part, flags=re.IGNORECASE)
                self.organism = organism_clean.strip()


class TemStaProReader:
    """
    Reader for TemStaPro dataset files.
    
    This class handles reading compressed FASTA files from the TemStaPro dataset,
    extracting protein sequences and their associated temperature information.
    """
    
    def __init__(self, data_dir: Union[str, Path] = None):
        """
        Initialize the reader.
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent / "data"
        self.temp_dir = None
        
    def __enter__(self):
        """Context manager entry."""
        self.temp_dir = tempfile.mkdtemp()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup temporary directory."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def list_available_files(self) -> List[Path]:
        """
        List all available TemStaPro data files.
        
        Returns:
            List of available file paths
        """
        if not self.data_dir.exists():
            return []
        
        patterns = [
            "TemStaPro-*.fasta.tar.gz",
            "TemStaPro-*.fasta.gz",
            "*.fasta.tar.gz",
            "*.fasta.gz"
        ]
        
        files = []
        for pattern in patterns:
            files.extend(self.data_dir.glob(pattern))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for file in files:
            if file not in seen:
                seen.add(file)
                unique_files.append(file)
        
        return sorted(unique_files)
    
    def extract_archive(self, archive_path: Path, extract_to: Path = None) -> List[Path]:
        """
        Extract a compressed archive.
        
        Args:
            archive_path: Path to the archive file
            extract_to: Directory to extract to (uses temp dir if None)
            
        Returns:
            List of extracted file paths
        """
        if extract_to is None:
            if self.temp_dir is None:
                self.temp_dir = tempfile.mkdtemp()
            extract_to = Path(self.temp_dir)
        
        extracted_files = []
        
        if archive_path.suffix == '.gz' and '.tar' in archive_path.name:
            # Handle .tar.gz files
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_to)
                extracted_files = [extract_to / member.name for member in tar.getmembers() if member.isfile()]
        elif archive_path.suffix == '.gz':
            # Handle .gz files
            output_path = extract_to / archive_path.stem
            with gzip.open(archive_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            extracted_files = [output_path]
        else:
            # Not compressed
            extracted_files = [archive_path]
        
        return extracted_files
    
    def read_fasta_file(self, fasta_path: Path) -> Iterator[ProteinRecord]:
        """
        Read a FASTA file and yield ProteinRecord objects.
        
        Args:
            fasta_path: Path to the FASTA file
            
        Yields:
            ProteinRecord objects
        """
        try:
            with open(fasta_path, 'r') as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    yield ProteinRecord(
                        id=record.id,
                        sequence=str(record.seq),
                        description=record.description
                    )
        except Exception as e:
            print(f"Error reading FASTA file {fasta_path}: {e}")
    
    def read_dataset(self, file_pattern: str = None) -> Iterator[ProteinRecord]:
        """
        Read protein records from dataset files.
        
        Args:
            file_pattern: Pattern to match files (e.g., "*training*", "*testing*")
            
        Yields:
            ProteinRecord objects
        """
        available_files = self.list_available_files()
        
        if file_pattern:
            import fnmatch
            available_files = [f for f in available_files if fnmatch.fnmatch(f.name, file_pattern)]
        
        if not available_files:
            print(f"No files found matching pattern: {file_pattern}")
            return
        
        for file_path in available_files:
            print(f"Reading file: {file_path.name}")
            
            try:
                # Extract the archive
                extracted_files = self.extract_archive(file_path)
                
                # Read FASTA files
                for extracted_file in extracted_files:
                    if extracted_file.suffix in ['.fasta', '.fa', '.fas']:
                        yield from self.read_fasta_file(extracted_file)
                        
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    def load_training_data(self) -> List[ProteinRecord]:
        """
        Load training dataset.
        
        Returns:
            List of ProteinRecord objects from training files
        """
        records = list(self.read_dataset("*training*"))
        print(f"Loaded {len(records)} training records")
        return records
    
    def load_testing_data(self) -> List[ProteinRecord]:
        """
        Load testing dataset.
        
        Returns:
            List of ProteinRecord objects from testing files
        """
        records = list(self.read_dataset("*testing*"))
        print(f"Loaded {len(records)} testing records")
        return records
    
    def load_validation_data(self) -> List[ProteinRecord]:
        """
        Load validation dataset.
        
        Returns:
            List of ProteinRecord objects from validation files
        """
        records = list(self.read_dataset("*validation*"))
        print(f"Loaded {len(records)} validation records")
        return records
    
    def to_dataframe(self, records: List[ProteinRecord]) -> pd.DataFrame:
        """
        Convert protein records to pandas DataFrame.
        
        Args:
            records: List of ProteinRecord objects
            
        Returns:
            DataFrame with protein data
        """
        data = []
        for record in records:
            data.append({
                'id': record.id,
                'sequence': record.sequence,
                'description': record.description,
                'temperature': record.temperature,
                'organism': record.organism,
                'length': record.length
            })
        
        return pd.DataFrame(data)
    
    def get_dataset_statistics(self, records: List[ProteinRecord]) -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            records: List of ProteinRecord objects
            
        Returns:
            Dictionary with dataset statistics
        """
        if not records:
            return {}
        
        sequences = [r.sequence for r in records]
        temperatures = [r.temperature for r in records if r.temperature is not None]
        lengths = [r.length for r in records]
        
        stats = {
            'total_records': len(records),
            'records_with_temperature': len(temperatures),
            'unique_sequences': len(set(sequences)),
            'avg_sequence_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_sequence_length': min(lengths) if lengths else 0,
            'max_sequence_length': max(lengths) if lengths else 0,
        }
        
        if temperatures:
            stats.update({
                'avg_temperature': sum(temperatures) / len(temperatures),
                'min_temperature': min(temperatures),
                'max_temperature': max(temperatures),
            })
        
        return stats
    
    def filter_by_length(self, records: List[ProteinRecord], min_length: int = 10, max_length: int = 1000) -> List[ProteinRecord]:
        """
        Filter records by sequence length.
        
        Args:
            records: List of ProteinRecord objects
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            
        Returns:
            Filtered list of records
        """
        return [r for r in records if min_length <= r.length <= max_length]
    
    def filter_by_temperature(self, records: List[ProteinRecord], min_temp: float = None, max_temp: float = None) -> List[ProteinRecord]:
        """
        Filter records by temperature range.
        
        Args:
            records: List of ProteinRecord objects
            min_temp: Minimum temperature
            max_temp: Maximum temperature
            
        Returns:
            Filtered list of records
        """
        filtered = []
        for record in records:
            if record.temperature is None:
                continue
            
            if min_temp is not None and record.temperature < min_temp:
                continue
            
            if max_temp is not None and record.temperature > max_temp:
                continue
            
            filtered.append(record)
        
        return filtered


def load_supplementary_predictions(data_dir: Union[str, Path] = None) -> pd.DataFrame:
    """
    Load supplementary predictions file.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        DataFrame with prediction data
    """
    data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent / "data"
    supp_file = data_dir / "SupplementaryFileC2EPsPredictions.tsv"
    
    if not supp_file.exists():
        raise FileNotFoundError(f"Supplementary file not found: {supp_file}")
    
    return pd.read_csv(supp_file, sep='\t')


# Convenience functions
def quick_load_training() -> List[ProteinRecord]:
    """Quick function to load training data."""
    with TemStaProReader() as reader:
        return reader.load_training_data()


def quick_load_testing() -> List[ProteinRecord]:
    """Quick function to load testing data."""
    with TemStaProReader() as reader:
        return reader.load_testing_data()


def quick_load_validation() -> List[ProteinRecord]:
    """Quick function to load validation data."""
    with TemStaProReader() as reader:
        return reader.load_validation_data()


# Example usage
if __name__ == "__main__":
    # Example usage of the reader
    with TemStaProReader() as reader:
        # List available files
        files = reader.list_available_files()
        print(f"Found {len(files)} data files:")
        for file in files[:5]:  # Show first 5
            print(f"  - {file.name}")
        
        # Load a small sample for testing
        print("\nLoading sample data...")
        sample_records = []
        for i, record in enumerate(reader.read_dataset("*sample2k*")):
            sample_records.append(record)
            if i >= 10:  # Just load first 10 records for demo
                break
        
        if sample_records:
            print(f"\nLoaded {len(sample_records)} sample records")
            
            # Show statistics
            stats = reader.get_dataset_statistics(sample_records)
            print("\nDataset statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            # Convert to DataFrame
            df = reader.to_dataframe(sample_records)
            print(f"\nDataFrame shape: {df.shape}")
            print(df.head())
        else:
            print("No records loaded - check if data files are present")
