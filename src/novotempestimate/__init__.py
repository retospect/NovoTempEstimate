"""
NovoTempEstimate: Protein temperature estimation using LSTM networks.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .encoding import ProteinEncoder
from .optimization import OptunaLSTMOptimizer
from .data_reader import TemStaProReader, ProteinRecord, quick_load_training, quick_load_testing, quick_load_validation
from .peptide_encoder import PeptideEncoder, ModifiedResidue, ModificationType, encode_peptide, encode_peptide_batch

__all__ = ["ProteinEncoder", "OptunaLSTMOptimizer", "TemStaProReader", "ProteinRecord", "quick_load_training", "quick_load_testing", "quick_load_validation", "PeptideEncoder", "ModifiedResidue", "ModificationType", "encode_peptide", "encode_peptide_batch"]
