"""
NovoTempEstimate: Protein Temperature Estimation using LSTM Networks

A Python package for predicting optimal growth temperatures of proteins
using LSTM neural networks with PyTorch.
"""

__version__ = "0.1.0"
__author__ = "NovoTempEstimate Team"

from .encoding import ProteinEncoder
from .optimization import OptunaLSTMOptimizer
from .data_reader import TemStaProReader, ProteinRecord
from .peptide_encoder import PeptideEncoder, ModifiedResidue
from .model_factory import ModelFactory, ModelManager
from .mpnn_encoder import ProteinMPNNEncoder, MPNNConfig, GeometricFeatures
from .mpnn_model import ProteinTemperatureMPNN, SequenceToStructureMPNN, MPNNModelConfig
from .model import (
    ProteinTemperatureLSTM,
    TemperatureLoss,
    create_model,
    load_model,
    save_model,
)
from .trainer import ProteinTemperatureTrainer, TrainingConfig, ProteinDataset

__all__ = [
    "ProteinEncoder",
    "OptunaLSTMOptimizer",
    "TemStaProReader",
    "ProteinRecord",
    "PeptideEncoder",
    "ModifiedResidue",
    "ProteinTemperatureLSTM",
    "TemperatureLoss",
    "create_model",
    "load_model",
    "save_model",
    "ProteinTemperatureTrainer",
    "TrainingConfig",
    "ProteinDataset",
    "ModelFactory",
    "ModelManager",
    "ProteinMPNNEncoder",
    "MPNNConfig",
    "GeometricFeatures",
    "ProteinTemperatureMPNN",
    "SequenceToStructureMPNN",
    "MPNNModelConfig",
]
