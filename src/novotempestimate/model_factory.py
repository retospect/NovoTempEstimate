"""
Model factory for switching between different protein temperature prediction architectures.

This module provides a unified interface for creating and managing different
model architectures (LSTM, CNN, etc.) for protein temperature prediction.
"""

from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
from enum import Enum

from .model import ProteinTemperatureLSTM
from .cnn_model import ProteinTemperatureCNN, ProteinTemperatureCNNAdvanced
from .mpnn_model import ProteinTemperatureMPNN, SequenceToStructureMPNN, MPNNModelConfig
from .positional_encoder import ProteinPositionalEncoder
from .peptide_encoder import PeptideEncoder


class ModelType(Enum):
    """Enumeration of available model types."""

    LSTM = "lstm"
    CNN_BASIC = "cnn_basic"
    CNN_ADVANCED = "cnn_advanced"
    MPNN = "mpnn"
    MPNN_SEQ2STRUCT = "mpnn_seq2struct"


class ModelFactory:
    """Factory class for creating different protein temperature prediction models."""

    @staticmethod
    def create_model(
        model_type: Union[str, ModelType],
        model_config: Dict[str, Any],
        device: Optional[str] = None,
    ) -> nn.Module:
        """
        Create a model based on the specified type and configuration.

        Args:
            model_type: Type of model to create
            model_config: Configuration parameters for the model
            device: Device to run the model on

        Returns:
            Initialized model
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type.lower())

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        if model_type == ModelType.LSTM:
            return ModelFactory._create_lstm_model(model_config, device)
        elif model_type == ModelType.CNN_BASIC:
            return ModelFactory._create_cnn_basic_model(model_config, device)
        elif model_type == ModelType.CNN_ADVANCED:
            return ModelFactory._create_cnn_advanced_model(model_config, device)
        elif model_type == ModelType.MPNN:
            return ModelFactory._create_mpnn_model(model_config, device)
        elif model_type == ModelType.MPNN_SEQ2STRUCT:
            return ModelFactory._create_mpnn_seq2struct_model(model_config, device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def _create_lstm_model(
        config: Dict[str, Any], device: str
    ) -> ProteinTemperatureLSTM:
        """Create LSTM model."""
        # Default LSTM configuration
        default_config = {
            "vocab_size": 85,  # Will be updated based on encoder
            "lstm_hidden_size": 128,
            "lstm_num_layers": 2,
            "fc_hidden_size": 128,
            "fc_num_layers": 3,
            "dropout": 0.3,
            "bidirectional": True,
        }

        # Update with provided config
        default_config.update(config)

        # Create peptide encoder to get vocab size
        encoder = PeptideEncoder(include_modifications=True)
        default_config["vocab_size"] = encoder.vocab_size

        return ProteinTemperatureLSTM(device=device, **default_config)

    @staticmethod
    def _create_cnn_basic_model(
        config: Dict[str, Any], device: str
    ) -> ProteinTemperatureCNN:
        """Create basic CNN model."""
        # Default CNN configuration
        default_config = {
            "embedding_dim": 64,
            "grid_size": 32,
            "cnn_channels": [1024, 512, 512, 128],
            "kernel_size": 3,
            "dropout": 0.3,
            "max_sequence_length": 2000,
        }

        # Update with provided config
        default_config.update(config)

        return ProteinTemperatureCNN(device=device, **default_config)

    @staticmethod
    def _create_cnn_advanced_model(
        config: Dict[str, Any], device: str
    ) -> ProteinTemperatureCNNAdvanced:
        """Create advanced CNN model."""
        # Default advanced CNN configuration
        default_config = {
            "embedding_dim": 64,
            "grid_size": 32,
            "cnn_channels": [1024, 512, 512, 128],
            "kernel_size": 3,
            "dropout": 0.3,
            "max_sequence_length": 2000,
            "use_residual": True,
            "use_attention": True,
        }

        # Update with provided config
        default_config.update(config)

        return ProteinTemperatureCNNAdvanced(device=device, **default_config)

    @staticmethod
    def _create_mpnn_model(
        config: Dict[str, Any], device: str
    ) -> ProteinTemperatureMPNN:
        """Create MPNN model."""
        # Default MPNN configuration
        default_config = {
            "mpnn_hidden_dim": 128,
            "mpnn_num_layers": 3,
            "mpnn_num_neighbors": 32,
            "regression_hidden_dims": [256, 128, 64],
            "loss_type": "mse",
        }

        # Update with provided config
        default_config.update(config)

        mpnn_config = MPNNModelConfig(**default_config)
        return ProteinTemperatureMPNN(mpnn_config, device)

    @staticmethod
    def _create_mpnn_seq2struct_model(
        config: Dict[str, Any], device: str
    ) -> SequenceToStructureMPNN:
        """Create sequence-to-structure MPNN model."""
        # Default configuration
        default_config = {
            "mpnn_hidden_dim": 128,
            "mpnn_num_layers": 3,
            "mpnn_num_neighbors": 32,
            "regression_hidden_dims": [256, 128, 64],
            "loss_type": "mse",
            "vocab_size": 21,
        }

        # Update with provided config
        default_config.update(config)

        vocab_size = default_config.pop("vocab_size")
        mpnn_config = MPNNModelConfig(**default_config)
        return SequenceToStructureMPNN(mpnn_config, vocab_size, device)

    @staticmethod
    def get_model_configs() -> Dict[str, Dict[str, Any]]:
        """Get default configurations for all model types."""
        return {
            "lstm": {
                "lstm_hidden_size": 128,
                "lstm_num_layers": 2,
                "fc_hidden_size": 128,
                "fc_num_layers": 3,
                "dropout": 0.3,
                "bidirectional": True,
            },
            "cnn_basic": {
                "embedding_dim": 64,
                "grid_size": 32,
                "cnn_channels": [1024, 512, 512, 128],
                "kernel_size": 3,
                "dropout": 0.3,
                "max_sequence_length": 2000,
            },
            "cnn_advanced": {
                "embedding_dim": 64,
                "grid_size": 32,
                "cnn_channels": [1024, 512, 512, 128],
                "kernel_size": 3,
                "dropout": 0.3,
                "max_sequence_length": 2000,
                "use_residual": True,
                "use_attention": True,
            },
        }

    @staticmethod
    def load_model(
        checkpoint_path: str,
        model_type: Union[str, ModelType],
        model_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ) -> nn.Module:
        """
        Load a trained model from checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint
            model_type: Type of model to load
            model_config: Configuration parameters (if None, will try to load from checkpoint)
            device: Device to run the model on

        Returns:
            Loaded model
        """
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Try to get config from checkpoint
        if model_config is None and "model_config" in checkpoint:
            saved_config = checkpoint["model_config"]
            model_type_from_checkpoint = saved_config.get("model_type", model_type)

            if isinstance(model_type_from_checkpoint, str):
                model_type = ModelType(model_type_from_checkpoint.lower())

            model_config = saved_config

        # Create model
        if model_config is None:
            model_config = ModelFactory.get_model_configs()[model_type.value]

        model = ModelFactory.create_model(model_type, model_config, device)

        # Load state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        return model

    @staticmethod
    def save_model(
        model: nn.Module,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        model_type: Optional[str] = None,
    ):
        """
        Save model checkpoint with metadata.

        Args:
            model: Model to save
            checkpoint_path: Path to save the checkpoint
            optimizer: Optimizer state to save
            epoch: Current epoch
            loss: Current loss value
            metrics: Additional metrics to save
            model_type: Type of model being saved
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "metrics": metrics or {},
        }

        # Add model config if available
        if hasattr(model, "get_model_info"):
            model_info = model.get_model_info()
            if model_type:
                model_info["model_type"] = model_type
            checkpoint["model_config"] = model_info

        # Add optimizer state if provided
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)


class ModelManager:
    """Manager class for handling multiple models and comparisons."""

    def __init__(self):
        """Initialize the model manager."""
        self.models = {}
        self.model_configs = {}
        self.model_types = {}

    def add_model(
        self,
        name: str,
        model_type: Union[str, ModelType],
        model_config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        """Add a new model to the manager."""
        model = ModelFactory.create_model(model_type, model_config, device)
        self.models[name] = model
        self.model_configs[name] = model_config
        self.model_types[name] = model_type

    def load_model(
        self,
        name: str,
        checkpoint_path: str,
        model_type: Union[str, ModelType],
        model_config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        """Load a model from checkpoint and add to manager."""
        model = ModelFactory.load_model(
            checkpoint_path, model_type, model_config, device
        )
        self.models[name] = model
        self.model_types[name] = model_type

        if hasattr(model, "get_model_info"):
            self.model_configs[name] = model.get_model_info()

    def get_model(self, name: str) -> nn.Module:
        """Get a model by name."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found")
        return self.models[name]

    def list_models(self) -> Dict[str, str]:
        """List all available models."""
        return {name: str(model_type) for name, model_type in self.model_types.items()}

    def compare_models(self) -> Dict[str, Dict[str, Any]]:
        """Compare all models in the manager."""
        comparison = {}

        for name, model in self.models.items():
            if hasattr(model, "get_model_info"):
                comparison[name] = model.get_model_info()
            else:
                comparison[name] = {
                    "model_type": str(self.model_types[name]),
                    "total_parameters": sum(p.numel() for p in model.parameters()),
                }

        return comparison

    def remove_model(self, name: str):
        """Remove a model from the manager."""
        if name in self.models:
            del self.models[name]
            del self.model_configs[name]
            del self.model_types[name]


# Convenience functions
def create_lstm_model(
    config: Optional[Dict[str, Any]] = None, device: Optional[str] = None
) -> ProteinTemperatureLSTM:
    """Create LSTM model with default or custom configuration."""
    config = config or {}
    return ModelFactory.create_model(ModelType.LSTM, config, device)


def create_cnn_model(
    config: Optional[Dict[str, Any]] = None, device: Optional[str] = None
) -> ProteinTemperatureCNN:
    """Create basic CNN model with default or custom configuration."""
    config = config or {}
    return ModelFactory.create_model(ModelType.CNN_BASIC, config, device)


def create_advanced_cnn_model(
    config: Optional[Dict[str, Any]] = None, device: Optional[str] = None
) -> ProteinTemperatureCNNAdvanced:
    """Create advanced CNN model with default or custom configuration."""
    config = config or {}
    return ModelFactory.create_model(ModelType.CNN_ADVANCED, config, device)
