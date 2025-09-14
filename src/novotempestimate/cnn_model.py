"""
CNN-based protein temperature prediction model with positional encoding.

This module implements a 4-layer CNN architecture that processes 2D representations
of protein sequences with positional encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import numpy as np

from .positional_encoder import ProteinPositionalEncoder


class ProteinTemperatureCNN(nn.Module):
    """
    CNN-based model for protein temperature prediction.

    Architecture:
    - Positional encoding of protein sequences into 2D grids
    - 4-layer CNN: 1024 -> 512 -> 512 -> 128 channels
    - Global average pooling
    - Final linear layer for temperature prediction
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        grid_size: int = 32,
        cnn_channels: list = [1024, 512, 512, 128],
        kernel_size: int = 3,
        dropout: float = 0.3,
        max_sequence_length: int = 2000,
        device: Optional[str] = None,
    ):
        """
        Initialize the CNN model.

        Args:
            embedding_dim: Dimension of positional embeddings
            grid_size: Size of 2D grid representation
            cnn_channels: List of channel sizes for CNN layers [1024, 512, 512, 128]
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
            max_sequence_length: Maximum sequence length
            device: Device to run model on
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.grid_size = grid_size
        self.cnn_channels = cnn_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length

        # Set device
        if device is None:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Initialize positional encoder
        self.positional_encoder = ProteinPositionalEncoder(
            embedding_dim=embedding_dim, max_sequence_length=max_sequence_length
        )

        # CNN layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input channels = embedding_dim
        in_channels = embedding_dim

        for i, out_channels in enumerate(cnn_channels):
            # Convolutional layer
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            )
            self.conv_layers.append(conv)

            # Batch normalization
            self.batch_norms.append(nn.BatchNorm2d(out_channels))

            # Dropout
            self.dropouts.append(nn.Dropout2d(dropout))

            in_channels = out_channels

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(cnn_channels[-1], cnn_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cnn_channels[-1] // 2, 1),
        )

        # Initialize weights
        self._init_weights()

        # Move to device
        self.to(self.device)

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, sequences: list) -> torch.Tensor:
        """
        Forward pass through the CNN model.

        Args:
            sequences: List of protein sequence strings

        Returns:
            Temperature predictions of shape (batch_size, 1)
        """
        # Convert sequences to 2D grid representations
        batch_2d = self.positional_encoder.create_batch_2d_representation(
            sequences, self.grid_size
        ).to(self.device)

        x = batch_2d  # Shape: (batch_size, embedding_dim, grid_size, grid_size)

        # Pass through CNN layers
        for i, (conv, bn, dropout) in enumerate(
            zip(self.conv_layers, self.batch_norms, self.dropouts)
        ):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)

            # Optional: Add max pooling every 2 layers to reduce spatial dimensions
            if i % 2 == 1 and i < len(self.conv_layers) - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Global average pooling
        x = self.global_avg_pool(x)  # Shape: (batch_size, channels, 1, 1)
        x = x.view(x.size(0), -1)  # Shape: (batch_size, channels)

        # Final classification
        temperature_pred = self.classifier(x)

        return temperature_pred

    def predict(self, sequences: list) -> np.ndarray:
        """
        Make predictions and return as numpy array.

        Args:
            sequences: List of protein sequences

        Returns:
            Temperature predictions as numpy array
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(sequences)
            return predictions.cpu().numpy().flatten()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": "CNN",
            "embedding_dim": self.embedding_dim,
            "grid_size": self.grid_size,
            "cnn_channels": self.cnn_channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "max_sequence_length": self.max_sequence_length,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
        }


class ProteinTemperatureCNNAdvanced(nn.Module):
    """
    Advanced CNN model with residual connections and attention.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        grid_size: int = 32,
        cnn_channels: list = [1024, 512, 512, 128],
        kernel_size: int = 3,
        dropout: float = 0.3,
        max_sequence_length: int = 2000,
        use_residual: bool = True,
        use_attention: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize the advanced CNN model.

        Args:
            embedding_dim: Dimension of positional embeddings
            grid_size: Size of 2D grid representation
            cnn_channels: List of channel sizes for CNN layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
            max_sequence_length: Maximum sequence length
            use_residual: Whether to use residual connections
            use_attention: Whether to use attention mechanism
            device: Device to run model on
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.grid_size = grid_size
        self.cnn_channels = cnn_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.use_residual = use_residual
        self.use_attention = use_attention

        # Set device
        if device is None:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Initialize positional encoder
        self.positional_encoder = ProteinPositionalEncoder(
            embedding_dim=embedding_dim, max_sequence_length=max_sequence_length
        )

        # Build CNN blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = embedding_dim

        for i, out_channels in enumerate(cnn_channels):
            block = self._make_conv_block(
                in_channels, out_channels, kernel_size, dropout
            )
            self.conv_blocks.append(block)
            in_channels = out_channels

        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(cnn_channels[-1], cnn_channels[-1] // 4, 1),
                nn.ReLU(),
                nn.Conv2d(cnn_channels[-1] // 4, 1, 1),
                nn.Sigmoid(),
            )

        # Global pooling and classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(cnn_channels[-1], cnn_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cnn_channels[-1] // 2, cnn_channels[-1] // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cnn_channels[-1] // 4, 1),
        )

        # Initialize weights
        self._init_weights()

        # Move to device
        self.to(self.device)

    def _make_conv_block(
        self, in_channels: int, out_channels: int, kernel_size: int, dropout: float
    ):
        """Create a convolutional block."""
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        ]

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, sequences: list) -> torch.Tensor:
        """Forward pass through the advanced CNN model."""
        # Convert sequences to 2D grid representations
        batch_2d = self.positional_encoder.create_batch_2d_representation(
            sequences, self.grid_size
        ).to(self.device)

        x = batch_2d

        # Pass through CNN blocks with optional residual connections
        for i, block in enumerate(self.conv_blocks):
            identity = x
            x = block(x)

            # Residual connection (if dimensions match)
            if self.use_residual and identity.shape == x.shape:
                x = x + identity

            # Optional pooling
            if i % 2 == 1 and i < len(self.conv_blocks) - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Apply attention if enabled
        if self.use_attention:
            attention_weights = self.attention(x)
            x = x * attention_weights

        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Final classification
        temperature_pred = self.classifier(x)

        return temperature_pred

    def predict(self, sequences: list) -> np.ndarray:
        """Make predictions and return as numpy array."""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(sequences)
            return predictions.cpu().numpy().flatten()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": "CNN_Advanced",
            "embedding_dim": self.embedding_dim,
            "grid_size": self.grid_size,
            "cnn_channels": self.cnn_channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "max_sequence_length": self.max_sequence_length,
            "use_residual": self.use_residual,
            "use_attention": self.use_attention,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
        }


def create_cnn_model(model_type: str = "basic", **kwargs) -> nn.Module:
    """
    Factory function to create CNN models.

    Args:
        model_type: Type of CNN model ("basic" or "advanced")
        **kwargs: Additional model parameters

    Returns:
        Initialized CNN model
    """
    if model_type == "basic":
        return ProteinTemperatureCNN(**kwargs)
    elif model_type == "advanced":
        return ProteinTemperatureCNNAdvanced(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_cnn_model(
    checkpoint_path: str, model_type: str = "basic", **kwargs
) -> nn.Module:
    """
    Load a trained CNN model from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint
        model_type: Type of CNN model
        **kwargs: Additional model parameters

    Returns:
        Loaded model
    """
    model = create_cnn_model(model_type, **kwargs)

    checkpoint = torch.load(checkpoint_path, map_location=model.device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model
