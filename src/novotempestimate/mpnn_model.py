"""
MPNN-based protein temperature prediction model.

This module implements a complete temperature prediction model using ProteinMPNN-inspired
structural encoding with geometric message passing neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .mpnn_encoder import ProteinMPNNEncoder, MPNNConfig
from .model import TemperatureLoss


@dataclass
class MPNNModelConfig:
    """Configuration for MPNN temperature prediction model."""
    # MPNN encoder config
    mpnn_hidden_dim: int = 128
    mpnn_num_layers: int = 3
    mpnn_num_neighbors: int = 32
    mpnn_dropout: float = 0.1
    
    # Regression head config
    regression_hidden_dims: list = None
    regression_dropout: float = 0.3
    
    # Training config
    temperature_range: Tuple[float, float] = (0.0, 100.0)
    loss_type: str = "mse"
    
    def __post_init__(self):
        if self.regression_hidden_dims is None:
            self.regression_hidden_dims = [256, 128, 64]


class ProteinTemperatureMPNN(nn.Module):
    """
    MPNN-based protein temperature prediction model.
    
    Architecture:
    - ProteinMPNN encoder for structural feature extraction
    - Global pooling to aggregate sequence-level features
    - Multi-layer regression head for temperature prediction
    """
    
    def __init__(self, config: MPNNModelConfig, device: Optional[str] = None):
        super().__init__()
        self.config = config
        
        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        
        # Create MPNN encoder config
        mpnn_config = MPNNConfig(
            hidden_dim=config.mpnn_hidden_dim,
            num_encoder_layers=config.mpnn_num_layers,
            num_neighbors=config.mpnn_num_neighbors,
            dropout=config.mpnn_dropout
        )
        
        # MPNN structural encoder
        self.mpnn_encoder = ProteinMPNNEncoder(mpnn_config)
        
        # Global pooling strategies
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Attention-based pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(config.mpnn_hidden_dim, config.mpnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.mpnn_hidden_dim // 2, 1)
        )
        
        # Regression head
        regression_layers = []
        input_dim = config.mpnn_hidden_dim * 3  # avg + max + attention pooling
        
        for hidden_dim in config.regression_hidden_dims:
            regression_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.regression_dropout)
            ])
            input_dim = hidden_dim
        
        # Final temperature prediction layer
        regression_layers.append(nn.Linear(input_dim, 1))
        
        self.regression_head = nn.Sequential(*regression_layers)
        
        # Loss function
        self.loss_fn = TemperatureLoss(
            loss_type=config.loss_type,
            temperature_range=config.temperature_range
        )
        
        # Move to device
        self.to(device)
    
    def forward(self, aa_sequence: torch.Tensor, coordinates: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for temperature prediction.
        
        Args:
            aa_sequence: Amino acid sequence [batch, seq_len]
            coordinates: Backbone coordinates [batch, seq_len, 4, 3] (N, CA, C, O)
            mask: Valid position mask [batch, seq_len]
            
        Returns:
            Predicted temperatures [batch, 1]
        """
        # Encode structural features using MPNN
        encoded_features = self.mpnn_encoder(aa_sequence, coordinates, mask)  # [batch, seq_len, hidden_dim]
        
        # Global pooling to get sequence-level representation
        pooled_features = self._global_pool(encoded_features, mask)  # [batch, hidden_dim * 3]
        
        # Predict temperature
        temperature = self.regression_head(pooled_features)  # [batch, 1]
        
        # Clamp to valid temperature range
        temperature = torch.clamp(
            temperature, 
            self.config.temperature_range[0], 
            self.config.temperature_range[1]
        )
        
        return temperature
    
    def _global_pool(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multiple global pooling strategies and concatenate.
        
        Args:
            features: Sequence features [batch, seq_len, hidden_dim]
            mask: Valid position mask [batch, seq_len]
            
        Returns:
            Pooled features [batch, hidden_dim * 3]
        """
        batch_size, seq_len, hidden_dim = features.shape
        
        if mask is not None:
            # Apply mask to features
            masked_features = features * mask.unsqueeze(-1)
            
            # Average pooling (masked)
            seq_lengths = mask.sum(dim=1, keepdim=True).float()  # [batch, 1]
            avg_pooled = masked_features.sum(dim=1) / seq_lengths  # [batch, hidden_dim]
            
            # Max pooling (masked)
            masked_features_max = masked_features.clone()
            masked_features_max[~mask.unsqueeze(-1).expand_as(features)] = float('-inf')
            max_pooled = masked_features_max.max(dim=1)[0]  # [batch, hidden_dim]
            
            # Attention pooling (masked)
            attention_weights = self.attention_pool(masked_features)  # [batch, seq_len, 1]
            attention_weights = attention_weights.squeeze(-1)  # [batch, seq_len]
            
            # Apply mask to attention weights
            attention_weights = attention_weights.masked_fill(~mask, float('-inf'))
            attention_weights = F.softmax(attention_weights, dim=1)  # [batch, seq_len]
            
            attention_pooled = torch.sum(features * attention_weights.unsqueeze(-1), dim=1)  # [batch, hidden_dim]
        else:
            # Standard pooling without mask
            avg_pooled = features.mean(dim=1)  # [batch, hidden_dim]
            max_pooled = features.max(dim=1)[0]  # [batch, hidden_dim]
            
            # Attention pooling
            attention_weights = self.attention_pool(features)  # [batch, seq_len, 1]
            attention_weights = F.softmax(attention_weights.squeeze(-1), dim=1)  # [batch, seq_len]
            attention_pooled = torch.sum(features * attention_weights.unsqueeze(-1), dim=1)  # [batch, hidden_dim]
        
        # Concatenate all pooling strategies
        pooled = torch.cat([avg_pooled, max_pooled, attention_pooled], dim=1)  # [batch, hidden_dim * 3]
        
        return pooled
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute temperature prediction loss."""
        return self.loss_fn(predictions.squeeze(), targets)
    
    def predict(self, aa_sequence: torch.Tensor, coordinates: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Make temperature predictions (inference mode)."""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(aa_sequence, coordinates, mask)
        return predictions.squeeze()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        mpnn_params = sum(p.numel() for p in self.mpnn_encoder.parameters())
        regression_params = sum(p.numel() for p in self.regression_head.parameters())
        
        return {
            'model_type': 'ProteinMPNN_Temperature',
            'mpnn_hidden_dim': self.config.mpnn_hidden_dim,
            'mpnn_num_layers': self.config.mpnn_num_layers,
            'mpnn_num_neighbors': self.config.mpnn_num_neighbors,
            'regression_layers': len(self.config.regression_hidden_dims),
            'regression_hidden_dims': self.config.regression_hidden_dims,
            'temperature_range': self.config.temperature_range,
            'loss_type': self.config.loss_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'mpnn_parameters': mpnn_params,
            'regression_parameters': regression_params,
            'device': self.device
        }


class SequenceToStructureMPNN(nn.Module):
    """
    MPNN model that predicts structure from sequence, then predicts temperature.
    
    This is useful when only sequence data is available and we need to predict
    3D coordinates first before applying the structural MPNN encoder.
    """
    
    def __init__(self, config: MPNNModelConfig, vocab_size: int = 21, device: Optional[str] = None):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        
        # Sequence encoder for coordinate prediction
        self.sequence_encoder = nn.Sequential(
            nn.Embedding(vocab_size, 256),
            nn.LSTM(256, 256, num_layers=2, batch_first=True, bidirectional=True),
        )
        
        # Coordinate prediction head (predict N, CA, C, O coordinates)
        self.coord_predictor = nn.Sequential(
            nn.Linear(512, 512),  # 256 * 2 (bidirectional)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4 * 3)  # 4 atoms * 3 coordinates
        )
        
        # MPNN temperature prediction model
        self.mpnn_model = ProteinTemperatureMPNN(config, device)
        
        self.to(device)
    
    def forward(self, aa_sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict temperature from sequence via predicted structure.
        
        Args:
            aa_sequence: Amino acid sequence [batch, seq_len]
            mask: Valid position mask [batch, seq_len]
            
        Returns:
            Predicted temperatures [batch, 1]
        """
        batch_size, seq_len = aa_sequence.shape
        
        # Encode sequence
        sequence_features, _ = self.sequence_encoder(aa_sequence)  # [batch, seq_len, 512]
        
        # Predict coordinates
        predicted_coords = self.coord_predictor(sequence_features)  # [batch, seq_len, 12]
        predicted_coords = predicted_coords.view(batch_size, seq_len, 4, 3)  # [batch, seq_len, 4, 3]
        
        # Use MPNN model for temperature prediction
        temperature = self.mpnn_model(aa_sequence, predicted_coords, mask)
        
        return temperature
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        seq_params = sum(p.numel() for p in self.sequence_encoder.parameters())
        coord_params = sum(p.numel() for p in self.coord_predictor.parameters())
        mpnn_info = self.mpnn_model.get_model_info()
        
        return {
            'model_type': 'Sequence_to_Structure_MPNN',
            'vocab_size': self.vocab_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'sequence_encoder_parameters': seq_params,
            'coordinate_predictor_parameters': coord_params,
            'mpnn_model_info': mpnn_info,
            'device': self.device
        }


def create_mpnn_temperature_model(
    mpnn_hidden_dim: int = 128,
    mpnn_num_layers: int = 3,
    mpnn_num_neighbors: int = 32,
    regression_hidden_dims: list = None,
    loss_type: str = "mse",
    device: Optional[str] = None
) -> ProteinTemperatureMPNN:
    """Create an MPNN temperature prediction model with specified configuration."""
    
    if regression_hidden_dims is None:
        regression_hidden_dims = [256, 128, 64]
    
    config = MPNNModelConfig(
        mpnn_hidden_dim=mpnn_hidden_dim,
        mpnn_num_layers=mpnn_num_layers,
        mpnn_num_neighbors=mpnn_num_neighbors,
        regression_hidden_dims=regression_hidden_dims,
        loss_type=loss_type
    )
    
    return ProteinTemperatureMPNN(config, device)


def create_sequence_to_structure_mpnn(
    vocab_size: int = 21,
    mpnn_hidden_dim: int = 128,
    mpnn_num_layers: int = 3,
    device: Optional[str] = None
) -> SequenceToStructureMPNN:
    """Create a sequence-to-structure MPNN model."""
    
    config = MPNNModelConfig(
        mpnn_hidden_dim=mpnn_hidden_dim,
        mpnn_num_layers=mpnn_num_layers
    )
    
    return SequenceToStructureMPNN(config, vocab_size, device)
