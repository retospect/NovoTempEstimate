"""
LSTM-based protein temperature prediction model.

This module implements the neural network architecture for predicting
optimal growth temperatures of proteins based on their sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np


class ProteinTemperatureLSTM(nn.Module):
    """
    LSTM-based model for protein temperature prediction.
    
    Architecture:
    - LSTM layers for sequence processing
    - 3 fully connected layers (128 neurons each)
    - Dropout for regularization
    - Single output for temperature prediction
    """
    
    def __init__(
        self,
        vocab_size: int,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        fc_hidden_size: int = 128,
        fc_num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the protein temperature prediction model.
        
        Args:
            vocab_size: Size of the amino acid vocabulary
            lstm_hidden_size: Hidden size for LSTM layers
            lstm_num_layers: Number of LSTM layers
            fc_hidden_size: Hidden size for fully connected layers (128 as per paper)
            fc_num_layers: Number of fully connected layers (3 as per paper)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            device: Device to run the model on
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.fc_hidden_size = fc_hidden_size
        self.fc_num_layers = fc_num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Set device
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # LSTM layers for sequence processing
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output size
        lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)
        
        # Fully connected layers (3 layers of 128 neurons as per paper)
        fc_layers = []
        
        # First FC layer
        fc_layers.append(nn.Linear(lstm_output_size, fc_hidden_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(dropout))
        
        # Hidden FC layers
        for _ in range(fc_num_layers - 2):
            fc_layers.append(nn.Linear(fc_hidden_size, fc_hidden_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
        
        # Output layer
        fc_layers.append(nn.Linear(fc_hidden_size, 1))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Initialize weights
        self._init_weights()
        
        # Move to device
        self.to(self.device)
    
    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # LSTM input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # LSTM hidden-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # LSTM biases
                param.data.fill_(0.)
                # Set forget gate bias to 1
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.)
            elif 'weight' in name and len(param.shape) == 2:
                # FC layer weights
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name and len(param.shape) == 1:
                # FC layer biases
                param.data.fill_(0.)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, vocab_size)
            lengths: Actual sequence lengths for packed sequences
            
        Returns:
            Temperature predictions of shape (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # Pack sequences if lengths are provided
        if lengths is not None:
            # Sort by length for packing
            lengths_sorted, sort_idx = lengths.sort(descending=True)
            x_sorted = x[sort_idx]
            
            # Pack sequences
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
            )
            
            # LSTM forward pass
            lstm_out_packed, (hidden, cell) = self.lstm(x_packed)
            
            # Unpack sequences
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out_packed, batch_first=True
            )
            
            # Restore original order
            _, unsort_idx = sort_idx.sort()
            lstm_out = lstm_out[unsort_idx]
            
            # Use the last valid output for each sequence
            last_outputs = []
            for i, length in enumerate(lengths):
                last_outputs.append(lstm_out[i, length-1])
            sequence_output = torch.stack(last_outputs)
            
        else:
            # Standard forward pass without packing
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Use the last output
            sequence_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        temperature_pred = self.fc_layers(sequence_output)
        
        return temperature_pred
    
    def predict(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Make predictions and return as numpy array.
        
        Args:
            x: Input tensor
            lengths: Sequence lengths
            
        Returns:
            Temperature predictions as numpy array
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x, lengths)
            return predictions.cpu().numpy().flatten()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'vocab_size': self.vocab_size,
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_num_layers': self.lstm_num_layers,
            'fc_hidden_size': self.fc_hidden_size,
            'fc_num_layers': self.fc_num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device)
        }


class TemperatureLoss(nn.Module):
    """
    Custom loss function for temperature prediction.
    
    Combines MSE loss with optional regularization terms.
    """
    
    def __init__(self, loss_type: str = "mse", temperature_range: Tuple[float, float] = (0, 100)):
        """
        Initialize the loss function.
        
        Args:
            loss_type: Type of loss ('mse', 'mae', 'huber')
            temperature_range: Expected temperature range for normalization
        """
        super().__init__()
        self.loss_type = loss_type
        self.temp_min, self.temp_max = temperature_range
        
        if loss_type == "mse":
            self.base_loss = nn.MSELoss()
        elif loss_type == "mae":
            self.base_loss = nn.L1Loss()
        elif loss_type == "huber":
            self.base_loss = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth temperatures
            
        Returns:
            Loss value
        """
        # Basic loss
        loss = self.base_loss(predictions.squeeze(), targets)
        
        # Optional: Add penalty for predictions outside reasonable range
        out_of_range_penalty = 0.0
        if self.temp_min is not None and self.temp_max is not None:
            below_min = torch.relu(self.temp_min - predictions.squeeze())
            above_max = torch.relu(predictions.squeeze() - self.temp_max)
            out_of_range_penalty = torch.mean(below_min + above_max)
        
        return loss + 0.1 * out_of_range_penalty


def create_model(vocab_size: int, **kwargs) -> ProteinTemperatureLSTM:
    """
    Factory function to create a protein temperature prediction model.
    
    Args:
        vocab_size: Size of the amino acid vocabulary
        **kwargs: Additional model parameters
        
    Returns:
        Initialized model
    """
    return ProteinTemperatureLSTM(vocab_size=vocab_size, **kwargs)


def load_model(checkpoint_path: str, vocab_size: int, **kwargs) -> ProteinTemperatureLSTM:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        vocab_size: Size of the amino acid vocabulary
        **kwargs: Additional model parameters
        
    Returns:
        Loaded model
    """
    model = create_model(vocab_size, **kwargs)
    
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def save_model(model: ProteinTemperatureLSTM, checkpoint_path: str, 
               optimizer: Optional[torch.optim.Optimizer] = None,
               epoch: Optional[int] = None,
               loss: Optional[float] = None,
               metrics: Optional[Dict[str, float]] = None):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        checkpoint_path: Path to save the checkpoint
        optimizer: Optimizer state to save
        epoch: Current epoch
        loss: Current loss value
        metrics: Additional metrics to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model.get_model_info(),
        'epoch': epoch,
        'loss': loss,
        'metrics': metrics or {}
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
