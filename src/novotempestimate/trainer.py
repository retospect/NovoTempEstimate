"""
Training system for protein temperature prediction model.

This module implements the training pipeline with best practices including
data loading, validation, early stopping, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from tqdm import tqdm

from .model import ProteinTemperatureLSTM, TemperatureLoss, save_model
from .data_reader import TemStaProReader, ProteinRecord
from .peptide_encoder import PeptideEncoder


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model parameters
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    fc_hidden_size: int = 128
    fc_num_layers: int = 3
    dropout: float = 0.3
    bidirectional: bool = True
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    weight_decay: float = 1e-5
    
    # Data parameters
    max_sequence_length: int = 1000
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Training behavior
    early_stopping_patience: int = 10
    save_best_model: bool = True
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5
    
    # Loss function
    loss_type: str = "mse"
    temperature_range: Tuple[float, float] = (0, 100)
    
    # Optimization
    optimizer_type: str = "adam"
    scheduler_type: str = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Logging
    log_frequency: int = 100
    validate_frequency: int = 1


class ProteinDataset(Dataset):
    """Dataset for protein sequences and temperatures."""
    
    def __init__(self, records: List[ProteinRecord], encoder: PeptideEncoder, 
                 max_length: Optional[int] = None):
        """
        Initialize the dataset.
        
        Args:
            records: List of protein records
            encoder: Peptide encoder for sequence encoding
            max_length: Maximum sequence length (sequences will be truncated)
        """
        self.records = records
        self.encoder = encoder
        self.max_length = max_length
        
        # Pre-process data
        self.sequences = []
        self.temperatures = []
        self.lengths = []
        
        for record in records:
            sequence = record.sequence
            if max_length and len(sequence) > max_length:
                sequence = sequence[:max_length]
            
            self.sequences.append(sequence)
            self.temperatures.append(float(record.temperature))
            self.lengths.append(len(sequence))
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a single item from the dataset.
        
        Returns:
            Tuple of (encoded_sequence, temperature, sequence_length)
        """
        sequence = self.sequences[idx]
        temperature = self.temperatures[idx]
        length = self.lengths[idx]
        
        # Encode sequence
        encoded = self.encoder.encode_sequence(sequence)
        
        return encoded, torch.tensor(temperature, dtype=torch.float32), length


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader to handle variable-length sequences.
    
    Args:
        batch: List of (sequence, temperature, length) tuples
        
    Returns:
        Tuple of (padded_sequences, temperatures, lengths)
    """
    sequences, temperatures, lengths = zip(*batch)
    
    # Get maximum length in batch
    max_len = max(lengths)
    vocab_size = sequences[0].size(1)
    
    # Pad sequences
    padded_sequences = torch.zeros(len(sequences), max_len, vocab_size)
    for i, seq in enumerate(sequences):
        seq_len = seq.size(0)
        padded_sequences[i, :seq_len] = seq
    
    temperatures = torch.stack(temperatures)
    lengths = torch.tensor(lengths)
    
    return padded_sequences, temperatures, lengths


class ProteinTemperatureTrainer:
    """Trainer for protein temperature prediction model."""
    
    def __init__(self, config: TrainingConfig, output_dir: str = "models"):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            output_dir: Directory to save models and logs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.encoder = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rate': []
        }
        
        # Device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "training.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, data_pattern: str = "*training*", sample_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load and prepare data for training.
        
        Args:
            data_pattern: Pattern to match data files
            sample_size: Optional sample size for testing
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        self.logger.info("Loading protein data...")
        
        # Initialize encoder
        self.encoder = PeptideEncoder(include_modifications=True)
        
        # Load data
        records = []
        with TemStaProReader() as reader:
            for record in reader.read_dataset(data_pattern):
                # Filter valid records
                if (record.temperature is not None and record.temperature > 0 and 
                    len(record.sequence) > 10):
                    records.append(record)
                
                if sample_size and len(records) >= sample_size:
                    break
        
        self.logger.info(f"Loaded {len(records)} protein records")
        
        # Create dataset
        dataset = ProteinDataset(records, self.encoder, self.config.max_sequence_length)
        
        # Split data
        total_size = len(dataset)
        train_size = int(self.config.train_split * total_size)
        val_size = int(self.config.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        self.logger.info(f"Data split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, collate_fn=collate_fn, num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=0
        )
        
        return train_loader, val_loader, test_loader
    
    def setup_model(self):
        """Setup model, optimizer, scheduler, and loss function."""
        if self.encoder is None:
            raise ValueError("Encoder not initialized. Call load_data first.")
        
        # Create model
        self.model = ProteinTemperatureLSTM(
            vocab_size=self.encoder.vocab_size,
            lstm_hidden_size=self.config.lstm_hidden_size,
            lstm_num_layers=self.config.lstm_num_layers,
            fc_hidden_size=self.config.fc_hidden_size,
            fc_num_layers=self.config.fc_num_layers,
            dropout=self.config.dropout,
            bidirectional=self.config.bidirectional,
            device=self.device
        )
        
        self.logger.info(f"Model info: {self.model.get_model_info()}")
        
        # Setup optimizer
        if self.config.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")
        
        # Setup scheduler
        if self.config.scheduler_type == "reduce_on_plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        
        # Setup loss function
        self.criterion = TemperatureLoss(
            loss_type=self.config.loss_type,
            temperature_range=self.config.temperature_range
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (sequences, temperatures, lengths) in enumerate(progress_bar):
            sequences = sequences.to(self.device)
            temperatures = temperatures.to(self.device)
            lengths = lengths.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(sequences, lengths)
            loss = self.criterion(predictions, temperatures)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate metrics
            mae = torch.mean(torch.abs(predictions.squeeze() - temperatures))
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'MAE': f"{mae.item():.2f}°C"
            })
            
            # Log batch metrics
            if batch_idx % self.config.log_frequency == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                    f"Loss={loss.item():.4f}, MAE={mae.item():.2f}°C"
                )
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for sequences, temperatures, lengths in val_loader:
                sequences = sequences.to(self.device)
                temperatures = temperatures.to(self.device)
                lengths = lengths.to(self.device)
                
                predictions = self.model(sequences, lengths)
                loss = self.criterion(predictions, temperatures)
                mae = torch.mean(torch.abs(predictions.squeeze() - temperatures))
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        
        return avg_loss, avg_mae
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history and final metrics
        """
        self.logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_loss, train_mae = self.train_epoch(train_loader)
            
            # Validate
            if epoch % self.config.validate_frequency == 0:
                val_loss, val_mae = self.validate(val_loader)
                
                # Update learning rate
                if self.scheduler:
                    if self.config.scheduler_type == "reduce_on_plateau":
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Record history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['train_mae'].append(train_mae)
                self.training_history['val_mae'].append(val_mae)
                self.training_history['learning_rate'].append(
                    self.optimizer.param_groups[0]['lr']
                )
                
                # Log metrics
                self.logger.info(
                    f"Epoch {self.current_epoch}: "
                    f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                    f"Train MAE={train_mae:.2f}°C, Val MAE={val_mae:.2f}°C, "
                    f"LR={self.optimizer.param_groups[0]['lr']:.6f}"
                )
                
                # Early stopping and model saving
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stopping_counter = 0
                    
                    if self.config.save_best_model:
                        best_model_path = self.output_dir / "best_model.pth"
                        save_model(
                            self.model, best_model_path, self.optimizer,
                            epoch, val_loss, {'val_mae': val_mae}
                        )
                        self.logger.info(f"Saved best model with val_loss={val_loss:.4f}")
                else:
                    self.early_stopping_counter += 1
                
                # Check early stopping
                if self.early_stopping_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {self.current_epoch}")
                    break
            
            # Save checkpoint
            if (self.config.save_checkpoints and 
                epoch % self.config.checkpoint_frequency == 0):
                checkpoint_path = self.output_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
                save_model(
                    self.model, checkpoint_path, self.optimizer,
                    epoch, train_loss
                )
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        final_model_path = self.output_dir / "final_model.pth"
        save_model(self.model, final_model_path, self.optimizer, self.current_epoch)
        
        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        return {
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time,
            'final_epoch': self.current_epoch
        }


def create_trainer(config_dict: Optional[Dict[str, Any]] = None, 
                  output_dir: str = "models") -> ProteinTemperatureTrainer:
    """
    Factory function to create a trainer.
    
    Args:
        config_dict: Configuration dictionary
        output_dir: Output directory for models
        
    Returns:
        Initialized trainer
    """
    if config_dict:
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
    
    return ProteinTemperatureTrainer(config, output_dir)
