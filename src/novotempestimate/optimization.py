"""
Optuna-based optimization system for LSTM networks.

This module provides functionality to optimize LSTM network architectures
and hyperparameters using Optuna for protein temperature estimation.
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import logging


class LSTMModel(nn.Module):
    """
    LSTM model for protein temperature estimation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        """
        Initialize LSTM model.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Calculate output dimension after LSTM
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, 1),  # Single output for temperature
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use the last output (or mean of all outputs)
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            output = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            output = hidden[-1]

        # Pass through fully connected layers
        output = self.fc_layers(output)

        return output


class OptunaLSTMOptimizer:
    """
    Optuna-based optimizer for LSTM hyperparameters.
    """

    def __init__(
        self,
        input_dim: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "auto",
    ):
        """
        Initialize the optimizer.

        Args:
            input_dim: Input feature dimension
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use for training
        """
        self.input_dim = input_dim
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Set device
        if device == "auto":
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Best trial tracking
        self.best_trial = None
        self.best_model = None

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Validation loss (to minimize)
        """
        # Suggest hyperparameters
        hidden_dim = trial.suggest_int("hidden_dim", 32, 512, step=32)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["Adam", "AdamW", "RMSprop"]
        )

        # Create model
        model = LSTMModel(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        ).to(self.device)

        # Create optimizer
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        else:  # RMSprop
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

        # Loss function
        criterion = nn.MSELoss()

        # Training loop
        num_epochs = 50  # Fixed for optimization
        model.train()

        for epoch in range(num_epochs):
            train_loss = 0.0

            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Early stopping based on validation loss
            if epoch % 10 == 0:
                val_loss = self._evaluate_model(model, criterion)

                # Report intermediate value for pruning
                trial.report(val_loss, epoch)

                # Handle pruning based on the intermediate value
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        # Final validation
        final_val_loss = self._evaluate_model(model, criterion)

        return final_val_loss

    def _evaluate_model(self, model: nn.Module, criterion: nn.Module) -> float:
        """
        Evaluate model on validation set.

        Args:
            model: PyTorch model
            criterion: Loss function

        Returns:
            Validation loss
        """
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()

        model.train()
        return val_loss / len(self.val_loader)

    def optimize(
        self,
        n_trials: int = 100,
        study_name: Optional[str] = None,
        direction: str = "minimize",
    ) -> optuna.Study:
        """
        Run hyperparameter optimization.

        Args:
            n_trials: Number of optimization trials
            study_name: Name for the study
            direction: Optimization direction

        Returns:
            Optuna study object
        """
        # Create study
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        # Optimize
        study.optimize(self.objective, n_trials=n_trials)

        # Store best trial
        self.best_trial = study.best_trial

        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value}")
        print("Best params:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        return study

    def create_best_model(self) -> nn.Module:
        """
        Create model with best hyperparameters.

        Returns:
            Best model
        """
        if self.best_trial is None:
            raise ValueError("No optimization has been run yet.")

        params = self.best_trial.params

        model = LSTMModel(
            input_dim=self.input_dim,
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            bidirectional=params["bidirectional"],
        ).to(self.device)

        self.best_model = model
        return model

    def train_best_model(
        self,
        num_epochs: int = 200,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the best model with full epochs.

        Args:
            num_epochs: Number of training epochs
            save_path: Path to save the trained model

        Returns:
            Training history
        """
        if self.best_model is None:
            self.create_best_model()

        params = self.best_trial.params

        # Create optimizer with best parameters
        if params["optimizer"] == "Adam":
            optimizer = optim.Adam(
                self.best_model.parameters(), lr=params["learning_rate"]
            )
        elif params["optimizer"] == "AdamW":
            optimizer = optim.AdamW(
                self.best_model.parameters(), lr=params["learning_rate"]
            )
        else:
            optimizer = optim.RMSprop(
                self.best_model.parameters(), lr=params["learning_rate"]
            )

        criterion = nn.MSELoss()

        # Training history
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(num_epochs):
            # Training
            self.best_model.train()
            train_loss = 0.0

            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.best_model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            val_loss = self._evaluate_model(self.best_model, criterion)

            history["train_loss"].append(train_loss / len(self.train_loader))
            history["val_loss"].append(val_loss)

            if epoch % 20 == 0:
                print(
                    f"Epoch {epoch}: Train Loss = {train_loss/len(self.train_loader):.4f}, "
                    f"Val Loss = {val_loss:.4f}"
                )

        # Save model if path provided
        if save_path:
            torch.save(
                {
                    "model_state_dict": self.best_model.state_dict(),
                    "hyperparameters": params,
                    "history": history,
                },
                save_path,
            )
            print(f"Model saved to {save_path}")

        return history


# Utility functions for data preparation
def create_dummy_data(
    num_samples: int = 1000,
    seq_length: int = 100,
    input_dim: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create dummy data for testing the optimization system.

    Args:
        num_samples: Number of samples
        seq_length: Sequence length
        input_dim: Input dimension

    Returns:
        Tuple of (sequences, temperatures)
    """
    # Random protein sequences (encoded)
    sequences = torch.randn(num_samples, seq_length, input_dim)

    # Dummy temperatures (50-100°C range)
    temperatures = torch.rand(num_samples) * 50 + 50

    return sequences, temperatures


def test_optimizer():
    """Test the Optuna LSTM optimizer with dummy data."""
    print("Testing Optuna LSTM Optimizer...")

    # Create dummy data
    X, y = create_dummy_data(num_samples=500, seq_length=50, input_dim=20)

    # Split into train/val
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create optimizer
    optimizer = OptunaLSTMOptimizer(
        input_dim=20,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Run optimization (small number of trials for testing)
    study = optimizer.optimize(n_trials=10, study_name="test_lstm_optimization")

    # Create and train best model
    best_model = optimizer.create_best_model()
    history = optimizer.train_best_model(num_epochs=50)

    print("Optimization completed successfully!")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    test_optimizer()
