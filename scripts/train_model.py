#!/usr/bin/env python3
"""
Training script for protein temperature prediction model.

This script provides a command-line interface for training the LSTM-based
protein temperature prediction model with configurable parameters.

Usage:
    python scripts/train_model.py [options]

Example:
    python scripts/train_model.py --epochs 50 --batch-size 64 --sample-size 10000
"""

import argparse
import sys
from pathlib import Path
import json
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from novotempestimate.trainer import ProteinTemperatureTrainer, TrainingConfig


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train protein temperature prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument("--lstm-hidden-size", type=int, default=128,
                            help="LSTM hidden layer size")
    model_group.add_argument("--lstm-layers", type=int, default=2,
                            help="Number of LSTM layers")
    model_group.add_argument("--fc-hidden-size", type=int, default=128,
                            help="Fully connected layer size (as per paper)")
    model_group.add_argument("--fc-layers", type=int, default=3,
                            help="Number of FC layers (as per paper)")
    model_group.add_argument("--dropout", type=float, default=0.3,
                            help="Dropout probability")
    model_group.add_argument("--no-bidirectional", action="store_true",
                            help="Disable bidirectional LSTM")
    
    # Training parameters
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument("--batch-size", type=int, default=32,
                            help="Batch size for training")
    train_group.add_argument("--learning-rate", type=float, default=0.001,
                            help="Learning rate")
    train_group.add_argument("--epochs", type=int, default=100,
                            help="Number of training epochs")
    train_group.add_argument("--weight-decay", type=float, default=1e-5,
                            help="Weight decay for regularization")
    
    # Data parameters
    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument("--max-length", type=int, default=1000,
                           help="Maximum sequence length")
    data_group.add_argument("--train-split", type=float, default=0.8,
                           help="Training data split ratio")
    data_group.add_argument("--val-split", type=float, default=0.1,
                           help="Validation data split ratio")
    data_group.add_argument("--sample-size", type=int, default=None,
                           help="Sample size for testing (use subset of data)")
    data_group.add_argument("--data-pattern", type=str, default="*training*",
                           help="Pattern to match data files")
    
    # Training behavior
    behavior_group = parser.add_argument_group("Training Behavior")
    behavior_group.add_argument("--early-stopping-patience", type=int, default=10,
                               help="Early stopping patience")
    behavior_group.add_argument("--no-save-best", action="store_true",
                               help="Don't save best model")
    behavior_group.add_argument("--no-checkpoints", action="store_true",
                               help="Don't save checkpoints")
    behavior_group.add_argument("--checkpoint-freq", type=int, default=5,
                               help="Checkpoint saving frequency")
    
    # Optimization
    opt_group = parser.add_argument_group("Optimization")
    opt_group.add_argument("--optimizer", type=str, default="adam",
                          choices=["adam", "adamw"],
                          help="Optimizer type")
    opt_group.add_argument("--scheduler", type=str, default="reduce_on_plateau",
                          choices=["reduce_on_plateau", "cosine", "none"],
                          help="Learning rate scheduler")
    opt_group.add_argument("--scheduler-patience", type=int, default=5,
                          help="Scheduler patience")
    opt_group.add_argument("--scheduler-factor", type=float, default=0.5,
                          help="Scheduler reduction factor")
    
    # Loss function
    loss_group = parser.add_argument_group("Loss Function")
    loss_group.add_argument("--loss-type", type=str, default="mse",
                           choices=["mse", "mae", "huber"],
                           help="Loss function type")
    
    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output-dir", type=str, default="models",
                             help="Output directory for models and logs")
    output_group.add_argument("--log-freq", type=int, default=100,
                             help="Logging frequency (batches)")
    output_group.add_argument("--val-freq", type=int, default=1,
                             help="Validation frequency (epochs)")
    
    # Miscellaneous
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--config", type=str, default=None,
                           help="Load configuration from JSON file")
    misc_group.add_argument("--save-config", type=str, default=None,
                           help="Save configuration to JSON file")
    misc_group.add_argument("--device", type=str, default=None,
                           choices=["cpu", "mps", "cuda"],
                           help="Device to use for training")
    misc_group.add_argument("--seed", type=int, default=42,
                           help="Random seed for reproducibility")
    
    return parser.parse_args()


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config_to_file(config: TrainingConfig, config_path: str):
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)


def create_config_from_args(args) -> TrainingConfig:
    """Create training configuration from command line arguments."""
    config = TrainingConfig(
        # Model parameters
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_layers,
        fc_hidden_size=args.fc_hidden_size,
        fc_num_layers=args.fc_layers,
        dropout=args.dropout,
        bidirectional=not args.no_bidirectional,
        
        # Training parameters
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        weight_decay=args.weight_decay,
        
        # Data parameters
        max_sequence_length=args.max_length,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=1.0 - args.train_split - args.val_split,
        
        # Training behavior
        early_stopping_patience=args.early_stopping_patience,
        save_best_model=not args.no_save_best,
        save_checkpoints=not args.no_checkpoints,
        checkpoint_frequency=args.checkpoint_freq,
        
        # Loss function
        loss_type=args.loss_type,
        
        # Optimization
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler if args.scheduler != "none" else None,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        
        # Logging
        log_frequency=args.log_freq,
        validate_frequency=args.val_freq
    )
    
    return config


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load or create configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        config_dict = load_config_from_file(args.config)
        config = TrainingConfig(**config_dict)
    else:
        config = create_config_from_args(args)
    
    # Save configuration if requested
    if args.save_config:
        save_config_to_file(config, args.save_config)
        print(f"Configuration saved to {args.save_config}")
    
    # Print configuration
    print("Training Configuration:")
    print("=" * 50)
    for key, value in config.__dict__.items():
        print(f"{key:25}: {value}")
    print("=" * 50)
    
    # Check device availability
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) acceleration available")
    elif torch.cuda.is_available():
        print("✅ CUDA acceleration available")
    else:
        print("⚠️  No GPU acceleration available, using CPU")
    
    # Create trainer
    trainer = ProteinTemperatureTrainer(config, args.output_dir)
    
    try:
        # Load data
        print("\n🔄 Loading data...")
        train_loader, val_loader, test_loader = trainer.load_data(
            data_pattern=args.data_pattern,
            sample_size=args.sample_size
        )
        
        # Setup model
        print("🔄 Setting up model...")
        trainer.setup_model()
        
        # Start training
        print("🚀 Starting training...")
        results = trainer.train(train_loader, val_loader)
        
        # Print results
        print("\n" + "=" * 50)
        print("Training Results:")
        print(f"Best validation loss: {results['best_val_loss']:.4f}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Final epoch: {results['final_epoch']}")
        print("=" * 50)
        
        # Evaluate on test set
        print("\n🔄 Evaluating on test set...")
        test_loss, test_mae = trainer.validate(test_loader)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.2f}°C")
        
        print(f"\n✅ Training completed! Models saved to: {args.output_dir}")
        print(f"📊 Training history saved to: {args.output_dir}/training_history.json")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
