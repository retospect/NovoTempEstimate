#!/usr/bin/env python3
"""
Unified training script for protein temperature prediction models.

This script provides a single interface for training both LSTM and CNN models
with automatic model selection and configuration.

Usage:
    python scripts/train_unified.py --model-type lstm [lstm-options]
    python scripts/train_unified.py --model-type cnn [cnn-options]

Examples:
    # Train LSTM model
    python scripts/train_unified.py --model-type lstm --epochs 50 --batch-size 32

    # Train CNN model with positional encoding
    python scripts/train_unified.py --model-type cnn --epochs 50 --batch-size 16

    # Compare both models (train sequentially)
    python scripts/train_unified.py --model-type both --epochs 20
"""

import argparse
import sys
from pathlib import Path
import json
import torch
import time
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from novotempestimate.trainer import ProteinTemperatureTrainer, TrainingConfig
from novotempestimate.cnn_trainer import CNNProteinTemperatureTrainer, CNNTrainingConfig
from novotempestimate.model_factory import ModelFactory, ModelManager


def parse_arguments():
    """Parse command line arguments for unified training."""
    parser = argparse.ArgumentParser(
        description="Unified protein temperature prediction model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["lstm", "cnn", "both"],
        help="Model architecture to train",
    )
    model_group.add_argument(
        "--cnn-variant",
        type=str,
        default="cnn_basic",
        choices=["cnn_basic", "cnn_advanced"],
        help="CNN variant (only for CNN models)",
    )

    # Common training parameters
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (auto-selected if not specified)",
    )
    train_group.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    train_group.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for regularization",
    )

    # Data parameters
    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for testing (use subset of data)",
    )
    data_group.add_argument(
        "--train-split", type=float, default=0.8, help="Training data split ratio"
    )
    data_group.add_argument(
        "--val-split", type=float, default=0.1, help="Validation data split ratio"
    )

    # Model-specific parameters
    lstm_group = parser.add_argument_group("LSTM Parameters")
    lstm_group.add_argument(
        "--lstm-hidden-size", type=int, default=128, help="LSTM hidden size"
    )
    lstm_group.add_argument(
        "--lstm-layers", type=int, default=2, help="Number of LSTM layers"
    )
    lstm_group.add_argument(
        "--fc-layers", type=int, default=3, help="Number of fully connected layers"
    )

    cnn_group = parser.add_argument_group("CNN Parameters")
    cnn_group.add_argument(
        "--embedding-dim", type=int, default=64, help="Positional embedding dimension"
    )
    cnn_group.add_argument(
        "--grid-size", type=int, default=32, help="2D grid size for CNN input"
    )
    cnn_group.add_argument(
        "--cnn-channels",
        type=int,
        nargs="+",
        default=[1024, 512, 512, 128],
        help="CNN channel sizes",
    )

    # Training behavior
    behavior_group = parser.add_argument_group("Training Behavior")
    behavior_group.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    behavior_group.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout probability"
    )

    # Output and comparison
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="unified_models",
        help="Output directory for models",
    )
    output_group.add_argument(
        "--compare-models", action="store_true", help="Generate model comparison report"
    )
    output_group.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save model predictions for comparison",
    )

    # Miscellaneous
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="Device to use for training",
    )
    misc_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    misc_group.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def create_lstm_config(args) -> TrainingConfig:
    """Create LSTM training configuration from arguments."""
    batch_size = args.batch_size if args.batch_size else 32

    config = TrainingConfig(
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_layers,
        fc_hidden_size=128,
        fc_num_layers=args.fc_layers,
        dropout=args.dropout,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=1.0 - args.train_split - args.val_split,
    )

    return config


def create_cnn_config(args) -> CNNTrainingConfig:
    """Create CNN training configuration from arguments."""
    batch_size = args.batch_size if args.batch_size else 16  # Smaller default for CNN

    config = CNNTrainingConfig(
        model_type=args.cnn_variant,
        embedding_dim=args.embedding_dim,
        grid_size=args.grid_size,
        cnn_channels=args.cnn_channels,
        dropout=args.dropout,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=1.0 - args.train_split - args.val_split,
    )

    return config


def train_lstm_model(args, output_dir: Path) -> Dict[str, Any]:
    """Train LSTM model and return results."""
    print("🧠 Training LSTM Model")
    print("=" * 60)

    config = create_lstm_config(args)
    lstm_output_dir = output_dir / "lstm"
    lstm_output_dir.mkdir(exist_ok=True)

    # Print LSTM configuration
    print("LSTM Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key:25}: {value}")
    print()

    trainer = ProteinTemperatureTrainer(config, str(lstm_output_dir))

    # Load data and train
    print("🔄 Loading data for LSTM...")
    train_loader, val_loader, test_loader = trainer.load_data(
        sample_size=args.sample_size
    )

    print("🔄 Setting up LSTM model...")
    trainer.setup_model()

    print("🚀 Starting LSTM training...")
    start_time = time.time()
    results = trainer.train(train_loader, val_loader)
    training_time = time.time() - start_time

    # Test evaluation
    print("🔄 Evaluating LSTM on test set...")
    test_loss, test_mae = trainer.validate(test_loader)

    results.update(
        {
            "model_type": "LSTM",
            "test_loss": test_loss,
            "test_mae": test_mae,
            "total_training_time": training_time,
            "model_path": str(lstm_output_dir / "best_model.pth"),
            "config": config.__dict__,
        }
    )

    print(f"✅ LSTM training completed!")
    print(f"   Best val loss: {results['best_val_loss']:.4f}")
    print(f"   Test MAE: {test_mae:.2f}°C")
    print(f"   Training time: {training_time:.2f}s")
    print()

    return results


def train_cnn_model(args, output_dir: Path) -> Dict[str, Any]:
    """Train CNN model and return results."""
    print("🔬 Training CNN Model")
    print("=" * 60)

    config = create_cnn_config(args)
    cnn_output_dir = output_dir / "cnn"
    cnn_output_dir.mkdir(exist_ok=True)

    # Print CNN configuration
    print("CNN Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key:25}: {value}")
    print()

    trainer = CNNProteinTemperatureTrainer(config, str(cnn_output_dir))

    # Load data and train
    print("🔄 Loading data for CNN...")
    train_loader, val_loader, test_loader = trainer.load_data(
        sample_size=args.sample_size
    )

    print("🔄 Setting up CNN model...")
    trainer.setup_model()

    print("🚀 Starting CNN training...")
    start_time = time.time()
    results = trainer.train(train_loader, val_loader)
    training_time = time.time() - start_time

    # Test evaluation
    print("🔄 Evaluating CNN on test set...")
    test_loss, test_mae = trainer.validate(test_loader)

    results.update(
        {
            "model_type": "CNN",
            "test_loss": test_loss,
            "test_mae": test_mae,
            "total_training_time": training_time,
            "model_path": str(cnn_output_dir / "best_cnn_model.pth"),
            "config": config.__dict__,
        }
    )

    print(f"✅ CNN training completed!")
    print(f"   Best val loss: {results['best_val_loss']:.4f}")
    print(f"   Test MAE: {test_mae:.2f}°C")
    print(f"   Training time: {training_time:.2f}s")
    print()

    return results


def generate_comparison_report(
    lstm_results: Dict[str, Any], cnn_results: Dict[str, Any], output_dir: Path
):
    """Generate a comparison report between LSTM and CNN models."""
    print("📊 Model Comparison Report")
    print("=" * 80)

    # Performance comparison
    print("Performance Metrics:")
    print(f"{'Metric':<20} {'LSTM':<15} {'CNN':<15} {'Winner':<10}")
    print("-" * 65)

    # Validation loss
    lstm_val = lstm_results["best_val_loss"]
    cnn_val = cnn_results["best_val_loss"]
    val_winner = "LSTM" if lstm_val < cnn_val else "CNN"
    print(f"{'Val Loss':<20} {lstm_val:<15.4f} {cnn_val:<15.4f} {val_winner:<10}")

    # Test MAE
    lstm_mae = lstm_results["test_mae"]
    cnn_mae = cnn_results["test_mae"]
    mae_winner = "LSTM" if lstm_mae < cnn_mae else "CNN"
    print(f"{'Test MAE (°C)':<20} {lstm_mae:<15.2f} {cnn_mae:<15.2f} {mae_winner:<10}")

    # Training time
    lstm_time = lstm_results["total_training_time"]
    cnn_time = cnn_results["total_training_time"]
    time_winner = "LSTM" if lstm_time < cnn_time else "CNN"
    print(
        f"{'Training Time (s)':<20} {lstm_time:<15.2f} {cnn_time:<15.2f} {time_winner:<10}"
    )

    print()

    # Model architecture comparison
    print("Model Architecture:")
    print(f"{'Aspect':<20} {'LSTM':<30} {'CNN':<30}")
    print("-" * 85)

    # Get model info if available
    lstm_config = lstm_results["config"]
    cnn_config = cnn_results["config"]

    print(f"{'Input Type':<20} {'Sequential':<30} {'2D Grid + Positional':<30}")
    print(
        f"{'Hidden Size':<20} {lstm_config.get('lstm_hidden_size', 'N/A'):<30} {cnn_config.get('embedding_dim', 'N/A'):<30}"
    )
    lstm_layers = f"{lstm_config.get('lstm_num_layers', 'N/A')} LSTM + {lstm_config.get('fc_num_layers', 'N/A')} FC"
    cnn_layers = f"{len(cnn_config.get('cnn_channels', []))} CNN layers"
    print(f"{'Layers':<20} {lstm_layers:<30} {cnn_layers:<30}")
    print(
        f"{'Batch Size':<20} {lstm_config.get('batch_size', 'N/A'):<30} {cnn_config.get('batch_size', 'N/A'):<30}"
    )

    print()

    # Recommendations
    print("Recommendations:")
    if mae_winner == "LSTM" and time_winner == "LSTM":
        print("🏆 LSTM is the clear winner - better accuracy and faster training")
    elif mae_winner == "CNN" and time_winner == "CNN":
        print("🏆 CNN is the clear winner - better accuracy and faster training")
    elif mae_winner == "LSTM":
        print(
            "⚖️  LSTM has better accuracy but CNN trains faster - choose based on priority"
        )
    else:
        print(
            "⚖️  CNN has better accuracy but LSTM trains faster - choose based on priority"
        )

    # Save comparison report
    comparison_data = {
        "lstm_results": lstm_results,
        "cnn_results": cnn_results,
        "comparison": {
            "validation_loss_winner": val_winner,
            "test_mae_winner": mae_winner,
            "training_time_winner": time_winner,
            "overall_recommendation": mae_winner,  # Prioritize accuracy
        },
    }

    report_path = output_dir / "model_comparison.json"
    with open(report_path, "w") as f:
        json.dump(comparison_data, f, indent=2)

    print(f"📄 Detailed comparison saved to: {report_path}")
    print("=" * 80)


def main():
    """Main unified training function."""
    args = parse_arguments()

    # Set random seed
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Check device availability
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Silicon) acceleration available")
    elif torch.cuda.is_available():
        print("✅ CUDA acceleration available")
    else:
        print("⚠️  No GPU acceleration available, using CPU")

    print(f"🎯 Training {args.model_type.upper()} model(s)")
    print(f"📁 Output directory: {output_dir}")
    print()

    results = {}

    try:
        if args.model_type == "lstm":
            results["lstm"] = train_lstm_model(args, output_dir)

        elif args.model_type == "cnn":
            results["cnn"] = train_cnn_model(args, output_dir)

        elif args.model_type == "both":
            print("🔄 Training both LSTM and CNN models for comparison...")
            print()

            results["lstm"] = train_lstm_model(args, output_dir)
            results["cnn"] = train_cnn_model(args, output_dir)

            # Generate comparison report
            generate_comparison_report(results["lstm"], results["cnn"], output_dir)

        # Save overall results
        results_path = output_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"✅ All training completed successfully!")
        print(f"📊 Results saved to: {results_path}")

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
