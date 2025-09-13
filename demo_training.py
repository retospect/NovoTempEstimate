#!/usr/bin/env python3
"""
Demo script showing how to train the protein temperature prediction model.

This script demonstrates a quick training run with a small sample of data
to verify that the complete pipeline works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from novotempestimate.trainer import ProteinTemperatureTrainer, TrainingConfig


def main():
    """Run a demo training session."""
    print("🧬 Protein Temperature Prediction - Demo Training")
    print("=" * 50)
    
    # Create a configuration for quick demo training
    config = TrainingConfig(
        # Model parameters (as per paper)
        lstm_hidden_size=128,
        lstm_num_layers=2,
        fc_hidden_size=128,
        fc_num_layers=3,
        dropout=0.3,
        bidirectional=True,
        
        # Training parameters (reduced for demo)
        batch_size=16,
        learning_rate=0.001,
        num_epochs=5,  # Just 5 epochs for demo
        
        # Data parameters
        max_sequence_length=500,  # Shorter sequences for faster training
        train_split=0.8,
        val_split=0.2,
        test_split=0.0,  # Skip test split for demo
        
        # Early stopping
        early_stopping_patience=3,
        
        # Logging
        log_frequency=10,
        validate_frequency=1
    )
    
    print("Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    print()
    
    # Create trainer
    trainer = ProteinTemperatureTrainer(config, "demo_models")
    
    try:
        # Load a small sample of data for demo
        print("🔄 Loading demo data (sample of 1000 sequences)...")
        train_loader, val_loader, test_loader = trainer.load_data(
            data_pattern="*training*",  # Use training data which has temperature info
            sample_size=1000  # Small sample for demo
        )
        
        print("🔄 Setting up model...")
        trainer.setup_model()
        
        print("🚀 Starting demo training (5 epochs)...")
        results = trainer.train(train_loader, val_loader)
        
        print("\n" + "=" * 50)
        print("🎉 Demo Training Results:")
        print(f"✅ Best validation loss: {results['best_val_loss']:.4f}")
        print(f"⏱️  Training time: {results['training_time']:.1f} seconds")
        print(f"📊 Final epoch: {results['final_epoch']}")
        print("=" * 50)
        
        print(f"\n📁 Demo model saved to: demo_models/")
        print("🔍 You can now use this model for predictions!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Demo completed successfully!")
        print("\n📖 Next steps:")
        print("1. Run full training: python scripts/train_model.py --epochs 50 --sample-size 10000")
        print("2. Evaluate model: python scripts/evaluate_model.py --model-path demo_models/best_model.pth")
        print("3. Make predictions: python scripts/predict_temperature.py --model-path demo_models/best_model.pth --sequence 'MKLLVL...'")
    else:
        print("\n❌ Demo failed - check the error messages above")
        sys.exit(1)
